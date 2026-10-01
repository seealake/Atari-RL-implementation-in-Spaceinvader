import random
import pickle
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import tensorflow as tf

from dqn_atari import TrainingConfig, build_parser, get_output_folder, main


class CountingEnv(gym.Env):
    def __init__(self, episode_length=100):
        self.action_space = gym.spaces.Discrete(2)
        self.episode_length = episode_length
        self.resets, self.steps = 0, 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.resets += 1
        self.steps = 0
        return np.full((4, 4, 3), 10, dtype=np.uint8), {}

    def step(self, action):
        self.steps += 1
        return np.full((4, 4, 3), 10 + self.steps, dtype=np.uint8), 1.0, False, self.steps >= self.episode_length, {}


@pytest.mark.parametrize('double_q,expected_loss', [(False, 0.5), (True, 0.25)])
def test_dqn_targets_and_terminal_mask(agent_factory, monkeypatch, double_q, expected_loss):
    agent = agent_factory(double_q=double_q, gamma=0.5)
    weights = agent.q_network.get_weights()
    weights[0][:] = 0
    weights[1][:] = [1, 3]
    agent.q_network.set_weights(weights)
    weights[1][:] = [4, 2]
    agent.target_network.set_weights(weights)
    for _ in range(2):
        agent.memory.append(np.zeros((4, 4), dtype=np.uint8), 0, 0, True)
    states = np.zeros((2, 4, 4, 2), dtype=np.float32)
    monkeypatch.setattr(agent.memory, 'sample', lambda _: (
        states, np.zeros(2, dtype=np.int32), np.zeros(2, dtype=np.float32),
        states, np.array([0, 1], dtype=np.float32),
    ))
    assert float(agent.update_policy().numpy()) == pytest.approx(expected_loss)


def test_fit_keeps_training_separate_from_evaluation_and_restart(agent_factory):
    agent = agent_factory()
    training, evaluation = CountingEnv(), CountingEnv(3)
    results, final, steps = agent.fit(
        training, 6, eval_env=evaluation, eval_freq=2, eval_episodes=1,
        final_eval_episodes=1, restart_freq=3, checkpoint_freq=2, log_freq=2,
    )
    assert steps == 6 and agent.policy.step == 6
    assert int(agent.optimizer.iterations.numpy()) == 5
    assert training.resets == 2 and training.steps == 3
    assert [step for step, _, _ in results] == [2, 4, 6]
    assert final == (3.0, 0.0)
    assert len(list(agent.checkpoint_dir.glob('extra_data_*.pkl.gz'))) == 2


def test_evaluation_preserves_frame_history(agent_factory):
    agent = agent_factory()
    before = agent.preprocessor.process_state_for_network(np.ones((4, 4, 3), dtype=np.uint8))
    agent.evaluate(CountingEnv(2), 1)
    history = agent.preprocessor.preprocessors[-1]
    np.testing.assert_array_equal(np.stack(history.history, axis=-1), before)
    assert agent.policy.step == 0


def test_checkpoint_roundtrip_restores_target_optimizer_policy_and_rng(agent_factory):
    original = agent_factory(double_q=True)
    for step in range(10):
        original.memory.append(np.full((4, 4), step, dtype=np.uint8), step % 2, 1, False,
                               np.full((4, 4), step + 1, dtype=np.uint8))
    original.record_loss(float(original.update_policy().numpy()))
    original.steps = original.policy.step = 10
    original.target_network.set_weights([value + 0.25 for value in original.target_network.get_weights()])
    checkpoint = original.save_checkpoint(10, [(8, 2.0, 0.5)])
    expected_random = np.random.rand(3)
    expected_python = random.random()
    restored = agent_factory(double_q=True)
    assert restored.load_checkpoint(checkpoint) == 10
    for left, right in ((original.q_network.weights, restored.q_network.weights),
                        (original.target_network.weights, restored.target_network.weights),
                        (original.optimizer.variables, restored.optimizer.variables)):
        for source, target in zip(left, right, strict=True):
            np.testing.assert_array_equal(source.numpy(), target.numpy())
    np.testing.assert_array_equal(np.random.rand(3), expected_random)
    assert random.random() == expected_python
    assert restored.policy.get_config() == original.policy.get_config()
    assert restored.evaluation_results == [(8, 2.0, 0.5)]
    assert restored.losses == original.losses
    assert len(restored.memory) == 10
    restored.update_policy()
    assert int(restored.optimizer.iterations.numpy()) == 2


def test_missing_checkpoint_does_not_silently_start_over(agent_factory, tmp_path):
    with pytest.raises(FileNotFoundError):
        agent_factory().load_checkpoint(tmp_path / 'missing')


def test_legacy_weight_checkpoint_still_loads(agent_factory, tmp_path):
    original = agent_factory()
    checkpoint = tmp_path / 'legacy.pkl'
    with checkpoint.open('wb') as handle:
        pickle.dump(dict(step=12, q_network_weights=original.q_network.get_weights(),
                         target_network_weights=original.target_network.get_weights(),
                         optimizer_weights=[value.numpy() for value in original.optimizer.variables],
                         memory=original.memory, policy_state=original.policy.get_config()), handle)
    restored = agent_factory()
    assert restored.load_checkpoint(checkpoint) == 12
    for before, after in zip(original.q_network.weights, restored.q_network.weights, strict=True):
        np.testing.assert_array_equal(before.numpy(), after.numpy())


@pytest.mark.parametrize('changes', [dict(iterations=0), dict(train_freq=0), dict(memory_size=4),
                                     dict(tau=0), dict(epsilon_end=2), dict(threads=-1)])
def test_config_rejects_invalid_values(changes):
    with pytest.raises(ValueError):
        TrainingConfig(**changes).validate()


def test_default_cli_and_bounded_smoke_config():
    assert build_parser().parse_args([]).mode == 'deep'
    assert not build_parser().parse_args(['--no-record-video']).record_video
    config = TrainingConfig().smoke_test()
    config.validate()
    assert config.iterations == 128 and config.threads <= 2 and config.memory_size == 256


def test_start_step_cannot_skip_training():
    with pytest.raises(SystemExit) as error:
        main(['--start-step', '100'])
    assert error.value.code != 0


def test_missing_cli_checkpoint_fails_before_allocating_replay(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(['--checkpoint-file', str(tmp_path / 'missing')])
    assert error.value.code != 0
    assert not list(tmp_path.iterdir())


def test_resume_sources_are_mutually_exclusive():
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(['--resume', 'run-a', '--checkpoint-file', 'run-b/ckpt-10'])
    assert error.value.code != 0


def test_concurrent_runs_reserve_distinct_directories(tmp_path):
    with ThreadPoolExecutor(max_workers=4) as pool:
        folders = list(pool.map(lambda _: get_output_folder(tmp_path, 'ALE/SpaceInvaders-v5'), range(4)))
    assert len(set(folders)) == 4
    assert all(Path(folder).is_dir() for folder in folders)


@pytest.mark.parametrize('parent_config', [{'iterations': 17}, asdict(TrainingConfig(iterations=17))])
def test_checkpoint_ignores_unrelated_parent_config(tmp_path, monkeypatch, parent_config):
    (tmp_path / 'config.json').write_text(json.dumps(parent_config), encoding='utf-8')
    seen_iterations = []

    def stop_before_training(config):
        seen_iterations.append(config.iterations)
        raise ValueError('Stop before allocating training resources')

    monkeypatch.setattr(TrainingConfig, 'validate', stop_before_training)
    with pytest.raises(SystemExit):
        main(['--checkpoint-file', str(tmp_path / 'other_exports' / 'ckpt-1')])
    assert seen_iterations == [TrainingConfig.iterations]
