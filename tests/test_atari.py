import hashlib
from pathlib import Path
import subprocess
import sys

from moviepy import VideoFileClip
import numpy as np
import pytest
import tensorflow as tf

from dqn_atari import create_deep_q_network, create_dueling_q_network, create_linear_model, make_env


@pytest.mark.integration
@pytest.mark.parametrize('mode', ['linear', 'linear_double', 'deep', 'double', 'dueling'])
def test_all_modes_train_on_real_atari_and_reload(agent_factory, mode):
    env = make_env('ALE/SpaceInvaders-v5', record_video=False, max_episode_length=8)
    evaluation = make_env('ALE/SpaceInvaders-v5', record_video=False, max_episode_length=8)
    try:
        shape = (40, 40, 4)
        if mode in ('linear', 'linear_double'):
            model = create_linear_model(shape, env.action_space.n)
        elif mode == 'dueling':
            model = create_dueling_q_network(shape, env.action_space.n, hidden_units=16)
        else:
            model = create_deep_q_network(shape, env.action_space.n, hidden_units=16)
        agent = agent_factory(model=model, input_shape=shape, num_actions=env.action_space.n,
                              double_q=mode in ('double', 'linear_double'), dueling=mode == 'dueling')
        before = [value.copy() for value in agent.q_network.get_weights()]
        agent.fit(env, 6, eval_env=evaluation, eval_freq=3, eval_episodes=1,
                  final_eval_episodes=1, max_episode_length=8, restart_freq=3)
        assert int(agent.optimizer.iterations.numpy()) == 5
        assert np.isfinite(agent.losses).all()
        assert any(not np.array_equal(a, b) for a, b in zip(before, agent.q_network.get_weights()))
        loaded = tf.keras.models.load_model(agent.output_dir / 'final_model.keras')
        sample = np.ones((1, *shape), dtype=np.float32)
        np.testing.assert_allclose(loaded(sample).numpy(), agent.q_network(sample).numpy())
        restored = agent_factory(model=tf.keras.models.clone_model(agent.q_network), input_shape=shape,
                                 num_actions=env.action_space.n, double_q=agent.double_q, dueling=agent.dueling)
        assert restored.load_checkpoint(agent.checkpoint_manager.latest_checkpoint) == 6
        for before, after in zip(agent.target_network.weights, restored.target_network.weights, strict=True):
            np.testing.assert_array_equal(before.numpy(), after.numpy())
        np.testing.assert_allclose(restored.q_network(sample).numpy(), loaded(sample).numpy())
        assert int(restored.optimizer.iterations.numpy()) == 5
    finally:
        env.close()
        evaluation.close()


@pytest.mark.integration
def test_cli_video_and_resume(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'dqn_atari.py'

    def run(*arguments):
        result = subprocess.run([sys.executable, str(script), *arguments], text=True,
                                capture_output=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr

    run('--smoke-test', '--iterations', '32', '--frame-size', '40', '--hidden-units', '16',
        '--batch-size', '2', '--burn-in', '4', '--max-episode-length', '8',
        '--eval-freq', '16', '--checkpoint-freq', '16', '--restart-freq', '8',
        '--output', str(tmp_path))
    directory = next((tmp_path / 'deep').glob('*-run*'))
    report = (directory / 'final_results.txt').read_text()
    assert 'Completed steps: 32\n' in report and 'Gradient updates: 8\n' in report
    videos = list((directory / 'videos').rglob('*.mp4'))
    assert videos
    checksums = {video: hashlib.sha256(video.read_bytes()).hexdigest() for video in videos}
    with VideoFileClip(str(videos[0]), audio=False) as clip:
        first = clip.get_frame(0)
        last = clip.get_frame(clip.duration - 1 / clip.fps)
        assert first.shape == (210, 160, 3)
        assert first.std() > 0 and clip.duration * clip.fps >= 2
        assert not np.array_equal(first, last)
    run('--resume', str(directory), '--iterations', '16')
    report = (directory / 'final_results.txt').read_text()
    assert 'Completed steps: 48\n' in report and 'Gradient updates: 12\n' in report
    for video, checksum in checksums.items():
        assert hashlib.sha256(video.read_bytes()).hexdigest() == checksum
    assert len(list((directory / 'videos').rglob('*.mp4'))) > len(videos)
    assert len(list((directory / 'checkpoints').glob('*.index'))) == 2
