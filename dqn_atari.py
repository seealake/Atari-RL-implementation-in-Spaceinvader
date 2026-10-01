#!/usr/bin/env python
"""Run Atari Environment with DQN."""
"""Supports: LINEAR, DQN, DOUBLE DQN, DUELING DQN"""
import argparse
from contextlib import ExitStack
from dataclasses import asdict, dataclass, fields, replace
import json
import os
from pathlib import Path
import random
import sys
import time

if not (3, 11) <= sys.version_info[:2] < (3, 14):
    raise SystemExit('Use 64-bit Python 3.11-3.13 (recommended: Python 3.12). See README.md.')

import numpy as np
import tensorflow as tf
import gymnasium as gym
import ale_py
from gymnasium.wrappers import RecordVideo
import deeprl as tfrl
from deeprl.dqn import DQNAgent
from deeprl.policy import GreedyEpsilonPolicy, ExponentialDecayGreedyEpsilonPolicy
from deeprl.preprocessors import AtariPreprocessor, HistoryPreprocessor, PreprocessorSequence
from matplotlib.figure import Figure

gym.register_envs(ale_py)


@dataclass
class TrainingConfig:
    """Shared defaults for the CLI, saved experiments and short smoke tests."""

    env: str = 'ALE/SpaceInvaders-v5'
    mode: str = 'deep'
    output: str = 'atari-v0'
    seed: int = 0
    iterations: int = 1000000
    frame_size: int = 84
    history_length: int = 4
    hidden_units: int = 512
    memory_size: int = 500000
    batch_size: int = 64
    burn_in: int = 50000
    gamma: float = 0.99
    learning_rate: float = 3e-4
    train_freq: int = 4
    target_update_freq: int = 5000
    tau: float = 0.001
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 1e-5
    reward_clip: float = 1.0
    gradient_clip: float = 5.0
    checkpoint_dir: str = 'checkpoints'
    checkpoint_freq: int = 50000
    keep_checkpoints: int = 2
    restart_freq: int = 500000
    eval_freq: int = 10000
    eval_episodes: int = 10
    final_eval_episodes: int = 100
    max_episode_length: int = 27000
    record_video: bool = True
    video_freq: int = 100
    video_length: int = 1000
    threads: int = 0
    log_freq: int = 1000

    def smoke_test(self):
        limits = dict(iterations=128, memory_size=256, batch_size=8, burn_in=16,
                      target_update_freq=32, checkpoint_freq=64, eval_freq=64,
                      eval_episodes=1, final_eval_episodes=1, max_episode_length=64,
                      video_length=64, log_freq=64)
        return replace(self, **{name: min(getattr(self, name), cap) for name, cap in limits.items()},
                       threads=min(self.threads or 2, 2))

    def validate(self):
        nonnegative = {'seed', 'burn_in', 'checkpoint_freq', 'restart_freq', 'eval_freq',
                       'eval_episodes', 'final_eval_episodes', 'threads'}
        for field in fields(self):
            value = getattr(self, field.name)
            if field.type is int and (type(value) is not int or value < (0 if field.name in nonnegative else 1)):
                raise ValueError(f'Invalid value for {field.name}: {value}')
            if field.type is float and (not np.isfinite(value) or value < 0):
                raise ValueError(f'Invalid value for {field.name}: {value}')
        if not 0 <= self.gamma <= 1 or not 0 < self.tau <= 1:
            raise ValueError('gamma must be in [0, 1] and tau in (0, 1]')
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError('Require 0 <= epsilon_end <= epsilon_start <= 1')
        if min(self.learning_rate, self.reward_clip, self.gradient_clip) <= 0:
            raise ValueError('Learning rate and clipping thresholds must be positive')
        if self.memory_size < max(self.batch_size, self.history_length + 1):
            raise ValueError('memory_size must accommodate batch_size and frame history')
        path = Path(self.checkpoint_dir)
        if path.is_absolute() or '..' in path.parts or path == Path('.'):
            raise ValueError('checkpoint_dir must be a subdirectory of the run folder')


def create_optimizer():
    """Create Adam optimizer with standard learning rate."""
    return tf.keras.optimizers.Adam(learning_rate=3e-4)

def create_linear_model(input_shape, num_actions, model_name='linear_q_network'):
    """Create a linear Q-network."""
    num_actions = int(num_actions)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_actions, activation=None)  # Linear output layer
    ])
    model.compile(optimizer=create_optimizer(), loss='mse')
    return model

def create_deep_q_network(input_shape, num_actions, model_name='deep_q_network', hidden_units=TrainingConfig.hidden_units):
    """Create a deep Q-network with CNN architecture."""
    num_actions = int(num_actions)
    inputs = tf.keras.Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    x = tf.keras.layers.Conv2D(32, (8, 8), strides=4, activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(64, (4, 4), strides=2, activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), strides=1, activation='relu')(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(hidden_units, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_actions, activation=None)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    model.compile(optimizer=create_optimizer(), loss='huber') 
    return model

def create_dueling_q_network(input_shape, num_actions, model_name='dueling_q_network', hidden_units=TrainingConfig.hidden_units):
    """Create a dueling deep Q-network."""
    num_actions = int(num_actions)
    inputs = tf.keras.Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    x = tf.keras.layers.Conv2D(32, (8, 8), strides=4, activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(64, (4, 4), strides=2, activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), strides=1, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    value_stream = tf.keras.layers.Dense(hidden_units, activation='relu')(x)
    value = tf.keras.layers.Dense(1)(value_stream)
    advantage_stream = tf.keras.layers.Dense(hidden_units, activation='relu')(x)
    advantage = tf.keras.layers.Dense(num_actions)(advantage_stream)
    q_values = value + (advantage - tf.keras.ops.mean(advantage, axis=1, keepdims=True))
    model = tf.keras.Model(inputs=inputs, outputs=q_values, name=model_name)
    model.compile(optimizer=create_optimizer(), loss='huber')
    return model

def make_env(env_name, output_directory=None, record_video=True, *, video_freq=TrainingConfig.video_freq,
             video_length=TrainingConfig.video_length, max_episode_length=TrainingConfig.max_episode_length):
    env = gym.make(env_name, render_mode='rgb_array' if record_video else None,
                   max_episode_steps=max_episode_length)
    try:
        if record_video:
            if output_directory is None:
                raise ValueError('Video recording requires an output directory')
            video_directory = Path(output_directory) / f'session-{time.time_ns()}'
            env = RecordVideo(env, video_folder=str(video_directory),
                              episode_trigger=lambda episode: episode % video_freq == 0,
                              video_length=video_length)
        return env
    except Exception:
        env.close()
        raise

def get_output_folder(parent_dir, env_name):
    """Return save folder."""
    # Sanitize env_name to be a valid folder name (replace slashes with underscores)
    safe_env_name = env_name.replace('/', '_').replace('\\', '_')
    
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, safe_env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    return parent_dir

def plot_learning_curve(evaluation_results, output_dir, mode):
    if not evaluation_results:
        print("No evaluation results to plot")
        return
    
    steps, means, stds = zip(*evaluation_results)
    figure = Figure(figsize=(10, 6))
    axes = figure.subplots()
    axes.plot(steps, means)
    axes.fill_between(steps, [m-s for m,s in zip(means, stds)], [m+s for m,s in zip(means, stds)], alpha=0.2)
    axes.set(title=f'Learning Curve for {mode.capitalize()} Q-Network', xlabel='Steps', ylabel='Mean Reward')
    os.makedirs(output_dir, exist_ok=True)
    figure.savefig(os.path.join(output_dir, f'{mode}_learning_curve.png'))

def create_final_evaluation_table(results):
    """Create a table with final evaluation results for all models."""
    table = "Model Type | Average Total Reward (100 episodes)\n"
    table += "----------|------------------------------------\n"
    for model, (mean, std) in results.items():
        table += f"{model:10} | {mean:.2f} +/- {std:.2f}\n"
    return table

def build_parser():
    parser = argparse.ArgumentParser(description='Train DQN on Atari Space Invaders')
    descriptions = {
        'iterations': 'Additional environment steps to train',
        'burn_in': 'Completed steps before gradient updates start',
        'memory_size': 'Replay capacity in single uint8 frames',
        'tau': 'Target update weight (1 = periodic hard updates)',
        'threads': 'TensorFlow CPU threads (0 = automatic)',
        'restart_freq': 'Optional environment-reset interval; weights are retained (0 = off)',
        'checkpoint_freq': 'Checkpoint interval (0 = final checkpoint only)',
        'eval_freq': 'Periodic evaluation interval (0 = off)',
        'video_length': 'Maximum frames per recorded clip',
    }
    for field in fields(TrainingConfig):
        options = list(dict.fromkeys(('--' + field.name.replace('_', '-'), '--' + field.name)))
        if field.name == 'output':
            options.append('-o')
        kwargs = dict(default=field.default, help=descriptions.get(field.name, field.name.replace('_', ' ')))
        kwargs['help'] += f' (default: {field.default})'
        if field.type is bool:
            kwargs['action'] = argparse.BooleanOptionalAction
        else:
            kwargs['type'] = field.type
        if field.name == 'mode':
            kwargs['choices'] = ['linear', 'linear_double', 'deep', 'double', 'dueling']
        parser.add_argument(*options, **kwargs)
    parser.add_argument('--smoke-test', '--smoke_test', action='store_true',
                        help='Cap training at 128 steps, use small replay and at most 2 CPU threads')
    parser.add_argument('--resume', type=Path, help='Existing run directory to resume')
    parser.add_argument('--checkpoint-file', '--checkpoint_file', type=Path,
                        help='Specific trusted TensorFlow checkpoint prefix, directory or sidecar')
    parser.add_argument('--start-step', '--start_step', type=int, default=0,
                        help='Optional expected resume step; requires a checkpoint')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    run_dir, saved_config = None, None
    source = args.resume or args.checkpoint_file
    try:
        if source is not None:
            source = source.expanduser().resolve()
            if args.resume and not (source / 'config.json').is_file():
                raise ValueError('--resume expects a run directory; use --checkpoint-file for a specific checkpoint')
            run_dir = next((path for path in (source, source.parent, source.parent.parent)
                            if (path / 'config.json').is_file()), None)
            if args.resume and run_dir is None:
                raise ValueError('--resume must point to an existing run containing config.json')
            if run_dir is not None:
                with (run_dir / 'config.json').open(encoding='utf-8') as handle:
                    saved_config = json.load(handle)
                parser.set_defaults(**{field.name: saved_config[field.name] for field in fields(TrainingConfig)
                                       if field.name in saved_config})
                args = parser.parse_args(argv)
        config = TrainingConfig(**{field.name: getattr(args, field.name) for field in fields(TrainingConfig)})
        if args.smoke_test:
            config = config.smoke_test()
        config.validate()
        if args.start_step < 0 or (args.start_step and source is None):
            raise ValueError('--start-step requires a checkpoint; it cannot skip untrained steps')
        if saved_config is not None:
            fixed = ('env', 'mode', 'frame_size', 'history_length', 'hidden_units',
                     'memory_size', 'batch_size', 'burn_in', 'gamma', 'learning_rate',
                     'train_freq', 'target_update_freq', 'tau', 'epsilon_start',
                     'epsilon_end', 'epsilon_decay', 'reward_clip', 'gradient_clip', 'checkpoint_dir')
            for name in fixed:
                if name in saved_config and getattr(config, name) != saved_config[name]:
                    raise ValueError(f'Cannot change {name} when resuming this run')

        checkpoint = None
        if source is not None:
            checkpoint = args.checkpoint_file or (run_dir / config.checkpoint_dir)
            checkpoint = checkpoint.expanduser()
            if args.checkpoint_file and run_dir is not None and not checkpoint.is_absolute():
                candidate = run_dir / config.checkpoint_dir / checkpoint
                if candidate.exists() or Path(str(candidate) + '.index').is_file():
                    checkpoint = candidate
            if checkpoint.is_dir():
                latest = tf.train.latest_checkpoint(str(checkpoint))
                if latest is None:
                    raise FileNotFoundError(f'No checkpoint found in {checkpoint}')
                checkpoint = Path(latest)
            elif not checkpoint.is_file() and not Path(str(checkpoint) + '.index').is_file():
                raise FileNotFoundError(f'Checkpoint does not exist: {checkpoint}')

        if config.threads:
            tf.config.threading.set_intra_op_parallelism_threads(config.threads)
            tf.config.threading.set_inter_op_parallelism_threads(config.threads)
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"TensorFlow {tf.__version__}; device: {'GPU' if gpus else 'CPU'}")
        np.random.seed(config.seed)
        tf.random.set_seed(config.seed)
        random.seed(config.seed)

        output_dir = run_dir or Path(get_output_folder(os.path.join(config.output, config.mode), config.env))
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = output_dir / config.checkpoint_dir
        input_shape = (config.frame_size, config.frame_size, config.history_length)
        with ExitStack() as resources:
            env = make_env(config.env, output_dir / 'videos', config.record_video,
                           video_freq=config.video_freq, video_length=config.video_length,
                           max_episode_length=config.max_episode_length)
            resources.callback(env.close)
            eval_env = make_env(config.env, record_video=False, max_episode_length=config.max_episode_length)
            resources.callback(eval_env.close)
            if config.mode in ('linear', 'linear_double'):
                model = create_linear_model(input_shape, env.action_space.n)
            elif config.mode in ('deep', 'double'):
                model = create_deep_q_network(input_shape, env.action_space.n, hidden_units=config.hidden_units)
            else:
                model = create_dueling_q_network(input_shape, env.action_space.n, hidden_units=config.hidden_units)

            policy = ExponentialDecayGreedyEpsilonPolicy(
                GreedyEpsilonPolicy(config.epsilon_start), config.epsilon_start,
                config.epsilon_end, config.epsilon_decay,
            )
            capacity = config.memory_size if checkpoint is None else max(config.batch_size, config.history_length + 1)
            memory = tfrl.core.ReplayMemory(capacity, config.frame_size, config.frame_size, config.history_length)
            preprocessor = PreprocessorSequence([
                AtariPreprocessor(new_size=(config.frame_size, config.frame_size)),
                HistoryPreprocessor(history_length=config.history_length),
            ])
            agent = DQNAgent(
                model=model, input_shape=input_shape, num_actions=env.action_space.n,
                preprocessor=preprocessor, memory=memory, policy=policy,
                gamma=config.gamma, target_update_freq=config.target_update_freq,
                num_burn_in=config.burn_in, train_freq=config.train_freq, batch_size=config.batch_size,
                double_q=config.mode in ('linear_double', 'double'), dueling=config.mode == 'dueling',
                tau=config.tau, checkpoint_dir=checkpoint_dir, output_dir=output_dir,
                learning_rate=config.learning_rate, keep_checkpoints=config.keep_checkpoints,
                reward_clip=config.reward_clip, gradient_clip=config.gradient_clip,
            )
            resources.callback(agent.close)
            if checkpoint is not None:
                agent.load_checkpoint(checkpoint, step=args.start_step or None)
                if agent.memory is memory:
                    agent.memory = tfrl.core.ReplayMemory(
                        config.memory_size, config.frame_size, config.frame_size, config.history_length
                    )

            with (output_dir / 'config.json').open('w', encoding='utf-8') as handle:
                json.dump(asdict(config), handle, indent=2)
            print(f'Run directory: {output_dir.resolve()}')
            print(f'Replay frame capacity: {agent.memory.frames.nbytes / 1024 ** 2:.1f} MiB')
            try:
                evaluation_results, final_result, _ = agent.fit(
                    env, num_iterations=agent.steps + config.iterations, start_step=agent.steps,
                    max_episode_length=config.max_episode_length, checkpoint_freq=config.checkpoint_freq,
                    restart_freq=config.restart_freq, eval_env=eval_env, eval_freq=config.eval_freq,
                    eval_episodes=config.eval_episodes, final_eval_episodes=config.final_eval_episodes,
                    seed=config.seed, log_freq=config.log_freq,
                )
            except KeyboardInterrupt:
                agent.save_checkpoint(agent.steps, agent.evaluation_results)
                print(f'Training interrupted at step {agent.steps}; checkpoint saved.')
                return 130
            plot_learning_curve(evaluation_results, output_dir, config.mode)
            with (output_dir / 'final_results.txt').open('w', encoding='utf-8') as handle:
                handle.write(f'Mode: {config.mode}\nCompleted steps: {agent.steps}\n')
                handle.write(f'Gradient updates: {int(agent.optimizer.iterations.numpy())}\n')
                if final_result is not None:
                    mean, std = final_result
                    handle.write(f'Final Mean Reward: {mean:.2f}\nFinal Std Reward: {std:.2f}\n')
        return 0
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
