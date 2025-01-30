#!/usr/bin/env python
"""Yes. Run Atari Environment with DQN."""
"""LINEAR, DQN, DOUBLE DQN, DUELING DQN"""
import argparse
import os
import random
import numpy as np
import tensorflow as tf
import gym
from gym.wrappers import RecordVideo
import deeprl_hw2 as tfrl
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.objectives import mean_huber_loss
from deeprl_hw2.policy import LinearDecayGreedyEpsilonPolicy, GreedyEpsilonPolicy, ExponentialDecayGreedyEpsilonPolicy
from deeprl_hw2.preprocessors import AtariPreprocessor, HistoryPreprocessor, PreprocessorSequence
import matplotlib.pyplot as plt
import gc
import pandas as pd

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(f"Error enabling GPU memory growth: {e}")

def create_optimizer():
    return tf.keras.optimizers.RMSprop(
        learning_rate=3e-4,
        rho=0.95,
        momentum=0.0,
        epsilon=0.00001,
        centered=True
    )

def create_linear_model(input_shape, num_actions, model_name='linear_q_network'):
    """Create a linear Q-network."""
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(num_actions, activation=None)  # Linear output layer
    ])
    model.compile(optimizer=create_optimizer(), loss='mse')
    return model

def create_deep_q_network(input_shape, num_actions, model_name='deep_q_network'):
    inputs = tf.keras.Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    x = tf.keras.layers.Conv2D(16, (8, 8), strides=4, activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(32, (4, 4), strides=2, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_actions, activation=None)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    model.compile(optimizer=create_optimizer(), loss=mean_huber_loss)
    return model

def create_dueling_q_network(input_shape, num_actions, model_name='dueling_q_network'):
    """Create a dueling deep Q-network."""
    inputs = tf.keras.Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    x = tf.keras.layers.Conv2D(32, (8, 8), strides=4, activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(64, (4, 4), strides=2, activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), strides=1, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    value_stream = tf.keras.layers.Dense(512, activation='relu')(x)
    value = tf.keras.layers.Dense(1)(value_stream)
    advantage_stream = tf.keras.layers.Dense(512, activation='relu')(x)
    advantage = tf.keras.layers.Dense(num_actions)(advantage_stream)
    q_values = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
    model = tf.keras.Model(inputs=inputs, outputs=q_values, name=model_name)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00025, rho=0.95, epsilon=0.01)    
    model.compile(optimizer=optimizer, loss='mse')
    return model

def make_env(env_name, output_directory):
    env = gym.make(env_name)
    env = RecordVideo(env, video_folder=output_directory, episode_trigger=lambda x: True)
    return env

def get_output_folder(parent_dir, env_name):
    """Return save folder."""
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

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    return parent_dir

def plot_learning_curve(evaluation_results, output_dir, mode):
    steps, means, stds = zip(*evaluation_results)
    plt.figure(figsize=(10, 6))
    plt.plot(steps, means)
    plt.fill_between(steps, [m-s for m,s in zip(means, stds)], [m+s for m,s in zip(means, stds)], alpha=0.2)
    plt.title(f'Learning Curve for {mode.capitalize()} Q-Network')
    plt.xlabel('Steps')
    plt.ylabel('Mean Reward')
    plt.savefig(os.path.join(output_dir, f'{mode}_learning_curve.png'))
    plt.close()

def create_final_evaluation_table(results):
    """Create a table with final evaluation results for all models."""
    table = "Model Type | Average Total Reward (100 episodes)\n"
    table += "----------|------------------------------------\n"
    for model, (mean, std) in results.items():
        table += f"{model:10} | {mean:.2f} +/- {std:.2f}\n"
    return table

def main():
    try:
        parser = argparse.ArgumentParser(description='Run DQN on Atari Space Invaders')
        parser.add_argument('--env', default='SpaceInvaders-v0', help='Atari env name')
        parser.add_argument('-o', '--output', default='atari-v0', help='Directory to save data to')
        parser.add_argument('--seed', default=0, type=int, help='Random seed')
        parser.add_argument('--mode', required=True, choices=['linear', 'linear_double', 'deep', 'double', 'dueling'],
                            help='Type of Q-network to train')
        parser.add_argument('--iterations', type=int, default=1000000, help='Number of training iterations')
        parser.add_argument('--checkpoint_dir', default='checkpoints', help='Directory to save checkpoints')
        parser.add_argument('--checkpoint_freq', type=int, default=50000, help='Frequency of saving checkpoints')
        parser.add_argument('--restart_freq', type=int, default=500000, help='Frequency of restarting the training')
        parser.add_argument('--start_step', type=int, default=0, help='Step to start or resume training from')
        parser.add_argument('--checkpoint_file', type=str, help='Specific checkpoint file to load') 
        args = parser.parse_args()
        
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
        random.seed(args.seed)

        output_dir = get_output_folder(os.path.join(args.output, args.mode), args.env)
        checkpoint_dir = os.path.join(output_dir, args.checkpoint_dir)
        output_directory = os.path.join(output_dir, "videos")
        env = make_env(args.env, output_directory)
        num_actions = env.action_space.n
        input_shape = (84, 84, 4)

        total_iterations = args.iterations
        iterations_done = 0
        all_evaluation_results = []

        while iterations_done < total_iterations:
            if args.mode == 'linear' or args.mode == 'linear_double':
                model = create_linear_model(input_shape, num_actions)
            elif args.mode == 'deep' or args.mode == 'double':
                model = create_deep_q_network(input_shape, num_actions)
            elif args.mode == 'dueling':
                model = create_dueling_q_network(input_shape, num_actions)

            atari_preprocessor = AtariPreprocessor(new_size=(84, 84))
            history_preprocessor = HistoryPreprocessor(history_length=4)
            preprocessor = PreprocessorSequence([atari_preprocessor, history_preprocessor])
            
            policy = ExponentialDecayGreedyEpsilonPolicy(
                GreedyEpsilonPolicy(1.0),
                start_value=1.0,
                end_value=0.05,
                decay_rate=1e-6
            )

            agent = DQNAgent(
                model=model,
                input_shape=input_shape,
                num_actions=num_actions,
                preprocessor=preprocessor,
                memory=tfrl.core.ReplayMemory(max_size=500000, frame_height=84, frame_width=84, history_length=4),
                policy=policy,
                gamma=0.99,
                target_update_freq=10000,
                num_burn_in=50000,
                train_freq=4,
                batch_size=16,
                double_q=args.mode in ['linear_double', 'double'],
                dueling=args.mode == 'dueling',
                tau=0.001,
                checkpoint_dir=checkpoint_dir
            )

            if args.checkpoint_file:
                checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_file)
                if os.path.exists(checkpoint_path):
                    step = agent.load_checkpoint(checkpoint_path)
                    args.start_step = step
                    print(f"Resumed training from checkpoint: {checkpoint_path}")
                else:
                    print(f"Specified checkpoint not found: {checkpoint_path}")
                    args.start_step = 0
            elif args.start_step > 0:
                latest_tf_checkpoint = tf.train.latest_checkpoint(args.checkpoint_dir)
                if latest_tf_checkpoint:
                    step = int(latest_tf_checkpoint.split('_')[-1])
                    agent.load_checkpoint(args.checkpoint_dir, step)
                    args.start_step = step
                    print(f"Resumed training from step {args.start_step}")
                else:
                    print(f"No checkpoint found, starting from beginning")
                    args.start_step = 0

            evaluation_results, final_result = agent.fit(
                env, 
                num_iterations=total_iterations - iterations_done,
                start_step=args.start_step,
                max_episode_length=None,
                checkpoint_dir=checkpoint_dir,
                checkpoint_freq=args.checkpoint_freq,
                restart_freq=args.restart_freq
            )

            all_evaluation_results.extend(evaluation_results)

            if final_result is None:
                iterations_done += evaluation_results[-1][0] - args.start_step
                args.start_step = evaluation_results[-1][0]
                print(f"Restarting training from step {args.start_step}")
            else:
                break

            del agent
            gc.collect()
            tf.keras.backend.clear_session()

        plot_learning_curve(all_evaluation_results, output_dir, args.mode)
        final_mean_reward, final_std_reward = final_result if final_result else (None, None)
        
        if final_mean_reward is not None and final_std_reward is not None:
            print(f"Final Result for {args.mode} Q-network:")
            print(f"Mean Reward: {final_mean_reward:.2f} +/- {final_std_reward:.2f}")
            
            model_save_path = os.path.join(output_dir, 'final_saved_model')
            tf.keras.models.save_model(agent.q_network, model_save_path)
            print(f"Final model saved to {model_save_path}")
            
            with open(os.path.join(output_dir, 'final_results.txt'), 'w') as f:
                f.write(f"Mode: {args.mode}\n")
                f.write(f"Final Mean Reward: {final_mean_reward:.2f}\n")
                f.write(f"Final Std Reward: {final_std_reward:.2f}\n")
        else:
            print("Training did not complete. No final results available.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()