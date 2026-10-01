import numpy as np
from matplotlib.figure import Figure
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import gc
import os
import logging
import pickle
import copy
import gzip
from pathlib import Path
import random
import re

class DQNAgent:
    def __init__(self, model, input_shape, num_actions, preprocessor, memory, policy,
                 gamma=0.99, target_update_freq=1000, num_burn_in=50000, 
                 train_freq=4, batch_size=32, double_q=False, dueling=False, tau=0.001,
                 checkpoint_dir='checkpoints', output_dir='.', learning_rate=3e-4,
                 keep_checkpoints=2, reward_clip=1.0, gradient_clip=5.0):
        # Configure logging - only set basicConfig once
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f'{__name__}.{id(self)}')
        self.logger.setLevel(logging.INFO)
        
        # Avoid adding duplicate file handlers by checking existing handlers
        has_file_handler = any(isinstance(h, logging.FileHandler) for h in self.logger.handlers)
        if not has_file_handler:
            file_handler = logging.FileHandler(self.output_dir / 'dqn_training.log', encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)
        
        self.losses = []
        self.evaluation_results = []
        self.input_shape = input_shape
        self.num_actions = int(num_actions)
        self.memory = memory
        self.policy = policy
        self.preprocessor = preprocessor
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.double_q = double_q
        self.dueling = dueling
        self.tau = tau
        self.steps = 0
        self.reward_clip, self.gradient_clip = reward_clip, gradient_clip
        self.optimizer = Adam(learning_rate=learning_rate)
        self.loss_func = tf.keras.losses.Huber()
        self.q_network = model
        self.target_network = tf.keras.models.clone_model(model)
        self.target_network.set_weights(self.q_network.get_weights())
        self.q_network.compile(optimizer=self.optimizer, loss=self.loss_func)
        self.optimizer.build(self.q_network.trainable_variables)
        
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.checkpoint = tf.train.Checkpoint(
            model=self.q_network, target_network=self.target_network, optimizer=self.optimizer
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, self.checkpoint_dir, max_to_keep=keep_checkpoints
        )
        
        self.logger.info("DQNAgent initialized")

    def update_policy(self):
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)  # Convert to int32 for one_hot
        rewards = tf.convert_to_tensor(np.clip(rewards, -self.reward_clip, self.reward_clip), dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        target_q_values_next = self.target_network(next_states)
        
        if self.double_q:
            # Double DQN: use online network to select actions, target network to evaluate
            q_values_next = self.q_network(next_states)
            next_actions = tf.argmax(q_values_next, axis=1)
        else:
            # Standard DQN: use target network to select actions
            next_actions = tf.argmax(target_q_values_next, axis=1)
        
        # Cast next_actions to int32 for one_hot compatibility
        next_actions = tf.cast(next_actions, tf.int32)
        
        # Compute target values and stop gradient to prevent backpropagation
        target_values = rewards + self.gamma * tf.reduce_sum(
            tf.one_hot(next_actions, self.num_actions) * target_q_values_next, axis=1
        ) * (1 - dones)
        target_values = tf.stop_gradient(target_values)

        with tf.GradientTape() as tape:
            q_values = self.q_network(states)
            q_values_for_actions = tf.reduce_sum(q_values * tf.one_hot(actions, self.num_actions), axis=1)
            loss = self.loss_func(target_values, q_values_for_actions)

        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        tf.debugging.assert_all_finite(loss, 'Non-finite DQN loss')
        gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clip)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        assert states.shape[0] == self.batch_size
        assert actions.shape[0] == self.batch_size
        assert rewards.shape[0] == self.batch_size
        assert next_states.shape[0] == self.batch_size
        assert dones.shape[0] == self.batch_size
        return loss

    def soft_update_target_network(self):
        """Soft update of target network."""
        for target_param, local_param in zip(self.target_network.trainable_variables, self.q_network.trainable_variables):
            target_param.assign(self.tau * local_param + (1.0 - self.tau) * target_param)

    def fit(self, env, num_iterations, start_step=None, max_episode_length=None,
            checkpoint_dir=None, checkpoint_freq=50000, restart_freq=500000,
            eval_env=None, eval_freq=10000, eval_episodes=10, final_eval_episodes=100,
            seed=0, log_freq=1000):
        if start_step is not None and start_step != self.steps:
            raise ValueError("start_step must match the loaded checkpoint")
        if num_iterations <= self.steps:
            raise ValueError("The end step must exceed the completed step count")
        if ((eval_freq and eval_episodes) or final_eval_episodes) and (
                eval_env is None or eval_env is env):
            raise ValueError("Pass a separate environment for evaluation")
        if checkpoint_dir is not None and Path(checkpoint_dir) != Path(self.checkpoint_dir):
            raise ValueError("Set checkpoint_dir when constructing DQNAgent")
        if hasattr(self.policy, 'step'):
            self.policy.step = self.steps
        elif hasattr(self.policy, 'current_step'):
            self.policy.current_step = self.steps

        def reset(seed=None):
            observation, _ = env.reset(seed=seed)
            self.preprocessor.reset()
            self.memory.start_new_episode()
            return self.preprocessor.process_state_for_network(observation)

        state = reset(seed=seed + self.steps)
        env.action_space.seed(seed + self.steps)
        episode_steps, episode_reward = 0, 0.0
        self.logger.info("Training from step %d to %d", self.steps, num_iterations)
        for t in range(self.steps, num_iterations):
            action = self.select_action(state[None, ...])
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_steps += 1
            episode_reward += float(reward)
            truncated = truncated or (
                max_episode_length is not None and episode_steps >= max_episode_length
            )
            done = bool(terminated or truncated)
            next_state = self.preprocessor.process_state_for_network(observation)
            self.memory.append(
                np.rint(state[..., -1] * 255).astype(np.uint8), action, reward, done,
                next_frame=np.rint(next_state[..., -1] * 255).astype(np.uint8),
                terminal=terminated,
            )
            self.steps = t + 1
            if self.steps >= self.num_burn_in and self.steps % self.train_freq == 0:
                loss = self.update_policy()
                if loss is not None:
                    self.record_loss(float(loss.numpy()))
            if self.steps >= self.num_burn_in and self.steps % self.target_update_freq == 0:
                self.soft_update_target_network()

            state = next_state
            if done:
                self.logger.info("Step %d: episode reward %.1f", self.steps, episode_reward)
                state = reset()
                episode_steps, episode_reward = 0, 0.0
            if eval_freq and eval_episodes and self.steps % eval_freq == 0:
                mean, std = self.evaluate(
                    eval_env, eval_episodes, max_episode_length, seed=seed
                )
                self.evaluation_results.append((self.steps, mean, std))
            restart = bool(restart_freq and self.steps % restart_freq == 0)
            if restart and self.steps < num_iterations:
                # Reset the environment only, never discard learned weights or Adam state.
                state = reset()
                episode_steps, episode_reward = 0, 0.0
                gc.collect()
            periodic_save = bool(checkpoint_freq and self.steps % checkpoint_freq == 0)
            if (periodic_save or restart) and self.steps < num_iterations:
                self.save_checkpoint(self.steps, self.evaluation_results)
            if self.steps % log_freq == 0 or self.steps == num_iterations:
                epsilon = getattr(getattr(self.policy, 'epsilon_policy', self.policy), 'epsilon', 0)
                self.logger.info(
                    "Step %d/%d: epsilon %.4f, gradient updates %d",
                    self.steps, num_iterations, epsilon, int(self.optimizer.iterations.numpy()),
                )

        final_result = None
        if final_eval_episodes:
            final_result = self.evaluate(
                eval_env, final_eval_episodes, max_episode_length, seed=seed
            )
        self.save_checkpoint(self.steps, self.evaluation_results)
        self.plot_smoothed_losses()
        self.save_model(self.output_dir / 'final_model.keras')
        return list(self.evaluation_results), final_result, self.steps


    def select_action(self, state, is_training=True):
        q_values = self.q_network(tf.convert_to_tensor(state, dtype=tf.float32), training=False).numpy()
        if is_training:
            action = self.policy.select_action(q_values)
            # Ensure action is a Python int, not numpy int (for gym compatibility)
            return int(action)
        else:
            # Flatten q_values if it's 2D (batch_size=1, num_actions)
            if q_values.ndim > 1:
                q_values = q_values.flatten()
            return int(np.argmax(q_values))

    def evaluate(self, env, num_episodes, max_episode_length=None, seed=0):
        preprocessor = copy.deepcopy(self.preprocessor)
        episode_rewards = []
        for i in range(num_episodes):
            # Handle both old and new Gym API
            reset_result = env.reset(seed=seed + i)
            if isinstance(reset_result, tuple):
                state, _ = reset_result  # New Gym API (v0.26+)
            else:
                state = reset_result  # Old Gym API
            preprocessor.reset()
            state = preprocessor.process_state_for_network(state)
            done = False
            step = 0
            episode_reward = 0

            while not done and (max_episode_length is None or step < max_episode_length):
                action = self.select_action(np.array([state]), is_training=False)
                # Handle both old and new Gym API for step()
                step_result = env.step(action)
                if len(step_result) == 5:
                    next_state, reward, terminated, truncated, _ = step_result  # New Gym API
                    done = terminated or truncated
                else:
                    next_state, reward, done, _ = step_result  # Old Gym API
                next_state = preprocessor.process_state_for_network(next_state)
                episode_reward += reward
                state = next_state
                step += 1

            episode_rewards.append(episode_reward)
            self.logger.debug("Evaluation episode %d: reward %s", i + 1, episode_reward)

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        print(f"Evaluation complete. Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        return float(mean_reward), float(std_reward)

    def save_model(self, filepath):
        filepath = Path(filepath)
        if filepath.suffix != '.keras':
            filepath = Path(str(filepath) + '.keras')
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.q_network.save(filepath)
        print(f"Model saved to {filepath}")
        return filepath

    def load_model(self, filepath):
        """Load network weights; use load_checkpoint to resume optimizer and replay."""
        model = tf.keras.models.load_model(filepath, compile=False)
        self.q_network.set_weights(model.get_weights())
        self.target_network.set_weights(model.get_weights())
        print(f"Model loaded from {filepath}")

    @staticmethod
    def _checkpoint_step(path):
        match = re.search(r"(?:ckpt-|checkpoint[-_])(\d+)", Path(path).name)
        if not match:
            raise ValueError(f"Unrecognized checkpoint name: {path}")
        return int(match.group(1))

    def save_checkpoint(self, step, evaluation_results):
        if step != self.steps:
            raise ValueError("Checkpoint step must equal completed training steps")
        self.evaluation_results = list(evaluation_results)
        extra_data = {
            'step': step, 'memory': self.memory,
            'policy_state': self.policy.get_config(),
            'evaluation_results': evaluation_results, 'losses': self.losses,
            'numpy_random_state': np.random.get_state(),
            'python_random_state': random.getstate(),
            'agent_config': {name: getattr(self, name) for name in (
                'gamma', 'tau', 'target_update_freq', 'num_burn_in', 'train_freq',
                'batch_size', 'double_q', 'dueling', 'reward_clip', 'gradient_clip',
            )},
        }
        extra_path = Path(self.checkpoint_dir) / f'extra_data_{step}.pkl.gz'
        temporary = extra_path.with_suffix('.gz.tmp')
        with gzip.open(temporary, 'wb', compresslevel=1) as handle:
            pickle.dump(extra_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(extra_path)
        path = self.checkpoint_manager.save(checkpoint_number=step)
        retained = {self._checkpoint_step(p) for p in self.checkpoint_manager.checkpoints}
        for old in Path(self.checkpoint_dir).glob('extra_data_*.pkl.gz'):
            saved_step = int(old.name.removeprefix('extra_data_').removesuffix('.pkl.gz'))
            if saved_step not in retained:
                old.unlink()
        self.logger.info("Checkpoint saved at step %d: %s", step, path)
        return path

    def load_checkpoint(self, checkpoint_path, step=None):
        path = Path(checkpoint_path).expanduser()
        data = None
        if path.name.endswith(('.pkl', '.pkl.gz')):
            opener = gzip.open if path.suffix == '.gz' else open
            with opener(path, 'rb') as handle:
                data = pickle.load(handle)
            weights = data.get('q_network_weights', data.get('model'))
            if weights is not None:
                if step is not None and data.get('step', 0) != step:
                    raise ValueError('Requested step does not match the legacy checkpoint')
                self.q_network.set_weights(weights)
                self.target_network.set_weights(data.get('target_network_weights', weights))
                if 'optimizer_weights' in data:
                    self.optimizer.set_weights(data['optimizer_weights'])
                else:
                    self.logger.warning('Legacy weights restored without optimizer state')
                if 'target_network_weights' not in data:
                    self.logger.warning('Legacy target weights synchronized from the online model')
                self._restore_training_state(data)
                return self.steps
            candidates = [
                candidate.with_suffix('') for candidate in path.parent.glob('*.index')
                if self._checkpoint_step(candidate) == data['step']
            ]
            if not candidates:
                raise FileNotFoundError("The sidecar has no matching TensorFlow checkpoint")
            path = candidates[-1]
        elif path.is_dir():
            latest = tf.train.latest_checkpoint(str(path))
            if latest is None:
                raise FileNotFoundError(f"No checkpoint in {path}")
            path = Path(latest)
        elif path.suffix == '.index':
            path = path.with_suffix('')
        if not Path(str(path) + '.index').is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        loaded_step = self._checkpoint_step(path)
        if step is not None and step != loaded_step:
            raise ValueError(f"Checkpoint is at step {loaded_step}, not {step}")
        if data is None:
            sidecar = path.parent / f'extra_data_{loaded_step}.pkl.gz'
            if not sidecar.exists():
                sidecar = path.parent / f'extra_data_{loaded_step}.pkl'
            opener = gzip.open if sidecar.suffix == '.gz' else open
            with opener(sidecar, 'rb') as handle:
                data = pickle.load(handle)
        if data['step'] != loaded_step:
            raise ValueError("Checkpoint and replay sidecar steps do not match")

        has_target = any(
            name.startswith('target_network/') for name, _ in tf.train.list_variables(str(path))
        )
        checkpoint = self.checkpoint if has_target else tf.train.Checkpoint(
            model=self.q_network, optimizer=self.optimizer
        )
        status = checkpoint.restore(str(path))
        status.assert_existing_objects_matched()
        if not has_target:
            status.expect_partial()
            self.target_network.set_weights(self.q_network.get_weights())
            self.logger.warning("Legacy checkpoint has no target weights; synchronized from online model")
        if 'memory' not in data:
            raise ValueError('Checkpoint sidecar is missing replay memory')
        self._restore_training_state(data)
        self.logger.info("Restored training from step %d: %s", self.steps, path)
        return self.steps

    def _restore_training_state(self, data):
        if 'memory' in data:
            memory = data['memory']
            if (memory.frame_height, memory.frame_width, memory.history_length) != tuple(self.input_shape):
                raise ValueError('Checkpoint replay shape does not match the network input')
            self.memory = memory
        else:
            self.logger.warning('Legacy weights have no replay memory; using a fresh buffer')
        if data.get('policy_state') is not None:
            self.policy.set_config(data['policy_state'])
        self.losses = data.get('losses', [])
        self.evaluation_results = data.get('evaluation_results', [])
        for name, value in data.get('agent_config', {}).items():
            setattr(self, name, value)
        if 'numpy_random_state' in data:
            np.random.set_state(data['numpy_random_state'])
        if 'python_random_state' in data:
            random.setstate(data['python_random_state'])
        self.steps = int(data.get('step', 0))

    def get_latest_checkpoint(self, checkpoint_dir):
        path = tf.train.latest_checkpoint(str(checkpoint_dir))
        return (path, self._checkpoint_step(path)) if path else (None, None)
    def record_loss(self, loss):
        self.losses.append(loss)

    def plot_smoothed_losses(self, window_size=1000):
        if len(self.losses) == 0:
            self.logger.warning("No losses to plot")
            return
        
        # Adjust window size if losses are fewer than window_size
        effective_window = min(window_size, len(self.losses))
        if effective_window < 1:
            effective_window = 1
            
        smoothed_losses = np.convolve(self.losses, np.ones(effective_window)/effective_window, mode='valid')
        figure = Figure(figsize=(10, 5))
        axes = figure.subplots()
        axes.plot(smoothed_losses)
        axes.set(title='Training Loss', xlabel='Gradient updates', ylabel='Loss')
        figure.savefig(self.output_dir / 'Training_loss.png')

    def close(self):
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)



