import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import gc
import os
import logging
import pickle
class DQNAgent:
    def __init__(self, model, input_shape, num_actions, preprocessor, memory, policy,
                 gamma=0.99, target_update_freq=1000, num_burn_in=50000, 
                 train_freq=4, batch_size=32, double_q=False, dueling=False, tau=0.001, 
                 checkpoint_dir='checkpoints'):
        # Configure logging - only set basicConfig once
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Avoid adding duplicate file handlers by checking existing handlers
        has_file_handler = any(isinstance(h, logging.FileHandler) for h in self.logger.handlers)
        if not has_file_handler:
            file_handler = logging.FileHandler('dqn_training.log')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)
        
        self.losses = []
        self.input_shape = input_shape
        self.num_actions = num_actions
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
        self.optimizer = Adam(learning_rate=3e-4)
        self.loss_func = tf.keras.losses.Huber()
        self.q_network = model
        self.target_network = tf.keras.models.clone_model(model)
        self.target_network.set_weights(self.q_network.get_weights())
        self.q_network.compile(optimizer=self.optimizer, loss=self.loss_func)
        
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        
        self.logger.info("DQNAgent initialized")

    def update_policy(self):
        # Need enough samples for both batch_size and memory sampling requirements
        # Memory.sample requires at least history_length + 2 samples
        min_samples_needed = max(self.batch_size, self.memory.history_length + 2)
        if len(self.memory) < min_samples_needed:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)  # Convert to int32 for one_hot
        rewards = tf.convert_to_tensor(np.clip(rewards, -1, 1), dtype=tf.float32)
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
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)  
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

    def fit(self, env, num_iterations, start_step=0, max_episode_length=None, checkpoint_dir='checkpoints', checkpoint_freq=50000, restart_freq=500000):
        # Handle both old and new Gym API
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state, _ = reset_result  # New Gym API (v0.26+)
        else:
            state = reset_result  # Old Gym API
        self.preprocessor.reset()
        state = self.preprocessor.process_state_for_network(state)

        evaluation_results = []
        
        # Synchronize policy step counter with training start_step
        # This ensures epsilon decay is consistent when resuming training
        if hasattr(self.policy, 'step'):
            self.policy.step = start_step
        elif hasattr(self.policy, 'current_step'):
            self.policy.current_step = start_step

        for t in range(start_step, num_iterations):
            if t % 100 == 0: 
                self.logger.info(f"Step {t}: Training in progress")
        
            if t % 1000 == 0:
                current_epsilon = self.policy.epsilon_policy.epsilon
                self.logger.info(f"Step {t}: Current epsilon: {current_epsilon:.4f}")
        
            # Store the current state's last frame before taking action
            frame_to_store = (state[:,:,-1] * 255).astype(np.uint8)
            
            action = self.select_action(np.expand_dims(state, axis=0), is_training=True)
            # Handle both old and new Gym API for step()
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, _ = step_result  # New Gym API
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_result  # Old Gym API
            next_state = self.preprocessor.process_state_for_network(next_state)
        
            # Store transition: frame from current state, action taken, reward received, done flag
            # The memory stores: frame[i] is the last frame of state from which action[i] was taken
            # reward[i] and done[i] are the results of taking action[i]
            self.memory.append(frame_to_store, action, reward, done)
            if t >= self.num_burn_in and t % self.train_freq == 0:
                loss = self.update_policy()
                if loss is not None:
                    self.record_loss(float(loss.numpy()) if hasattr(loss, 'numpy') else float(loss))
        
            if t % checkpoint_freq == 0 and t > 0:
                self.save_checkpoint(t, evaluation_results)
        
            state = next_state
            if done:
                # Handle both old and new Gym API
                reset_result = env.reset()
                if isinstance(reset_result, tuple):
                    state, _ = reset_result
                else:
                    state = reset_result
                self.preprocessor.reset()
                state = self.preprocessor.process_state_for_network(state)
                self.logger.info(f"Episode ended at step {t}")
        
            if t >= self.num_burn_in and t % self.target_update_freq == 0:
                self.soft_update_target_network()
                self.logger.info(f"Target network updated at step {t}")
        
            if t % 10000 == 0 and t > 0:  
                gc.collect()
                self.logger.info(f"Step {t}: Garbage collection performed")
                self.save_model(f'checkpoint_model_{t}')
                self.logger.info(f"Model checkpoint saved at step {t}")
                
                # Save preprocessor state before evaluation
                saved_preprocessor_state = None
                if hasattr(self.preprocessor, 'save_state'):
                    saved_preprocessor_state = self.preprocessor.save_state()
                
                mean_reward, std_reward = self.evaluate(env, num_episodes=10)
                
                # Restore preprocessor state after evaluation
                if saved_preprocessor_state is not None and hasattr(self.preprocessor, 'restore_state'):
                    self.preprocessor.restore_state(saved_preprocessor_state)
                
                # After evaluation, the environment is in an unknown state.
                # We need to reset and rebuild the state to continue training.
                # Note: This means we lose the current episode progress, but ensures consistency.
                reset_result = env.reset()
                if isinstance(reset_result, tuple):
                    state, _ = reset_result
                else:
                    state = reset_result
                self.preprocessor.reset()
                state = self.preprocessor.process_state_for_network(state)
                
                evaluation_results.append((t, mean_reward, std_reward))
                self.logger.info(f"Step {t}: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
                if len(self.losses) > 0:
                    avg_loss = np.mean(self.losses[-10000:]) 
                    self.logger.info(f"Step {t}: Average loss: {avg_loss:.4f}")
                else:
                    self.logger.info(f"Step {t}: No losses recorded yet")
            self.logger.debug(f"Step {t}: Action {action}, Reward {reward}, Done {done}")
        
            if t > start_step and t % restart_freq == 0:
                self.save_checkpoint(t, evaluation_results)
                self.logger.info(f"Training paused at step {t} for potential restart")
                return evaluation_results, None, t  
        final_mean_reward, final_std_reward = self.evaluate(env, num_episodes=100)
        self.plot_smoothed_losses()
        self.logger.info(f"Final evaluation - Mean reward: {final_mean_reward:.2f} +/- {final_std_reward:.2f}")
        self.save_model(f'final_model_{num_iterations}')
        self.logger.info(f"Final model saved after {num_iterations} iterations")
        # Always return 3 values for consistent interface: (evaluation_results, final_result, stop_step)
        return evaluation_results, (final_mean_reward, final_std_reward), num_iterations


    def select_action(self, state, is_training=True):
        q_values = self.q_network.predict(state, verbose=0)
        if is_training:
            action = self.policy.select_action(q_values)
            # Ensure action is a Python int, not numpy int (for gym compatibility)
            return int(action)
        else:
            # Flatten q_values if it's 2D (batch_size=1, num_actions)
            if q_values.ndim > 1:
                q_values = q_values.flatten()
            return int(np.argmax(q_values))

    def evaluate(self, env, num_episodes, max_episode_length=None):
        episode_rewards = []
        for i in range(num_episodes):
            # Handle both old and new Gym API
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                state, _ = reset_result  # New Gym API (v0.26+)
            else:
                state = reset_result  # Old Gym API
            self.preprocessor.reset()
            state = self.preprocessor.process_state_for_network(state)
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
                next_state = self.preprocessor.process_state_for_network(next_state)
                episode_reward += reward
                state = next_state
                step += 1

            episode_rewards.append(episode_reward)
            print(f"Episode {i+1}/{num_episodes} - Reward: {episode_reward}")

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        print(f"Evaluation complete. Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        return mean_reward, std_reward

    def save_model(self, filepath):
        tf.saved_model.save(self.q_network, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        self.q_network = tf.saved_model.load(filepath)
        print(f"Model loaded from {filepath}")

    def save_checkpoint(self, step, evaluation_results):
        checkpoint = tf.train.Checkpoint(model=self.q_network, optimizer=self.optimizer)
        checkpoint.save(file_prefix=os.path.join(self.checkpoint_dir, f"checkpoint_{step}"))
        extra_data = {
            'step': step,
            'memory': self.memory,
            'policy_state': self.policy.get_config() if hasattr(self.policy, 'get_config') else None,
            'evaluation_results': evaluation_results,
            'losses': self.losses  # Save losses for plotting after resuming
        }
        with open(os.path.join(self.checkpoint_dir, f'extra_data_{step}.pkl'), 'wb') as f:
            pickle.dump(extra_data, f)
    
        self.logger.info(f"Checkpoint saved at step {step}")

    def load_checkpoint(self, checkpoint_path, step=None):
        loaded_step = 0  # Default value to avoid UnboundLocalError
        
        if checkpoint_path.endswith('.pkl'):
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
        
            print(f"Loaded checkpoint data keys: {checkpoint_data.keys()}")

            if 'q_network_weights' in checkpoint_data:
                self.q_network.set_weights(checkpoint_data['q_network_weights'])
                if 'target_network_weights' in checkpoint_data:
                    self.target_network.set_weights(checkpoint_data['target_network_weights'])
                else:
                    self.target_network.set_weights(self.q_network.get_weights())
            elif 'model' in checkpoint_data:
                self.q_network.set_weights(checkpoint_data['model'])
                self.target_network.set_weights(checkpoint_data['model'])
            else:
                print("Warning: No model weights found in checkpoint. Using random initialization.")

            try:
                if 'optimizer_weights' in checkpoint_data:
                    self.optimizer.set_weights(checkpoint_data['optimizer_weights'])
            except ValueError:
                print("Warning: Failed to load optimizer weights. Reinitializing optimizer.")
                self.optimizer = Adam(learning_rate=self.optimizer.learning_rate)
        
            if 'memory' in checkpoint_data:
                self.memory = checkpoint_data['memory']
            if 'policy_state' in checkpoint_data and hasattr(self.policy, 'set_config'):
                self.policy.set_config(checkpoint_data['policy_state'])
            if 'losses' in checkpoint_data:
                self.losses = checkpoint_data['losses']
        
            loaded_step = checkpoint_data.get('step', 0)
            self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        else:
            # Handle TensorFlow checkpoint directory
            checkpoint = tf.train.Checkpoint(model=self.q_network, optimizer=self.optimizer)
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
            if latest_checkpoint:
                checkpoint.restore(latest_checkpoint)
                # Try to extract step from checkpoint filename
                try:
                    loaded_step = int(latest_checkpoint.split('_')[-1].split('-')[0])
                except (ValueError, IndexError):
                    loaded_step = step if step is not None else 0
                self.target_network.set_weights(self.q_network.get_weights())
                self.logger.info(f"TensorFlow checkpoint loaded from {latest_checkpoint}")
                
                # Try to load extra data if available
                extra_data_file = os.path.join(checkpoint_path, f'extra_data_{loaded_step}.pkl')
                if os.path.exists(extra_data_file):
                    with open(extra_data_file, 'rb') as f:
                        extra_data = pickle.load(f)
                    if 'memory' in extra_data:
                        self.memory = extra_data['memory']
                    if 'policy_state' in extra_data and hasattr(self.policy, 'set_config'):
                        self.policy.set_config(extra_data['policy_state'])
                    if 'losses' in extra_data:
                        self.losses = extra_data['losses']
            else:
                print(f"Warning: No checkpoint found at {checkpoint_path}")
                loaded_step = step if step is not None else 0
                
        return loaded_step

    def get_latest_checkpoint(self, checkpoint_dir):
        latest_tf_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    
        if latest_tf_checkpoint:
            # Handle checkpoint filename format like 'checkpoint_100000-1'
            try:
                step = int(latest_tf_checkpoint.split('_')[-1].split('-')[0])
            except (ValueError, IndexError):
                self.logger.warning(f"Could not parse step from checkpoint: {latest_tf_checkpoint}")
                return None, None
            extra_data_file = os.path.join(checkpoint_dir, f'extra_data_{step}.pkl')
            if os.path.exists(extra_data_file):
                return latest_tf_checkpoint, step
        return None, None
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
        plt.figure(figsize=(10, 5))
        plt.plot(smoothed_losses)
        plt.title('Training Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.savefig('Training_loss.png')
        plt.close()



