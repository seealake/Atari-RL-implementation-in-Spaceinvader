import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, Lambda
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import gc
import os
import logging
import pickle
from tqdm import tqdm
class DQNAgent:
    def __init__(self, model, input_shape, num_actions, preprocessor, memory, policy,
                 gamma=0.99, target_update_freq=1000, num_burn_in=50000, 
                 train_freq=4, batch_size=32, double_q=False, dueling=False, tau=0.001, 
                 checkpoint_dir='checkpoints'):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
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
        file_handler = logging.FileHandler('dqn_training.log')
        file_handler.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)

    @tf.function
    def update_policy(self):
        if len(self.memory) < self.batch_size:
            return 

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)

        rewards = np.clip(rewards, -1, 1) 

        q_values_next = self.q_network(next_states)
        next_actions = tf.argmax(q_values_next, axis=1)
        target_q_values_next = self.target_network(next_states)
        target_values = rewards + self.gamma * tf.reduce_sum(tf.one_hot(next_actions, self.num_actions) * target_q_values_next, axis=1) * (1 - dones)

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
        state = env.reset()
        state = self.preprocessor.process_state_for_network(state)

        evaluation_results = []

        for t in range(start_step, num_iterations):
            if t % 100 == 0: 
                self.logger.info(f"Step {t}: Training in progress")
        
            if t % 1000 == 0:
                current_epsilon = self.policy.epsilon_policy.epsilon
                self.logger.info(f"Step {t}: Current epsilon: {current_epsilon:.4f}")
        
            action = self.select_action(np.expand_dims(state, axis=0), is_training=True)
            next_state, reward, done, _ = env.step(action)
            next_state = self.preprocessor.process_state_for_network(next_state)
        
            self.memory.append(next_state[:,:,0], action, reward, done)
            if t >= self.num_burn_in and t % self.train_freq == 0:
                loss = self.update_policy()
                self.record_loss(loss)
        
            if t % checkpoint_freq == 0:
                self.save_checkpoint(t, evaluation_results)
        
            state = next_state
            if done:
                state = env.reset()
                state = self.preprocessor.process_state_for_network(state)
                self.logger.info(f"Episode ended at step {t}")
        
            if t >= self.num_burn_in and t % self.target_update_freq == 0:
                self.soft_update_target_network()
                self.logger.info(f"Target network updated at step {t}")
        
            if t % 10000 == 0:  
                gc.collect()
                self.logger.info(f"Step {t}: Garbage collection performed")
                self.save_model(f'checkpoint_model_{t}')
                self.logger.info(f"Model checkpoint saved at step {t}")       
                mean_reward, std_reward = self.evaluate(env, num_episodes=10)
                evaluation_results.append((t, mean_reward, std_reward))
                self.logger.info(f"Step {t}: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
                avg_loss = np.mean(self.losses[-10000:]) 
                self.logger.info(f"Step {t}: Average loss: {avg_loss:.4f}")
            self.logger.debug(f"Step {t}: Action {action}, Reward {reward}, Done {done}")
        
            if t > start_step and t % restart_freq == 0:
                self.save_checkpoint(t, evaluation_results)
                self.logger.info(f"Training paused at step {t} for potential restart")
                return evaluation_results, None  
        final_mean_reward, final_std_reward = self.evaluate(env, num_episodes=100)
        self.plot_smoothed_losses()
        self.logger.info(f"Final evaluation - Mean reward: {final_mean_reward:.2f} +/- {final_std_reward:.2f}")
        self.save_model(f'final_model_{num_iterations}')
        self.logger.info(f"Final model saved after {num_iterations} iterations")
        return evaluation_results, (final_mean_reward, final_std_reward)


    def select_action(self, state, is_training=True):
        q_values = self.q_network.predict(state)
        if is_training:
            return self.policy.select_action(q_values)
        else:
            return np.argmax(q_values) 

    def evaluate(self, env, num_episodes, max_episode_length=None):
        episode_rewards = []
        for i in range(num_episodes):
            state = env.reset()
            state = self.preprocessor.process_state_for_network(state)
            done = False
            step = 0
            episode_reward = 0

            while not done and (max_episode_length is None or step < max_episode_length):
                action = self.select_action(np.array([state]), is_training=False)
                next_state, reward, done, _ = env.step(action)
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
            'evaluation_results': evaluation_results
        }
        with open(os.path.join(self.checkpoint_dir, f'extra_data_{step}.pkl'), 'wb') as f:
            pickle.dump(extra_data, f)
    
        self.logger.info(f"Checkpoint saved at step {step}")

    def load_checkpoint(self, checkpoint_path, step=None):
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
        
            step = checkpoint_data.get('step', 0)
            self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        else:
            pass 
        return step

    def get_latest_checkpoint(self, checkpoint_dir):
        latest_tf_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    
        if latest_tf_checkpoint:
            step = int(latest_tf_checkpoint.split('_')[-1])
            extra_data_file = os.path.join(checkpoint_dir, f'extra_data_{step}.pkl')
            if os.path.exists(extra_data_file):
                return latest_tf_checkpoint, step
        return None, None
    def record_loss(self, loss):
        self.losses.append(loss)

    def plot_smoothed_losses(self, window_size=1000):
        smoothed_losses = np.convolve(self.losses, np.ones(window_size)/window_size, mode='valid')
        plt.figure(figsize=(10, 5))
        plt.plot(smoothed_losses)
        plt.title('Training Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.savefig('Training_loss.png')
        plt.close()



