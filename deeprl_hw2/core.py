"""Core classes."""

from collections import deque
import random
import numpy as np
from PIL import Image

class Sample:
    """Represents a reinforcement learning sample (state, action, reward, next_state, done)."""
    
    def __init__(self, state, action, reward, next_state, is_terminal):
        """Initializes the sample with state, action, reward, next_state, and terminal flag."""
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.is_terminal = is_terminal

class Preprocessor:
    """Preprocessor base class for DQN."""
    
    def process_state_for_network(self, state):
        """Preprocess state for the network (e.g., resize, normalize)."""
        return self._preprocess_frame(state) / 255.0  

    def process_state_for_memory(self, state):
        """Preprocess state for the memory (e.g., resize, uint8 conversion)."""
        return self._preprocess_frame(state).astype(np.uint8)

    def _preprocess_frame(self, frame, new_size=(84, 84)):
        """Resize the frame to 84x84 and convert to grayscale."""
        frame = np.mean(frame, axis=2).astype(np.uint8)  
        frame = Image.fromarray(frame)
        frame = frame.resize(new_size)
        return np.array(frame)

    def process_batch(self, samples):
        """Process a batch of samples."""
        return [self.process_state_for_network(sample) for sample in samples]

    def process_reward(self, reward):
        """Clip the reward between -1 and 1."""
        return np.clip(reward, -1, 1)

    def reset(self):
        """Reset internal states (if any)."""
        pass


class ReplayMemory:
    def __init__(self, max_size, frame_height, frame_width, history_length=4):
        self.max_size = max_size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.history_length = history_length
        self.frames = np.zeros((max_size, frame_height, frame_width), dtype=np.uint8)
        self.actions = np.zeros(max_size, dtype=np.int8)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=bool)       
        self.current = 0
        self.count = 0

    def append(self, frame, action, reward, done):
        self.frames[self.current] = frame
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.dones[self.current] = done
        
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.max_size

    def sample(self, batch_size):
        max_index = min(self.count, self.max_size)
        indices = np.random.randint(self.history_length, max_index, size=batch_size)
        
        states = np.array([self._get_state(index - 1) for index in indices], dtype=np.float32) / 255.0
        next_states = np.array([self._get_state(index) for index in indices], dtype=np.float32) / 255.0
        
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        dones = self.dones[indices]
        
        return states, actions, rewards, next_states, dones

    def _get_state(self, index):
        if index < self.history_length - 1:
            frames = [self.frames[0]] * (self.history_length - 1 - index)
            frames.extend([self.frames[i % self.count] for i in range(index + 1)])
        else:
            frames = [self.frames[i % self.count] for i in range(index - self.history_length + 1, index + 1)]
        return np.stack(frames, axis=-1)

    def __len__(self):
        return self.count


