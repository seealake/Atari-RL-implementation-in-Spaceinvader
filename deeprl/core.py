"""Core classes."""

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
        
        # Need at least history_length + 1 frames to sample (for state and next_state)
        if max_index <= self.history_length:
            raise ValueError(f"Not enough samples in memory. Have {max_index}, need at least {self.history_length + 1}")
        
        # Ensure we don't sample from indices that would wrap around incorrectly
        # or sample terminal states as the "current" state
        valid_indices = []
        max_attempts = batch_size * 10  # Prevent infinite loop
        attempts = 0
        
        while len(valid_indices) < batch_size and attempts < max_attempts:
            # Sample index for the 'next_state' frame (the frame after taking action)
            idx = np.random.randint(self.history_length, max_index)
            attempts += 1
            
            # Skip if any frame in the state history crosses an episode boundary
            # For the state (ending at idx-1), we need to check if any of the frames
            # from (idx - history_length) to (idx - 2) are terminal states
            # (because if frame at idx-2 is terminal, then frame at idx-1 starts a new episode)
            valid = True
            for i in range(2, self.history_length + 1):
                check_idx = (idx - i) % self.max_size
                if self.dones[check_idx]:
                    valid = False
                    break
            
            # Also avoid sampling the current write position if buffer is full
            if self.count >= self.max_size:
                if abs(idx - self.current) < self.history_length + 1:
                    valid = False
            
            if valid:
                valid_indices.append(idx)
        
        # If we couldn't find enough valid samples, fall back to random sampling
        if len(valid_indices) < batch_size:
            remaining = batch_size - len(valid_indices)
            fallback_indices = np.random.randint(self.history_length, max_index, size=remaining)
            valid_indices.extend(fallback_indices.tolist())
        
        indices = np.array(valid_indices)
        
        # Memory layout: at index i, we store:
        # - frames[i]: the frame observed after taking action
        # - actions[i]: the action taken that led to frames[i]
        # - rewards[i]: the reward received for taking that action
        # - dones[i]: whether frames[i] is a terminal state
        #
        # For a transition (s, a, r, s'):
        # Memory layout at index i stores the frame AFTER taking action[i] and receiving reward[i]
        # So for a valid transition at index i:
        # - state s = stack of frames ending at index (i-1), i.e., frames before action was taken
        # - action a = actions[i-1], the action taken from state s
        # - reward r = rewards[i-1], the reward received for taking that action
        # - next_state s' = stack of frames ending at index i, i.e., the resulting state
        # - done = dones[i-1], whether the episode ended after taking the action
        #
        # Note: We sample 'index' which represents the next_state frame index
        
        states = np.array([self._get_state(index - 1) for index in indices], dtype=np.float32) / 255.0
        next_states = np.array([self._get_state(index) for index in indices], dtype=np.float32) / 255.0
        
        # Use (index - 1) to get the action, reward, and done corresponding to the transition
        prev_indices = (indices - 1) % self.max_size
        actions = self.actions[prev_indices]
        rewards = self.rewards[prev_indices]
        dones = self.dones[prev_indices].astype(np.float32)
        
        return states, actions, rewards, next_states, dones

    def _get_state(self, index):
        """Get a state consisting of history_length consecutive frames ending at index."""
        if index < self.history_length - 1:
            # Not enough frames, pad with zeros or first frame
            frames = [self.frames[0]] * (self.history_length - 1 - index)
            frames.extend([self.frames[i % self.max_size] for i in range(index + 1)])
        else:
            frames = [self.frames[(index - self.history_length + 1 + i) % self.max_size] for i in range(self.history_length)]
        return np.stack(frames, axis=-1)

    def __len__(self):
        return self.count


