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
        self.actions = np.zeros(max_size, dtype=np.int32)  # Use int32 to support larger action spaces
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
        
        # Need at least history_length + 2 frames to sample:
        # - history_length frames for the current state
        # - 1 frame for the next state's last frame (index + 1)
        # - So we need indices from [history_length, max_index-1), which requires max_index > history_length + 1
        if max_index <= self.history_length + 1:
            raise ValueError(f"Not enough samples in memory. Have {max_index}, need at least {self.history_length + 2}")
        
        # Ensure we don't sample from indices that would wrap around incorrectly
        # or sample terminal states as the "current" state
        valid_indices = []
        max_attempts = batch_size * 10  # Prevent infinite loop
        attempts = 0
        
        while len(valid_indices) < batch_size and attempts < max_attempts:
            # Sample index for the state (the frame from which action was taken)
            # We need idx and idx+1 to be valid, so sample from [history_length, max_index-1)
            # idx must be >= history_length to ensure we have enough frames for the state stack
            idx = np.random.randint(self.history_length, max_index - 1)
            attempts += 1
            
            # Skip if state history crosses an episode boundary
            #
            # state s = frames[idx-history_length+1, ..., idx-1, idx]
            # next_state s' = frames[idx-history_length+2, ..., idx, idx+1]
            # 
            # We need to check that:
            # 1. dones[idx-history_length+1] to dones[idx-1] are all False 
            #    (so state stack doesn't cross episode boundaries)
            # Note: dones[idx] can be True (terminal transition is valid)
            #
            # We check dones at indices: idx-history_length+1, ..., idx-1
            # which corresponds to i in range(1, history_length) for check_idx = idx - i
            valid = True
            for i in range(1, self.history_length):
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
            fallback_indices = np.random.randint(self.history_length, max_index - 1, size=remaining)
            valid_indices.extend(fallback_indices.tolist())
        
        indices = np.array(valid_indices)
        
        # Memory layout: at index i, we store:
        # - frames[i]: the last frame of the state from which actions[i] was taken
        # - actions[i]: the action taken from the state ending at frames[i]
        # - rewards[i]: the reward received for taking actions[i]
        # - dones[i]: whether the episode ended after taking actions[i]
        
        states = np.array([self._get_state(index) for index in indices], dtype=np.float32) / 255.0
        
        # For terminal states, next_state doesn't matter (Q-value is 0)
        # But we still need to provide a valid next_state array
        # For non-terminal states, get the actual next state
        next_states = np.array([self._get_state(index + 1) for index in indices], dtype=np.float32) / 255.0
        
        # Use the sampled index directly to get the action, reward, and done for the transition
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        dones = self.dones[indices].astype(np.float32)
        
        return states, actions, rewards, next_states, dones

    def _get_state(self, index):
        """Get a state consisting of history_length consecutive frames ending at index.
        
        Parameters
        ----------
        index: int
            The index of the last frame in the state stack.
            
        Returns
        -------
        np.ndarray
            A stack of history_length frames with shape (frame_height, frame_width, history_length).
        """
        # Wrap index to handle circular buffer
        index = index % self.max_size
        
        # When buffer is full, we can always use circular indexing
        # When buffer is not full, we need to handle the case where index < history_length - 1
        if self.count < self.max_size and index < self.history_length - 1:
            # Buffer not full and not enough frames before this index, pad with first frame
            frames = []
            for i in range(self.history_length):
                frame_idx = index - (self.history_length - 1 - i)
                if frame_idx < 0:
                    frames.append(self.frames[0])  # Pad with first frame
                else:
                    frames.append(self.frames[frame_idx])
        else:
            # Buffer is full or we have enough frames, use circular indexing
            frames = [self.frames[(index - self.history_length + 1 + i) % self.max_size] for i in range(self.history_length)]
        return np.stack(frames, axis=-1)

    def __len__(self):
        return self.count


