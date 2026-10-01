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
    """Frame-efficient replay with separate episode boundaries and TD terminals."""

    def __init__(self, max_size, frame_height, frame_width, history_length=4):
        if history_length < 1 or max_size < history_length + 1:
            raise ValueError("Replay capacity must be greater than history_length")
        if frame_height < 1 or frame_width < 1:
            raise ValueError("Frame dimensions must be positive")
        self.max_size = max_size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.history_length = history_length
        self.frames = np.zeros((max_size, frame_height, frame_width), dtype=np.uint8)
        self.actions = np.zeros(max_size, dtype=np.int32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=bool)
        self.terminals = np.zeros(max_size, dtype=bool)
        self.episode_starts = np.zeros(max_size, dtype=bool)
        self.current = 0
        self.count = 0
        self._new_episode = True
        # Only the newest transition and non-terminal episode boundaries need
        # an extra frame. All other successors already live in the ring buffer.
        self._next_frames = {}

    def append(self, frame, action, reward, done, next_frame=None, terminal=None):
        if np.shape(frame) != (self.frame_height, self.frame_width):
            raise ValueError("Unexpected replay frame shape")
        if next_frame is not None and np.shape(next_frame) != np.shape(frame):
            raise ValueError("Unexpected successor frame shape")
        terminal = bool(done) if terminal is None else bool(terminal)
        if terminal and not done:
            raise ValueError("A terminal transition must end the episode")
        if self.count and not self._new_episode:
            self._next_frames.pop((self.current - 1) % self.max_size, None)
        self._next_frames.pop(self.current, None)
        self.frames[self.current] = frame
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.dones[self.current] = done
        self.terminals[self.current] = terminal
        self.episode_starts[self.current] = self._new_episode
        if next_frame is not None and not terminal:
            self._next_frames[self.current] = np.array(next_frame, dtype=np.uint8, copy=True)
        self._new_episode = bool(done)
        self.count = min(self.count + 1, self.max_size)
        self.current = (self.current + 1) % self.max_size

    def start_new_episode(self):
        """Break frame history after an external reset, including resume."""
        if self.count:
            self.dones[(self.current - 1) % self.max_size] = True
        self._new_episode = True

    def _has_history(self, index):
        oldest = self.current if self.count == self.max_size else 0
        for offset in range(self.history_length):
            cursor = (index - offset) % self.max_size
            if self.episode_starts[cursor] or offset == self.history_length - 1:
                return True
            if cursor == oldest:
                return False
        return False

    def _is_valid(self, index):
        if not self._has_history(index):
            return False
        if self.terminals[index] or index in self._next_frames:
            return True
        newest = (self.current - 1) % self.max_size
        return index != newest and not self.dones[index]

    def sample(self, batch_size):
        if batch_size < 1 or not self.count:
            raise ValueError("Cannot sample an empty replay buffer or an empty batch")
        indices = []
        for index in np.random.randint(self.count, size=batch_size * 20):
            if self._is_valid(index):
                indices.append(index)
                if len(indices) == batch_size:
                    break
        if len(indices) < batch_size:
            valid = [i for i in range(self.count) if self._is_valid(i)]
            if not valid:
                raise ValueError("Replay buffer has no complete transitions")
            indices = np.random.choice(valid, size=batch_size).tolist()

        states = np.stack([self._get_state(i) for i in indices])
        next_states = np.zeros_like(states)
        for row, index in enumerate(indices):
            if not self.terminals[index]:
                next_frame = self._next_frames.get(index)
                if next_frame is None:
                    next_frame = self.frames[(index + 1) % self.max_size]
                next_states[row, ..., :-1] = states[row, ..., 1:]
                next_states[row, ..., -1] = next_frame
        return (
            states.astype(np.float32) / 255.0,
            self.actions[indices],
            self.rewards[indices],
            next_states.astype(np.float32) / 255.0,
            self.terminals[indices].astype(np.float32),
        )

    def _get_state(self, index):
        """Reconstruct history, zero-padding only at actual episode starts."""
        if not 0 <= index < self.count or not self._has_history(index):
            raise ValueError("The requested frame history has been overwritten")
        state = np.zeros(
            (self.frame_height, self.frame_width, self.history_length), dtype=np.uint8
        )
        for offset in range(self.history_length):
            cursor = (index - offset) % self.max_size
            state[..., self.history_length - 1 - offset] = self.frames[cursor]
            if self.episode_starts[cursor]:
                break
        return state

    def __getstate__(self):
        state = self.__dict__.copy()
        for name, value in state.items():
            if isinstance(value, np.ndarray):
                state[name] = value[:self.count]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Older checkpoints did not distinguish truncation from termination.
        if 'terminals' not in state:
            self.terminals = self.dones.copy()
            self.episode_starts = np.roll(self.dones, 1)
            self.episode_starts[self.current if self.count == self.max_size else 0] = self.count < self.max_size
            self._next_frames, self._new_episode = {}, True
        for name, value in list(self.__dict__.items()):
            if isinstance(value, np.ndarray) and len(value) < self.max_size:
                expanded = np.zeros((self.max_size, *value.shape[1:]), dtype=value.dtype)
                expanded[:len(value)] = value
                setattr(self, name, expanded)

    def __len__(self):
        return self.count


