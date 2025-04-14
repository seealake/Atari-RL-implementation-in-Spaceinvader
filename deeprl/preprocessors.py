"""Suggested Preprocessors."""

import numpy as np
from PIL import Image

from deeprl import utils
from deeprl.core import Preprocessor


class HistoryPreprocessor(Preprocessor):
    """Keeps the last k states.

    Useful for domains where you need velocities, but the state
    contains only positions.

    When the environment starts, this will just fill the initial
    sequence values with zeros k times.

    Parameters
    ----------
    history_length: int
      Number of previous states to prepend to state being processed.

    """

    def __init__(self, history_length=4):
        """Initialize the HistoryPreprocessor with the desired history length."""
        self.history_length = history_length
        self.history = []

    def process_state_for_network(self, state):
        """Returns a stack of the last `history_length` frames."""
        if len(self.history) == 0:
            self.history = [np.zeros_like(state)] * self.history_length
        self.history.pop(0)
        self.history.append(state)
        return np.stack(self.history, axis=-1)

    def reset(self):
        """Reset the history. Useful when starting a new episode."""
        self.history = []

    def get_config(self):
        return {'history_length': self.history_length}


class AtariPreprocessor(Preprocessor):
    """Converts images to greyscale and downscales them to a specified size.

    Based on the preprocessing step described in:

    @article{mnih15_human_level_contr_throug_deep_reinf_learn,
    author =	 {Volodymyr Mnih and Koray Kavukcuoglu and David
                  Silver and Andrei A. Rusu and Joel Veness and Marc
                  G. Bellemare and Alex Graves and Martin Riedmiller
                  and Andreas K. Fidjeland and Georg Ostrovski and
                  Stig Petersen and Charles Beattie and Amir Sadik and
                  Ioannis Antonoglou and Helen King and Dharshan
                  Kumaran and Daan Wierstra and Shane Legg and Demis
                  Hassabis},
    title =	 {Human-Level Control Through Deep Reinforcement
                  Learning},
    journal =	 {Nature},
    volume =	 518,
    number =	 7540,
    pages =	 {529-533},
    year =	 2015,
    doi =        {10.1038/nature14236},
    url =	 {http://dx.doi.org/10.1038/nature14236},
    }

    Parameters
    ----------
    new_size: 2 element tuple
      The size that each image in the state should be scaled to. e.g., (84, 84)
    """

    def __init__(self, new_size=(84, 84)):
        """Initialize the AtariPreprocessor with the desired size."""
        self.new_size = new_size

    def process_state_for_memory(self, state):
        """Convert the state to greyscale, resize it, and store as uint8."""
        state = self._preprocess_frame(state)
        return state.astype(np.uint8)  # Store for memory as uint8 to save space

    def process_state_for_network(self, state):
        """Convert the state to greyscale, resize it, and store as float32."""
        gray = np.mean(state, axis=2).astype(np.uint8)
        resized = Image.fromarray(gray).resize(self.new_size)
        normalized = np.array(resized).astype(np.float32) / 255.0    
        return normalized 

    def _preprocess_frame(self, frame):
        """Convert the frame to greyscale and resize it."""
        frame = np.mean(frame, axis=2)  # Convert to greyscale
        frame = Image.fromarray(frame)
        frame = frame.resize(self.new_size)
        return np.array(frame)

    def process_batch(self, samples):
        """Convert the batch of states from uint8 to float32."""
        return [self.process_state_for_network(sample) for sample in samples]

    def process_reward(self, reward):
        """Clip the reward between -1 and 1."""
        return np.clip(reward, -1, 1)


class PreprocessorSequence(Preprocessor):
    """A class to stack multiple preprocessors together.

    This class allows you to chain preprocessors. For example, it can
    call AtariPreprocessor followed by HistoryPreprocessor to combine
    both of their preprocessing functionalities.

    Parameters
    ----------
    preprocessors: list
      List of preprocessors to be applied in sequence.
    """
    
    def __init__(self, preprocessors):
        """Initialize the PreprocessorSequence with a list of preprocessors."""
        self.preprocessors = preprocessors

    def process_state_for_network(self, state):
        """Pass the state through all preprocessors in sequence."""
        for preprocessor in self.preprocessors:
            state = preprocessor.process_state_for_network(state)
        return state

    def process_state_for_memory(self, state):
        """Pass the state through all preprocessors for memory."""
        for preprocessor in self.preprocessors:
            state = preprocessor.process_state_for_memory(state)
        return state

    def process_batch(self, samples):
        """Pass the batch through all preprocessors for batch processing."""
        for preprocessor in self.preprocessors:
            samples = preprocessor.process_batch(samples)
        return samples

    def process_reward(self, reward):
        """Pass the reward through all preprocessors."""
        for preprocessor in self.preprocessors:
            reward = preprocessor.process_reward(reward)
        return reward

    def reset(self):
        """Reset all preprocessors in the sequence."""
        for preprocessor in self.preprocessors:
            preprocessor.reset()

    def get_config(self):
        """Return configurations of all preprocessors."""
        return [preprocessor.get_config() for preprocessor in self.preprocessors]
