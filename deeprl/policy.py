"""RL Policy classes.

We have provided you with a base policy class, some example
implementations and some unimplemented classes that should be useful
in your code.
"""
import numpy as np


class Policy:
    """Base class representing an MDP policy.

    Policies are used by the agent to choose actions.

    Policies are designed to be stacked to get interesting behaviors
    of choices. For instance, in a discrete action space, the lowest
    level policy may take in Q-values and select the action index
    corresponding to the largest value. If this policy is wrapped in
    an epsilon greedy policy then with some probability epsilon, a
    random action will be chosen.
    """

    def select_action(self, **kwargs):
        """Used by agents to select actions.

        Returns
        -------
        Any:
          An object representing the chosen action. Type depends on
          the hierarchy of policy instances.
        """
        raise NotImplementedError('This method should be overridden.')


class UniformRandomPolicy(Policy):
    """Chooses a discrete action with uniform random probability.

    Parameters
    ----------
    num_actions: int
      Number of actions to choose from. Must be > 0.
    """

    def __init__(self, num_actions):
        assert num_actions >= 1
        self.num_actions = num_actions

    def select_action(self, **kwargs):
        """Return a random action index."""
        return np.random.randint(0, self.num_actions)

    def get_config(self):  # noqa: D102
        return {'num_actions': self.num_actions}


class GreedyPolicy(Policy):
    """Always returns best action according to Q-values.

    This is a pure exploitation policy.
    """

    def select_action(self, q_values, **kwargs):
        """Selects the action with the highest Q-value.

        Parameters
        ----------
        q_values: array-like
          Array-like structure of floats representing the Q-values for
          each action.

        Returns
        -------
        int:
          The action index chosen.
        """
        return np.argmax(q_values)


class GreedyEpsilonPolicy(Policy):
    """Selects greedy action or with some probability a random action.

    Standard greedy-epsilon implementation. With probability epsilon,
    choose a random action. Otherwise, choose the greedy action.

    Parameters
    ----------
    epsilon: float
     Initial probability of choosing a random action. Can be changed
     over time.
    """
    
    def __init__(self, epsilon):
        """Initialize the Greedy-Epsilon policy."""
        self.epsilon = epsilon

    def select_action(self, q_values, **kwargs):
        """Selects action based on epsilon-greedy policy.

        Parameters
        ----------
        q_values: np.array
          Array-like structure of floats representing the Q-values for
          each action. Can be 1D (num_actions,) or 2D (batch_size, num_actions).

        Returns
        -------
        int:
          The action index chosen.
        """
        q_values = np.atleast_2d(q_values)
        num_actions = q_values.shape[-1]
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(0, num_actions))
        else:
            # Flatten to 1D if needed before argmax to get correct action index
            return int(np.argmax(q_values.flatten()[:num_actions]))

    def get_config(self):
        return {'epsilon': self.epsilon}

    def set_config(self, config):
        self.epsilon = config['epsilon']


class LinearDecayGreedyEpsilonPolicy(Policy):
    def __init__(self, epsilon_policy, start_value, end_value, num_steps):
        self.epsilon_policy = epsilon_policy
        self.start_value = start_value
        self.end_value = end_value
        self.num_steps = num_steps
        self.current_step = 0

    def select_action(self, q_values, is_training=True):
        if is_training:
            # Avoid division by zero
            if self.num_steps > 0:
                epsilon = self.start_value - (self.start_value - self.end_value) * (self.current_step / self.num_steps)
            else:
                epsilon = self.end_value
            epsilon = max(epsilon, self.end_value)
            self.epsilon_policy.epsilon = epsilon
            self.current_step += 1
        return self.epsilon_policy.select_action(q_values)

    def reset(self):
        self.current_step = 0

    def get_config(self):
        return {
            'epsilon_policy': self.epsilon_policy.get_config(),
            'start_value': self.start_value,
            'end_value': self.end_value,
            'num_steps': self.num_steps,
            'current_step': self.current_step
        }

    def set_config(self, config):
        self.start_value = config['start_value']
        self.end_value = config['end_value']
        self.num_steps = config['num_steps']
        self.current_step = config['current_step']
        if hasattr(self.epsilon_policy, 'set_config'):
            self.epsilon_policy.set_config(config['epsilon_policy'])
        else:
            self.epsilon_policy.epsilon = config['epsilon_policy'].get('epsilon', self.end_value)


class ExponentialDecayGreedyEpsilonPolicy(Policy):
    def __init__(self, epsilon_policy, start_value, end_value, decay_rate):
        self.epsilon_policy = epsilon_policy
        self.start_value = start_value
        self.end_value = end_value
        self.decay_rate = decay_rate
        self.step = 0

    def select_action(self, q_values, is_training=True):
        if is_training:
            epsilon = self.end_value + (self.start_value - self.end_value) * np.exp(-self.decay_rate * self.step)
            self.epsilon_policy.epsilon = max(epsilon, self.end_value)
            self.step += 1
        return self.epsilon_policy.select_action(q_values)

    def reset(self):
        self.step = 0

    def get_config(self):
        return {
            'epsilon_policy': self.epsilon_policy.get_config(),
            'start_value': self.start_value,
            'end_value': self.end_value,
            'decay_rate': self.decay_rate,
            'step': self.step
        }

    def set_config(self, config):
        self.start_value = config['start_value']
        self.end_value = config['end_value']
        self.decay_rate = config['decay_rate']
        self.step = config['step']
        if hasattr(self.epsilon_policy, 'set_config'):
            self.epsilon_policy.set_config(config['epsilon_policy'])
        else:
            self.epsilon_policy.epsilon = config['epsilon_policy'].get('epsilon', self.end_value)


