import itertools

import pytest
import tensorflow as tf

from deeprl.core import ReplayMemory
from deeprl.dqn import DQNAgent
from deeprl.policy import ExponentialDecayGreedyEpsilonPolicy, GreedyEpsilonPolicy
from deeprl.preprocessors import AtariPreprocessor, HistoryPreprocessor, PreprocessorSequence
from dqn_atari import create_linear_model


tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)


@pytest.fixture
def agent_factory(tmp_path):
    agents, numbers = [], itertools.count()

    def create(model=None, input_shape=(4, 4, 2), num_actions=2, **kwargs):
        directory = tmp_path / str(next(numbers))
        defaults = dict(num_burn_in=2, batch_size=2, train_freq=1, target_update_freq=4)
        defaults.update(kwargs)
        agent = DQNAgent(
            model=model if model is not None else create_linear_model(input_shape, num_actions),
            input_shape=input_shape, num_actions=num_actions,
            preprocessor=PreprocessorSequence([
                AtariPreprocessor(input_shape[:2]), HistoryPreprocessor(input_shape[-1]),
            ]),
            memory=ReplayMemory(64, *input_shape[:2], input_shape[-1]),
            policy=ExponentialDecayGreedyEpsilonPolicy(GreedyEpsilonPolicy(1), 1, 0.05, 0.01),
            output_dir=directory, checkpoint_dir=directory / 'checkpoints', **defaults,
        )
        agents.append(agent)
        return agent

    yield create
    for agent in agents:
        agent.close()
    tf.keras.backend.clear_session()
