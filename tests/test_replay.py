import pickle

import numpy as np
import pytest

from deeprl.core import ReplayMemory


def frame(value):
    return np.full((1, 1), value, dtype=np.uint8)


def test_terminal_and_truncated_transitions_do_not_cross_episodes():
    memory = ReplayMemory(16, 1, 1, 4)
    memory.append(frame(1), 0, 10, False, frame(2), terminal=False)
    memory.append(frame(2), 1, 20, True, frame(3), terminal=False)
    memory.append(frame(9), 2, 30, True, frame(10), terminal=True)
    states, actions, rewards, successors, terminals = memory.sample(128)
    expected_states = [[0, 0, 0, 1], [0, 0, 1, 2], [0, 0, 0, 9]]
    expected_next = [[0, 0, 1, 2], [0, 1, 2, 3], [0, 0, 0, 0]]
    assert set(actions) == {0, 1, 2}
    for row, action in enumerate(actions):
        np.testing.assert_allclose(states[row, 0, 0] * 255, expected_states[action])
        np.testing.assert_allclose(successors[row, 0, 0] * 255, expected_next[action])
        assert rewards[row] == (action + 1) * 10
        assert terminals[row] == (action == 2)


def test_ring_wrap_rejects_overwritten_history():
    memory = ReplayMemory(5, 1, 1, 3)
    for step in range(20):
        memory.append(frame(step), step, step, False, frame(step + 1))
    states, actions, _, successors, _ = memory.sample(128)
    assert set(actions) == {17, 18, 19}
    for row, action in enumerate(actions):
        np.testing.assert_allclose(states[row, 0, 0] * 255, [action - 2, action - 1, action])
        np.testing.assert_allclose(successors[row, 0, 0] * 255, [action - 1, action, action + 1])


def test_external_reset_preserves_true_successor_and_zero_pads_new_episode():
    memory = ReplayMemory(8, 1, 1, 3)
    memory.append(frame(1), 0, 0, False, frame(2))
    memory.start_new_episode()
    memory.append(frame(20), 1, 0, False, frame(21))
    states, actions, _, successors, terminals = memory.sample(32)
    for row, action in enumerate(actions):
        assert terminals[row] == 0
        expected = [0, 0, 1] if action == 0 else [0, 0, 20]
        expected_next = [0, 1, 2] if action == 0 else [0, 20, 21]
        np.testing.assert_allclose(states[row, 0, 0] * 255, expected)
        np.testing.assert_allclose(successors[row, 0, 0] * 255, expected_next)


def test_incomplete_transition_never_uses_invalid_fallback():
    memory = ReplayMemory(8, 1, 1, 3)
    with pytest.raises(ValueError):
        memory.sample(4)
    memory.append(frame(1), 0, 0, False)
    with pytest.raises(ValueError, match='no complete transitions'):
        memory.sample(4)
    memory.append(frame(2), 1, 0, False)
    assert np.all(memory.sample(16)[1] == 0)


def test_checkpoint_only_serializes_populated_frames():
    memory = ReplayMemory(100000, 1, 1, 3)
    memory.append(frame(1), 0, 0, False, frame(2))
    data = pickle.dumps(memory)
    assert len(data) < 10000
    restored = pickle.loads(data)
    assert restored.frames.shape == memory.frames.shape
    assert restored.count == 1 and restored.current == 1
    np.testing.assert_array_equal(restored._get_state(0), memory._get_state(0))
    assert restored.sample(1)[3][0, 0, 0, -1] == np.float32(2 / 255)
