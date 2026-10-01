# Atari RL Implementation - Space Invaders

A Deep Reinforcement Learning implementation for playing Atari Space Invaders using various DQN (Deep Q-Network) variants.

## Overview

This project implements several variants of the Deep Q-Network algorithm to train an agent to play the classic Atari game Space Invaders. The implementation is based on the seminal paper "Human-Level Control Through Deep Reinforcement Learning" by Mnih et al. (2015).

### Supported Models

- **Linear Q-Network**: A simple linear model for baseline comparison
- **Linear Double Q-Network**: Linear model with Double DQN improvements
- **Deep Q-Network (DQN)**: Standard DQN with convolutional neural network
- **Double DQN**: Reduces overestimation bias by decoupling action selection and evaluation
- **Dueling DQN**: Separates state value and advantage functions for better learning

## Project Structure

```
├── dqn_atari.py           # Main training script
├── deeprl/
│   ├── __init__.py        # Package initialization
│   ├── core.py            # Core classes (ReplayMemory, Sample, Preprocessor)
│   ├── dqn.py             # DQN Agent implementation
│   ├── policy.py          # Policy classes (ε-greedy, linear/exponential decay)
│   ├── preprocessors.py   # Atari frame preprocessors
│   ├── objectives.py      # Loss functions (Huber loss)
│   └── utils.py           # Utility functions
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Packaging and test configuration
├── setup.py               # Package setup file
├── tests/                 # Bounded regression and real Atari tests
└── README.md              # This file
```

## Installation

### Requirements

- 64-bit CPython **3.11-3.13**, with **Python 3.12 recommended**. Python 3.14 is not supported by this pinned stack.
- Exact runtime versions are listed in `requirements.txt`: TensorFlow 2.21, Keras 3, Gymnasium 1.3 and ALE 0.12.1, plus image/video dependencies.
- Use a fresh virtual environment rather than upgrading an old Gym/TensorFlow environment in place.

The pinned ALE package includes Atari ROMs. No AutoROM command or separate ROM download is required; the script explicitly registers ALE with Gymnasium. See the [ALE release notes](https://ale.farama.org/release_notes/index.html).

The installation and short training/video tests have been verified on **Windows x64, Python 3.12, CPU**. Linux x64 and Apple Silicon macOS have upstream wheels but have not been runtime-tested here. Intel macOS is not supported by this stack.

### Setup

1. Clone the repository:
```bash
git clone https://github.com/seealake/Atari-RL-implementation-in-Spaceinvader.git
cd Atari-RL-implementation-in-Spaceinvader
```

2. Create an environment and install dependencies.

**Windows PowerShell** (activation is not required):
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe dqn_atari.py --smoke-test
```

**Linux / Apple Silicon macOS**:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python dqn_atari.py --smoke-test
```

The smoke test runs at most **128 training steps**, with 256 replay frames, at most two CPU threads, short evaluation episodes, checkpoint saving and video recording. It checks the pipeline, not playing strength. On Windows, use `.\.venv\Scripts\python.exe` instead of `python` in the examples below unless the virtual environment is activated.

Installing the local package is optional: `python -m pip install -e .` also enables the `train-atari` command.

### GPU Notes

The default install works on CPU. Modern TensorFlow does **not** support NVIDIA GPU training on native Windows; use Linux or WSL2 for that. Inside Linux/WSL2, the matching optional CUDA dependencies can be installed with `python -m pip install "tensorflow[and-cuda]==2.21.0"`, with a compatible NVIDIA driver already installed. GPU training has not been verified locally. See the [TensorFlow installation guide](https://www.tensorflow.org/install/pip).

## Usage

### Training

After installing requirements, start default DQN training with one command:

```bash
python dqn_atari.py
```

This starts a full run, not a short test. Use `--smoke-test` first on modest hardware. The original 500,000-frame replay default is preserved: frames alone use about **3.3 GiB** of RAM, in addition to TensorFlow, batches and model weights. For a smaller full run, use `--memory-size 100000 --threads 2` (about 673 MiB for frames). The smoke profile uses only about 1.7 MiB for replay frames.

The existing model options remain available:

```bash
# Standard DQN
python dqn_atari.py --mode deep --iterations 1000000

# Double DQN
python dqn_atari.py --mode double --iterations 1000000

# Dueling DQN
python dqn_atari.py --mode dueling --iterations 1000000

# Linear Q-Network (baseline)
python dqn_atari.py --mode linear --iterations 1000000
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--env` | `ALE/SpaceInvaders-v5` | Atari environment name |
| `--mode` | `deep` | Model type: `linear`, `linear_double`, `deep`, `double`, `dueling` |
| `--iterations` | 1000000 | Additional environment steps, including when resuming |
| `--output` | `atari-v0` | Output directory for results |
| `--seed` | 0 | Random seed for reproducibility |
| `--checkpoint_dir` | `checkpoints` | Directory for saving checkpoints |
| `--checkpoint_freq` | 50000 | Checkpoint saving frequency |
| `--restart_freq` | 500000 | Environment reset/checkpoint interval; never rebuilds the model; 0 disables it |
| `--resume` | None | Existing run directory; restores its configuration and latest checkpoint |
| `--start_step` | 0 | Optional expected checkpoint step; cannot skip training without a checkpoint |
| `--checkpoint_file` | None | Specific checkpoint prefix, `.index` file or replay sidecar |
| `--memory-size` | 500000 | Replay capacity in single frames |
| `--batch-size` | 64 | Training batch size |
| `--burn-in` | 50000 | Completed environment steps before training begins |
| `--eval-freq` | 10000 | Periodic evaluation interval; 0 disables periodic evaluation |
| `--eval-episodes` | 10 | Episodes per periodic evaluation |
| `--final-eval-episodes` | 100 | Final evaluation episodes; 0 skips final evaluation |
| `--max-episode-length` | 27000 | Limit for both training and evaluation episodes |
| `--no-record-video` | Off | Disable video recording |
| `--video-freq` | 100 | Record the first training episode and then every N episodes |
| `--video-length` | 1000 | Maximum recorded frames per clip |
| `--keep-checkpoints` | 2 | Number of complete checkpoints to retain |
| `--threads` | 0 | TensorFlow CPU threads; 0 lets TensorFlow choose |
| `--smoke-test` | Off | Apply the small, bounded verification profile |

Run `python dqn_atari.py --help` for all options. Hyphenated and underscore spellings are both accepted. Defaults live in the `TrainingConfig` dataclass in `dqn_atari.py` and the effective configuration is saved as `config.json`.

### Resume Training

```bash
# Resume the latest checkpoint and train 100,000 additional steps
python dqn_atari.py --resume atari-v0/deep/ALE_SpaceInvaders-v5-run1 --iterations 100000

# Or choose a retained checkpoint explicitly (the prefix has no extension)
python dqn_atari.py --checkpoint-file atari-v0/deep/ALE_SpaceInvaders-v5-run1/checkpoints/ckpt-50000 --iterations 100000
```

Replace the example directory with the run directory printed by your command. Resume restores online/target weights, Adam state, replay, exploration state, completed steps, metrics and Python/NumPy RNG state. The emulator starts a new episode; this is not a bit-for-bit restoration of a mid-episode trajectory.

Keep the TensorFlow checkpoint files and the matching `extra_data_<step>.pkl.gz` together. Only load checkpoints you trust: replay sidecars use Python pickle. Legacy prefixes, uncompressed sidecars and weight-bearing `.pkl` files are recognized, but incompatible TensorFlow/Keras versions may not restore. Legacy files missing optimizer, target or replay state emit warnings and cannot provide an exact training resume. A metadata-only `.pkl` still needs matching model files. Missing or incompatible model weights fail rather than silently starting from random weights.

Pressing Ctrl+C saves a checkpoint and closes the environments/video recorder. Normal completion also saves a final checkpoint, even if the periodic interval has not been reached.

## Key Features

- **Experience Replay**: Stores transitions in a replay buffer for stable training
- **Target Network**: Preserves the original periodic soft updates; `--tau 1` selects hard updates
- **Frame Preprocessing**: Converts frames to grayscale, resizes to 84x84, and stacks 4 frames
- **Reward Clipping**: Clips rewards to [-1, 1] for stable gradients
- **Epsilon-Greedy Exploration**: Supports both linear and exponential decay schedules
- **Checkpointing**: Automatic saving and loading of training progress
- **Independent Evaluation**: Evaluation uses a separate environment and frame history
- **GPU Support**: Automatic GPU detection and memory growth configuration

## Training Outputs

Each run writes to `atari-v0/{mode}/{environment}-runN/` by default:

- `Training_loss.png`: Smoothed training loss curve
- `{mode}_learning_curve.png`: Learning curve showing mean reward over time
- `final_results.txt`: Final evaluation results
- `dqn_training.log`: Detailed training logs
- `config.json`: Effective training configuration
- `final_model.keras`: Reloadable Keras model for inference or inspection
- `checkpoints/`: TensorFlow checkpoints and compressed replay/training sidecars
- `videos/session-*/`: Recorded MP4 clips; resume never overwrites previous clips

Loss plots require at least one gradient update; learning curves require at least one periodic evaluation. Short tests are not expected to achieve high scores.

## Algorithm Details

### DQN Architecture

The convolutional neural network architecture follows the original DQN paper:
- Conv2D: 32 filters, 8x8 kernel, stride 4, ReLU
- Conv2D: 64 filters, 4x4 kernel, stride 2, ReLU
- Conv2D: 64 filters, 3x3 kernel, stride 1, ReLU
- Dense: 512 units, ReLU
- Dense: num_actions (output layer)

### Hyperparameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| Discount Factor (gamma) | 0.99 | Future reward discount |
| Learning Rate | 3e-4 (Adam) | Optimizer learning rate used in DQNAgent |
| Replay Buffer Size | 500,000 | Original capacity; lower `--memory-size` to reduce RAM use |
| Batch Size | 64 (in main script) | Training batch size |
| Target Update Frequency | 5,000 steps | Steps between target network updates |
| Burn-in Period | 50,000 steps | Steps before training starts |
| Training Frequency | Every 4 steps | Steps between gradient updates |
| Soft Update tau | 0.001 | Original target-network update weight; `--tau 1` selects full synchronization |
| Initial epsilon | 1.0 | Starting exploration rate |
| Final epsilon | 0.05 | Minimum exploration rate |
| Epsilon Decay Rate | 1e-5 | Exponential decay rate for epsilon |

> **Note**: The `DQNAgent` class has a default `batch_size=32`, but the entrypoint passes `TrainingConfig.batch_size=64`. All modes train with Adam and Huber loss. Environment time limits reset frame history but do not suppress TD bootstrapping; only true termination does that.

## Tests

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

The suite covers replay boundaries/wraparound, TD targets, target/optimizer checkpoint restoration, all five models on short real Atari runs, Keras model reload, and a CLI video/resume test that decodes MP4 frames and checks that they change. Tests limit CPU threads and use small buffers. Use `python -m pytest -q -m "not integration"` to skip the real Atari/CLI tests.

## License

This project is for educational and research purposes.

## References

- [Human-Level Control Through Deep Reinforcement Learning](https://www.nature.com/articles/nature14236) - Mnih et al., 2015
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) - van Hasselt et al., 2015
- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581) - Wang et al., 2015
