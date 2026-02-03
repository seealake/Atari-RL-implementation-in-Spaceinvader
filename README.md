# Atari RL Implementation - Space Invaders

A Deep Reinforcement Learning implementation for playing Atari Space Invaders using various DQN (Deep Q-Network) variants.

## 🎮 Overview

This project implements several variants of the Deep Q-Network algorithm to train an agent to play the classic Atari game Space Invaders. The implementation is based on the seminal paper "Human-Level Control Through Deep Reinforcement Learning" by Mnih et al. (2015).

### Supported Models

- **Linear Q-Network**: A simple linear model for baseline comparison
- **Linear Double Q-Network**: Linear model with Double DQN improvements
- **Deep Q-Network (DQN)**: Standard DQN with convolutional neural network
- **Double DQN**: Reduces overestimation bias by decoupling action selection and evaluation
- **Dueling DQN**: Separates state value and advantage functions for better learning

## 📁 Project Structure

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
├── setup.py               # Package setup file
└── README.md              # This file
```

## 🛠️ Installation

### Requirements

- Python >= 3.8
- TensorFlow >= 2.0.0
- OpenAI Gym (with Atari support)
- Gymnasium (with Atari support)
- NumPy
- Pillow
- Matplotlib
- Keras
- SciPy
- h5py
- semver

### Setup

1. Clone the repository:
```bash
git clone https://github.com/seealake/Atari-RL-implementation-in-Spaceinvader.git
cd Atari-RL-implementation-in-Spaceinvader
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

## 🚀 Usage

### Training

Train a DQN agent with different model architectures:

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
| `--mode` | Required | Model type: `linear`, `linear_double`, `deep`, `double`, `dueling` |
| `--iterations` | 1000000 | Number of training iterations |
| `--output` | `atari-v0` | Output directory for results |
| `--seed` | 0 | Random seed for reproducibility |
| `--checkpoint_dir` | `checkpoints` | Directory for saving checkpoints |
| `--checkpoint_freq` | 50000 | Checkpoint saving frequency |
| `--restart_freq` | 500000 | Training restart frequency |
| `--start_step` | 0 | Step to resume training from |
| `--checkpoint_file` | None | Specific checkpoint file to load |

### Resume Training

```bash
# Resume from a specific step
python dqn_atari.py --mode deep --start_step 100000

# Resume from a specific checkpoint file
python dqn_atari.py --mode deep --checkpoint_file checkpoint_100000.pkl
```

## 🔧 Key Features

- **Experience Replay**: Stores transitions in a replay buffer for stable training
- **Target Network**: Uses a separate target network with soft updates for stability
- **Frame Preprocessing**: Converts frames to grayscale, resizes to 84x84, and stacks 4 frames
- **Reward Clipping**: Clips rewards to [-1, 1] for stable gradients
- **ε-Greedy Exploration**: Supports both linear and exponential decay schedules
- **Checkpointing**: Automatic saving and loading of training progress
- **GPU Support**: Automatic GPU detection and memory growth configuration

## 📊 Training Outputs

During training, the following outputs are generated:

- `Training_loss.png`: Smoothed training loss curve
- `{mode}_learning_curve.png`: Learning curve showing mean reward over time
- `final_results.txt`: Final evaluation results
- `dqn_training.log`: Detailed training logs
- `checkpoints/`: Model checkpoints for resuming training
- `videos/`: Recorded gameplay videos

## 🧠 Algorithm Details

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
| Discount Factor (γ) | 0.99 | Future reward discount |
| Learning Rate | 0.00025 (RMSprop) | Optimizer learning rate |
| Replay Buffer Size | 500,000 | Maximum experiences stored |
| Batch Size | 64 (in main script) | Training batch size |
| Target Update Frequency | 5,000 steps | Steps between target network updates |
| Burn-in Period | 50,000 steps | Steps before training starts |
| Training Frequency | Every 4 steps | Steps between gradient updates |
| Soft Update τ | 0.001 | Target network soft update weight |
| Initial ε | 1.0 | Starting exploration rate |
| Final ε | 0.05 | Minimum exploration rate |
| ε Decay Rate | 1e-5 | Exponential decay rate for ε |

> **Note**: The `DQNAgent` class has a default `batch_size=32`, but `dqn_atari.py` overrides this to `batch_size=64`.

## 📝 License

This project is for educational and research purposes.

## 🔗 References

- [Human-Level Control Through Deep Reinforcement Learning](https://www.nature.com/articles/nature14236) - Mnih et al., 2015
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) - van Hasselt et al., 2015
- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581) - Wang et al., 2015
