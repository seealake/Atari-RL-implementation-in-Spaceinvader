Atari RL Implementation - Space Invaders

Overview

This project trains an agent to play Atari Space Invaders with several Deep Q-Network variants. It is based on "Human-Level Control Through Deep Reinforcement Learning" by Mnih et al. (2015).

Supported models

linear: A linear Q-network for baseline comparison.

linear_double: A linear Q-network with Double DQN action selection.

deep: Standard DQN with a convolutional neural network.

double: Double DQN, which separates action selection from target-value evaluation.

dueling: Dueling DQN, which separates state-value and advantage streams.

Installation requirements

Use 64-bit CPython 3.11-3.13. Python 3.12 is recommended. Python 3.14 is not supported by the pinned dependencies.

Runtime versions are pinned in requirements.txt, including TensorFlow 2.21.0, Keras 3.15.1, Gymnasium 1.3.0, ALE 0.12.1, and image/video dependencies. Create a fresh virtual environment instead of upgrading an old Gym/TensorFlow environment in place.

The pinned ALE package includes Atari ROMs. No separate ROM download or AutoROM command is needed. The training script explicitly registers ALE with Gymnasium.

Installation, short training runs, video recording and checkpoint recovery have been tested on Windows x64 with Python 3.12 and CPU execution. Linux x64 and Apple Silicon macOS have upstream wheels but have not been runtime-tested here. Intel macOS is not supported by this stack.

Clone the repository

Replace REPOSITORY_URL with the clone URL from this repository's GitHub Code menu.

git clone REPOSITORY_URL

cd Atari-RL-implementation-in-Spaceinvader

Windows PowerShell setup

Run these commands from the repository directory. Virtual-environment activation is not required.

py -3.12 -m venv .venv

./.venv/Scripts/python.exe -m pip install --upgrade pip

./.venv/Scripts/python.exe -m pip install -r requirements.txt

./.venv/Scripts/python.exe dqn_atari.py --smoke-test

Linux / Apple Silicon macOS setup

python3.12 -m venv .venv

source .venv/bin/activate

python -m pip install --upgrade pip

python -m pip install -r requirements.txt

python dqn_atari.py --smoke-test

The smoke test runs at most 128 training steps, with at most 256 replay frames, two CPU threads, short evaluation episodes, checkpoint saving and video recording. It checks the pipeline, not playing strength.

On Windows, replace python in the remaining examples with ./.venv/Scripts/python.exe unless the virtual environment is activated.

Installing the local package is optional. The following command also enables the train-atari entrypoint:

python -m pip install -e .

GPU notes

The default installation works on CPU. Modern TensorFlow does not support NVIDIA GPU training on native Windows. Use Linux or WSL2 for that.

Inside Linux/WSL2, with a compatible NVIDIA driver already installed, the matching optional CUDA dependencies can be installed with:

python -m pip install "tensorflow[and-cuda]==2.21.0"

GPU training has not been verified locally. See the TensorFlow installation guide in the references below.

Training

After installing requirements, start default DQN training with:

python dqn_atari.py

This starts a full run, not a short test. On modest hardware, use --smoke-test first.

The original 500,000-frame replay default is preserved. With the default image size, replay frames alone use about 3.3 GiB of RAM, in addition to TensorFlow, batches and model weights. For a smaller full run, use:

python dqn_atari.py --memory-size 100000 --threads 2

That setting uses about 673 MiB for replay frames. The default smoke profile uses about 1.7 MiB.

To select a model:

python dqn_atari.py --mode deep --iterations 1000000

python dqn_atari.py --mode double --iterations 1000000

python dqn_atari.py --mode dueling --iterations 1000000

python dqn_atari.py --mode linear --iterations 1000000

python dqn_atari.py --mode linear_double --iterations 1000000

Configuration

Defaults live in the TrainingConfig class in dqn_atari.py. The effective configuration is saved to config.json in the run directory. Hyphenated and underscore spellings of options are both accepted.

For the full option list:

python dqn_atari.py --help

--env defaults to ALE/SpaceInvaders-v5. --mode defaults to deep.

--iterations defaults to 1000000 additional environment steps, including when resuming.

--output defaults to atari-v0. --seed defaults to 0.

--memory-size defaults to 500000 frames. --batch-size defaults to 64. --burn-in defaults to 50000 completed steps before gradient updates start.

--checkpoint-dir defaults to checkpoints, relative to the run directory. Nested subdirectories are supported. --checkpoint-freq defaults to 50000 steps; 0 disables periodic saving but not final saving. --keep-checkpoints defaults to 2.

--restart-freq defaults to 500000 steps. It resets the environment and saves a checkpoint without rebuilding the model or optimizer. Set it to 0 to disable these resets.

--eval-freq defaults to 10000 steps. --eval-episodes defaults to 10. Set --eval-freq to 0 to disable periodic evaluation.

--final-eval-episodes defaults to 100; 0 skips final evaluation. --max-episode-length defaults to 27000 steps for training and evaluation.

Video recording is enabled by default. --video-freq defaults to 100, recording the first training episode and then every 100 episodes. --video-length defaults to 1000 frames per clip. --no-record-video disables recording.

--threads defaults to 0, letting TensorFlow choose CPU thread counts. --smoke-test applies the bounded verification profile described above.

Resume training

To restore the latest checkpoint and train 100000 additional steps:

python dqn_atari.py --resume atari-v0/deep/ALE_SpaceInvaders-v5-run1 --iterations 100000

To select a retained checkpoint explicitly:

python dqn_atari.py --checkpoint-file atari-v0/deep/ALE_SpaceInvaders-v5-run1/checkpoints/ckpt-50000 --iterations 100000

Use the actual run directory printed by your training command. Choose either --resume or --checkpoint-file, not both. The checkpoint prefix has no extension; an .index file or matching replay sidecar can also be supplied.

--start-step is an optional expected checkpoint step. It cannot skip training without loading a checkpoint.

Resume restores online and target weights, Adam state, replay, exploration state, completed steps, metrics, and Python/NumPy RNG state. The emulator starts a new episode; this is not a bit-for-bit restoration of a mid-episode trajectory.

Keep the TensorFlow checkpoint files and matching extra_data_STEP.pkl.gz sidecar together. Only load checkpoints you trust: replay sidecars use Python pickle.

Legacy prefixes, uncompressed sidecars and weight-bearing .pkl files are recognized, but incompatible TensorFlow/Keras versions may not restore. Legacy files missing optimizer, target or replay state emit warnings and cannot provide an exact training resume. A metadata-only .pkl still needs matching model files. Missing or incompatible model weights fail instead of silently starting from random weights.

During training, Ctrl+C saves a checkpoint and closes the environments and video recorder. Normal completion also saves a final checkpoint, even if the periodic interval has not been reached.

Training outputs

Each new run gets its own directory under atari-v0/MODE/ENVIRONMENT-runN/ by default.

config.json: Effective training configuration.

Training_loss.png: Smoothed loss curve, generated after at least one gradient update.

MODE_learning_curve.png: Evaluation rewards over time, generated after at least one periodic evaluation.

final_results.txt: Completed step count, gradient-update count and final evaluation results.

dqn_training.log: Training log.

final_model.keras: Reloadable Keras model for inference or inspection.

checkpoints/: TensorFlow checkpoints and compressed replay/training sidecars.

videos/session-TIMESTAMP/: Recorded MP4 clips. Resuming creates a new video session rather than overwriting previous clips.

Short tests are not expected to achieve high game scores.

Algorithm and defaults

Frames are converted to grayscale, resized to 84x84, normalized, and stacked with a history length of 4. Replay stores single uint8 frames and reconstructs histories when sampling.

The convolutional network uses three ReLU convolutional layers: 32 filters with an 8x8 kernel and stride 4; 64 filters with a 4x4 kernel and stride 2; and 64 filters with a 3x3 kernel and stride 1. These are followed by a 512-unit dense layer and an output for each action. The dueling variant uses separate value and advantage streams.

All modes train with Adam at learning rate 3e-4 and Huber loss. Gamma defaults to 0.99. Gradient updates occur every 4 environment steps after burn-in. Rewards are clipped to [-1, 1], and gradient norms are clipped to 5.

The original target-network defaults are preserved: one soft update every 5000 steps with tau=0.001. Setting --tau 1 selects full synchronization at each target-update interval.

The default epsilon-greedy policy decays exponentially from 1.0 toward 0.05, with decay rate 1e-5. A linear-decay policy is also available in deeprl/policy.py.

The DQNAgent class itself defaults to batch_size=32; the entrypoint passes TrainingConfig.batch_size=64.

Evaluation uses a separate environment and frame history. Environment time limits reset history but do not suppress TD bootstrapping; only true termination does that.

Project structure

dqn_atari.py: Training entrypoint, configuration, model builders and environment creation.

deeprl/core.py: Replay memory and core classes.

deeprl/dqn.py: Agent, training, evaluation and checkpoint handling.

deeprl/policy.py: Exploration policies.

deeprl/preprocessors.py: Frame preprocessing and stacking.

deeprl/objectives.py: Alternative Huber-loss functions.

deeprl/utils.py: Network-update and TensorFlow helpers.

requirements.txt: Pinned runtime dependencies.

pyproject.toml and setup.py: Package installation and test configuration.

tests/: Bounded regression and real Atari tests.

Tests

python -m pip install -e ".[test]"

python -m pytest -q

The suite covers replay boundaries and wraparound, TD targets, target/optimizer restoration, all five models on short real Atari runs, Keras model reload, and CLI video/resume checks. MP4 clips are decoded to verify nonblank and changing frames. Tests limit CPU threads and use small buffers.

To skip the real Atari and CLI tests:

python -m pytest -q -m "not integration"

License

This project is for educational and research purposes.

References

Human-Level Control Through Deep Reinforcement Learning, Mnih et al. (2015):
https://www.nature.com/articles/nature14236

Deep Reinforcement Learning with Double Q-learning, van Hasselt et al. (2015):
https://arxiv.org/abs/1509.06461

Dueling Network Architectures for Deep Reinforcement Learning, Wang et al. (2015):
https://arxiv.org/abs/1511.06581

ALE release notes:
https://ale.farama.org/release_notes/index.html

TensorFlow installation guide:
https://www.tensorflow.org/install/pip
