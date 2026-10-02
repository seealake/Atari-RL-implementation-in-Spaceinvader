# Atari RL - Space Invaders

Educational TensorFlow implementation of five Q-network variants: linear, linear_double, deep (default DQN), double and dueling.

## Installation

Use 64-bit Python 3.11-3.13; Python 3.12 is recommended. Dependencies are pinned in requirements.txt, and Atari ROMs are included. Clone or download the repository from GitHub's Code menu, then open its directory.

Windows PowerShell:

py -3.12 -m venv .venv

./.venv/Scripts/python.exe -m pip install -r requirements.txt

Linux / Apple Silicon macOS:

python3.12 -m venv .venv

source .venv/bin/activate

python -m pip install -r requirements.txt

On Windows, replace python in the following commands with ./.venv/Scripts/python.exe unless the virtual environment is activated.

Verified on Windows x64, Python 3.12 and CPU. Other platforms and GPU training have not been runtime-tested. Modern TensorFlow uses CPU on native Windows; NVIDIA GPU training requires Linux or WSL2.

## Training

First check the pipeline with at most 128 training steps, a small replay buffer and two CPU threads:

python dqn_atari.py --smoke-test

Start a full run with the default settings:

python dqn_atari.py

The default replay frames alone use about 3.3 GiB of RAM. For a smaller buffer and limited CPU threads:

python dqn_atari.py --memory-size 100000 --threads 2

Select a model with --mode and disable video with --no-record-video. See all options with:

python dqn_atari.py --help

Configuration defaults live in TrainingConfig in dqn_atari.py. Each run saves its configuration, logs, plots, MP4 clips, final_model.keras and checkpoints under atari-v0/MODE/ENVIRONMENT-runN/. Short tests verify execution, not playing strength.

## Resume Training

Use the run directory printed during training. --iterations specifies additional training steps:

python dqn_atari.py --resume atari-v0/deep/ALE_SpaceInvaders-v5-run1 --iterations 100000

To select a specific checkpoint, use --checkpoint-file instead of --resume. Keep the TensorFlow files and matching replay sidecar together. Load only trusted checkpoints: sidecars use Python pickle.

Resume restores model, optimizer, replay and exploration state, but starts a new emulator episode. Legacy checkpoints may not be compatible. Ctrl+C during training saves a checkpoint and closes the video recorder.

## Tests

python -m pip install -e ".[test]"

python -m pytest -q

Tests use small buffers and limited CPU threads. They cover replay boundaries, all five models, model/checkpoint restoration and MP4 decoding. To skip real Atari and CLI tests:

python -m pytest -q -m "not integration"

## References

DQN: Mnih et al. (2015), https://www.nature.com/articles/nature14236

Double DQN: van Hasselt et al. (2015), https://arxiv.org/abs/1509.06461

Dueling DQN: Wang et al. (2015), https://arxiv.org/abs/1511.06581

Environment setup: https://ale.farama.org/release_notes/index.html

TensorFlow installation: https://www.tensorflow.org/install/pip
