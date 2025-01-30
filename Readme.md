# Introduction to DRL Homework 2

This file will help you setup and run the codes.


## Requirements

We strongly recommend you to use Anaconda or Miniconda to setup Python environments, as they will automatically install required dependencies.

To install requirements:

```setup
conda env create -n hw2
conda activate hw2
pip install -r requirements.txt
conda install tensorflow-gpu # can also be torch, tensorflow or keras, any DL library you like
```



## Running the Codes
With the environment ready, you can start writing and testing your codes. An example command is:
```setup
python dqn_atari.py --env Breakout-v0
```
You can change the env name to run on different environments.

If you have any problems with the code, feel free to contact us at bengisu@wustl.edu and d.kefei@wustl.edu.