# pytorch-DQN

## Introduction

A minimalistic implementation of DQN with

- Replay buffer
- Target network
- Epsilon-decay exploration

using

- Numpy
- PyTorch

tested on CartPole.

## Dependency structure

- train_cartpole.py (top)
    - params_pool.py
        - replay_buffer.py
    - replay_buffer.py