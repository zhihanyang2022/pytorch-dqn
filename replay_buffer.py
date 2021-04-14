import numpy as np
import torch
from collections import namedtuple

Batch = namedtuple('Batch', 's a r s_prime a_prime mask')

class ReplayBuffer:

    def __init__(self, size):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.next_actions = []
        self.masks = []
        self.size = size

    def push(self, state, action, reward, next_state, next_action, mask):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.next_actions.append(next_action)
        self.masks.append(mask)

        # clean up bellman samples that are too old to use
        if len(self.states) > self.size:
            self.states = self.states[1:]
            self.actions = self.actions[1:]
            self.rewards = self.rewards[1:]
            self.next_states = self.next_states[1:]
            self.next_actions = self.next_actions[1:]
            self.masks = self.masks[1:]

    def sample(self, batch_size) -> Batch:
        idxs = np.random.randint(len(self.states), size=batch_size)
        s = torch.tensor(self.states)[idxs].float()
        a = torch.tensor(self.actions)[idxs].view(-1, 1).long()
        r = torch.tensor(self.rewards)[idxs].view(-1, 1).float()
        s_prime = torch.tensor(self.next_states)[idxs].float()
        a_prime = torch.tensor(self.next_actions)[idxs].view(-1, 1).long()
        mask = torch.tensor(self.masks)[idxs].view(-1, 1).long()
        return Batch(s, a, r, s_prime, a_prime, mask)