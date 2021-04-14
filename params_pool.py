import numpy as np
import torch
import torch.nn as nn
from torch import optim

from replay_buffer import Batch

class ParamsPool:

    """
    ParamPool stands for parameter pool. This is inspired by the fact that everything
    in this class, including the behavior and target policies + the prediction and target
    Q networks all depend heavily on lots of parameters.

    Of course, it also involves methods to update the parameters in the face of new data.

    Exposed arguments:
        input_dim (int): dimension of input of the two q networks
        action_dim (int): dimension of output of the two q networks
        epsilon_multiplier (float): epsilon is multiplier by this constant after each episode
        epsilon_min (float): epsilon will be decayed until it reaches this threshold
        gamma (float): discount factor

    Un-exposed arguments (that you might want to play with):
        number of layers and neurons in each layer
        learning rate
        epsilon decay schedule
    """

    def __init__(self, input_dim, num_actions, epsilon_multiplier=0.99, epsilon_min=0.05, gamma=0.95):

        # ===== the Q prediction network =====

        self.q_prediction_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, num_actions),
        )

        # ===== the Q target network =====

        self.q_target_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, num_actions),
        )

        # ref: https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088
        self.q_target_net.eval()

        # ===== optimizer =====

        # ref: https://pytorch.org/docs/stable/optim.html
        self.optim = optim.Adam(self.q_prediction_net.parameters())

        # ===== hyper-parameters =====

        self.num_actions = num_actions  # for action selection
        self.epsilon = 1.0
        self.epsilon_multiplier = epsilon_multiplier
        self.epsilon_min = epsilon_min
        self.gamma = gamma

    def update_q_prediction_net(self, batch: Batch, debug: bool=False) -> None:

        targets = batch.r + self.gamma * self.q_target_net(batch.s_prime).gather(1, batch.a_prime) * batch.mask

        predictions = self.q_prediction_net(batch.s).gather(1, batch.a)

        loss = torch.mean((targets - predictions) ** 2)
        if debug: print(float(loss))

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def update_q_target_net(self) -> None:

        self.q_target_net.load_state_dict(self.q_prediction_net.state_dict())

    def act_using_behavior_policy(self, state: np.array) -> int:

        if np.random.uniform() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return self.act_using_target_policy(state)

    def act_using_target_policy(self, state: np.array) -> int:

        state = torch.tensor(state).unsqueeze(0).float()

        with torch.no_grad():
            q_values = self.q_prediction_net(state).numpy()

        return np.argmax(q_values)

    def decay_epsilon(self) -> None:
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_multiplier