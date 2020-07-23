import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def state_to_board(state, state_shape):
    """Converts the openspiel state representation of a board to a representation suitable for the neural network.

    :param state: openspiel state representation of the board
    :return: neural network suitable representation of the board. Last layer indicates the current player.
    """
    board = np.zeros((state_shape[0]+1, state_shape[1], state_shape[2])) + state.current_player()
    board[:-1] = np.asarray(state.observation_tensor()).reshape(state_shape)
    board[-1] = np.zeros((1, state_shape[1], state_shape[2])) + state.current_player()
    return board


class Net(nn.Module):
    """Neural network for connect_four.
    The input representation is a ([empty locations, own pieces, opponent pieces] x width x height) matrix.
    The output is a vector of size width, for the policy and an additional scalar for the value.
    """

    def __init__(self, state_shape, num_distinct_actions, **kwargs):
        super(Net, self).__init__()
        self.state_shape = state_shape
        self.num_filters_input = state_shape[0] + 1
        self.height = state_shape[1]
        self.width = state_shape[2]
        self.num_distinct_actions = num_distinct_actions

        self.device = kwargs.get('device', torch.device('cpu'))
        self.time1 = time.time()
        self.n_filts = 50

        self.resblock1 = ResidualBlock(self.num_filters_input, self.n_filts)
        self.resblock2 = ResidualBlock(self.n_filts, self.n_filts)
        self.resblock3 = ResidualBlock(self.n_filts, self.n_filts)
        self.resblock4 = ResidualBlock(self.n_filts, self.n_filts)
        self.resblock5 = ResidualBlock(self.n_filts, self.n_filts)

        self.fc1 = nn.Linear(self.n_filts*self.width*self.height, self.num_distinct_actions + 1)
        return

    def forward(self, x):
        """

        @param x: input tensor
        @return: Policy tensor and value tensor
        """
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)

        x = x.view(-1, self.height * self.width * self.n_filts)
        x = self.fc1(x)
        xp, v = x.split(self.num_distinct_actions, 1)

        return F.softmax(xp, dim=1), torch.tanh(v)

    def predict(self, state):
        """

        @param state: state to predict next move for
        @return: List of policy and the value
        """
        board = state_to_board(state, self.state_shape)

        with torch.no_grad():
            tens = torch.from_numpy(board).float().to(self.device)
            tens = tens.unsqueeze(0)
            p_t, v_t = self.forward(tens)
            ps = p_t.tolist()[0]
            v = float(v_t)
        return ps, v


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_1x1conv = (in_channels != out_channels)

        if self.use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, 1, padding=0)

    def forward(self, x):
        x2 = self.conv1(F.leaky_relu(self.bn1(x)))
        x2 = self.conv2(F.leaky_relu(self.bn2(x2)))
        if self.use_1x1conv:
            x = self.conv3(x)
        return x + x2