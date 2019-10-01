import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def state_to_board(state, state_shape):
    """Converts the openspiel state representation of a board to a representation suitable for the neural network.

    :param state: openspiel state representation of the board
    :return: neural network suitable representation of the board
    """
    board = np.asarray(state.information_state_as_normalized_vector()).reshape(state_shape)
    if state.current_player() == 0:
        board[[2, 1]] = board[[1, 2]]
    return board


class Net(nn.Module):
    """Neural network for connect_four.
    The input representation is a ([empty locations, own pieces, opponent pieces] x width x height) matrix.
    The output is a vector of size width, for the policy and an additional scalar for the value.
    """

    def __init__(self, state_shape, num_distinct_actions, **kwargs):
        super(Net, self).__init__()
        self.state_shape = state_shape
        self.num_states = state_shape[0]
        self.height = state_shape[1]
        self.width = state_shape[2]
        self.num_distinct_actions = num_distinct_actions

        self.device = kwargs.get('device', torch.device('cpu'))
        self.time1 = time.time()

        self.resblock1 = ResidualBlock(self.num_states, 50)
        self.resblock2 = ResidualBlock(50, 50)
        self.resblock3 = ResidualBlock(50, 50)
        self.resblock4 = ResidualBlock(50, 50)
        self.resblock5 = ResidualBlock(50, 50)
        self.fc1 = nn.Linear(50*self.width*self.height, self.num_distinct_actions + 1)
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

        x = x.view(-1, self.height * self.width * 50)
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


class NetOld(nn.Module):
    """Neural network for connect_four.
    The input representation is a ([empty locations, own pieces, opponent pieces] x width x height) matrix.
    The output is a vector of size width, for the policy and an additional scalar for the value.
    """

    def __init__(self, state_shape, num_distinct_actions, **kwargs):
        super(NetOld, self).__init__()
        self.state_shape = state_shape
        self.num_states = state_shape[0]
        self.height = state_shape[1]
        self.width = state_shape[2]
        self.num_distinct_actions = num_distinct_actions

        self.device = kwargs.get('device', torch.device('cpu'))
        self.time1 = time.time()

        self.conv1 = nn.Conv2d(self.num_states, 50, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(50)
        self.conv2 = nn.Conv2d(50, 50, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(50)
        self.conv3 = nn.Conv2d(50, 50, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(50)
        self.conv4 = nn.Conv2d(50, 50, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(50)
        # self.conv5 = nn.Conv2d(50, 50, (4, 4), padding=1)
        # self.bn5 = nn.BatchNorm2d(50)
        # self.conv6 = nn.Conv2d(50, 50, (4, 4), padding=1)
        # self.bn6 = nn.BatchNorm2d(50)
        self.fc1 = nn.Linear(self.height * self.width * 50, 50)
        self.fc1_bn = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, self.num_distinct_actions + 1)
        return

    def forward(self, x):
        """

        @param x: input tensor
        @return: Policy tensor and value tensor
        """
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        # x = F.leaky_relu(self.bn5(self.conv5(x)))
        # x = F.leaky_relu(self.bn6(self.conv6(x)))

        x = x.view(-1, (self.height) * (self.width) * 50)
        x = F.leaky_relu(self.fc1_bn(self.fc1(x)))
        x = self.fc2(x)
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