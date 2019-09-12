import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np


class Net(nn.Module):
    """Neural network for connect_four.
    The input representation is a ([empty locations, own pieces, opponent pieces] x width x height) matrix.
    The output is a vector of size width, for the policy and an additional scalar for the value.
    """

    def __init__(self, **kwargs):
        super(Net, self).__init__()
        self.width = int(kwargs.get('width', 7))
        self.height = int(kwargs.get('height', 6))
        self.device = kwargs.get('device', torch.device('cpu'))
        self.time1 = time.time()
        self.conv1 = nn.Conv2d(3, 50, 3, padding=1)
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
        self.fc1 = nn.Linear(self.height * self.width * 50, 100)
        self.fc1_bn = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, self.width + 1)
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
        xp, v = x.split(self.width, 1)
        return F.softmax(xp), F.tanh(v)

    def predict(self, state):
        """

        @param state: state to predict next move for
        @return: List of policy and the value
        """
        board = self.state_to_board(state)

        with torch.no_grad():
            tens = torch.from_numpy(board).float().to(self.device)
            tens = tens.unsqueeze(0)
            p_t, v_t = self.forward(tens)
            ps = p_t.tolist()[0]
            v = float(v_t)
        return ps, v

    @staticmethod
    def state_to_board(state):
        """Converts the openspiel state representation of a board to a representation suitable for the neural network.

        :param state: openspiel state representation of the board
        :return: neural network suitable representation of the board
        """
        board = np.asarray(state.information_state_as_normalized_vector()).reshape((3, 6, 7))
        if state.current_player() == 0:
            board[[2, 1]] = board[[1, 2]]
        board = np.flip(board, 1).copy()
        return board
