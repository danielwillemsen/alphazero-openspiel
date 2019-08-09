import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class Net(nn.Module):
	def __init__(self, **kwargs):
		super(Net, self).__init__()
		self.width = int(kwargs.get('width', 7))
		self.height = int(kwargs.get('height', 6))
		self.device = kwargs.get('device', torch.device('cpu'))
		self.time1 = time.time()
		self.conv1 = nn.Conv2d(3, 50, (4, 4), padding=1)
		self.conv2 = nn.Conv2d(50, 50, (4, 4), padding=2)
		self.conv3 = nn.Conv2d(50, 50, (4, 4), padding=1)
		self.conv4 = nn.Conv2d(50, 50, (4, 4), padding=2)
		self.conv5 = nn.Conv2d(50, 50, (4, 4), padding=1)
		self.conv6 = nn.Conv2d(50, 50, (4, 4), padding=1)

		self.fc1 = nn.Linear((self.height-2) * (self.width-2) * 50, 100)
		self.fc2 = nn.Linear(100, self.width + 1)

		return

	def forward(self, x):
		"""

		@param x: input tensor
		@return: Policy tensor and value tensor
		"""
		x = F.leaky_relu(self.conv1(x))
		#x = F.leaky_relu(self.conv2(x))
		#x = F.leaky_relu(self.conv3(x))
		#x = F.leaky_relu(self.conv4(x))
		#x = F.leaky_relu(self.conv5(x))
		x = F.leaky_relu(self.conv6(x))

		x = x.view(-1, (self.height-2) * (self.width-2) * 50)
		x = F.leaky_relu(self.fc1(x))
		x = self.fc2(x)
		xp, v = x.split(self.width, 1)
		return F.softmax(xp), F.tanh(v)

	def predict(self, game):
		"""

		@param game: game to predict next move for
		@return: List of policy and the value
		"""
		tens = torch.from_numpy(game.get_board_for_nn()).float().to(self.device)
		tens = tens.unsqueeze(0)

		p_t, v_t = self.forward(tens)
		ps = p_t.tolist()[0]
		v = float(v_t)
		return ps, v
