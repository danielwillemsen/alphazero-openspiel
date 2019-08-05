import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
	def __init__(self, **kwargs):
		super(Net, self).__init__()
		self.width = int(kwargs.get('width', 7))
		self.height = int(kwargs.get('height', 6))
		self.conv1 = nn.Conv2d(3, 160, (4, 4))
		self.fc1 = nn.Linear((self.height-3) * (self.width-3) * 160, 80)
		self.fc2 = nn.Linear(80, self.width + 1)
		return

	def forward(self, x):
		"""

		@param x: input tensor
		@return: Policy tensor and value tensor
		"""
		x = x.unsqueeze(0)
		x = F.relu(self.conv1(x))
		x = x.view(-1,(self.height-3) * (self.width-3) * 160)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		xp, v = x.split(self.width, 1)
		return F.softmax(xp), F.tanh(v)

	def predict(self, game):
		"""

		@param game: game to predict next move for
		@return: List of policy and the value
		"""
		tens = torch.from_numpy(game.get_board_for_nn()).type(torch.float)
		p_t, v_t = self.forward(tens)
		ps = p_t.tolist()[0]
		v = float(v_t)
		return ps, v
