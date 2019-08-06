import random
import torch
import torch.nn as nn
import copy

from connect4net import Net
from mctsagent import MCTSAgent


class Trainer:
	def __init__(self):

		self.board_width = 7
		self.board_height = 6
		self.n_in_row = 4
		self.n_games_per_generation = 25
		self.batches_per_generation = 3000
		self.n_games_buffer = 2000
		self.buffer = []
		self.n_tests_full = 10
		self.n_tests_net = 250

		self.criterion_policy = nn.BCELoss()
		self.criterion_value = nn.MSELoss()

		self.current_net = Net(width=self.board_width, height=self.board_height)
		self.current_agent = MCTSAgent(self.current_net,
										board_width=self.board_width,
										board_height=self.board_height,
										n_in_row=self.n_in_row)
		self.optimizer = torch.optim.Adam(self.current_net.parameters(), lr=0.0001, weight_decay=0.0001)

	def test_vs_random(self):
		print("Testing")
		wins = 0
		losses = 0
		for i in range(self.n_tests_full):
			result = self.current_agent.play_game_vs_random()
			if result == 1:
				wins += 1
			elif result == -1:
				losses += 1
		print("Full test: wins: " + str(float(wins)/self.n_tests_full) + " loss: " + str(
			float(losses)/self.n_tests_full))
		wins = 0
		losses = 0
		for i in range(self.n_tests_net):
			result = self.current_agent.play_game_vs_random_net_only()
			if result == 1:
				wins += 1
			elif result == -1:
				losses += 1
		print("Net only test: wins: " + str(float(wins) / self.n_tests_net) + " loss: " + str(
			float(losses) / self.n_tests_net))
		return

	def net_step(self, batch):
		self.current_net.zero_grad()
		x = torch.from_numpy(batch[1]).type(torch.float)
		p_r = torch.tensor(batch[2]).type(torch.float)
		v_r = torch.tensor(batch[3]).type(torch.float)
		p_t, v_t = self.current_net(x)
		loss_v = self.criterion_value(v_t, v_r)
		loss_p = self.criterion_policy(p_t, p_r)
		loss = loss_v + loss_p
		loss.backward()
		self.optimizer.step()
		return loss

	def train_network(self):
		# @todo add support for batches > 1
		print("Training Network")
		flattened_buffer = [sample for game in self.buffer for sample in game]
		loss_tot = 0
		for i in range(self.batches_per_generation):
			batch = random.choice(flattened_buffer)
			loss = self.net_step(batch)
			loss_tot += loss
			if i % 200 == 0:
				print("Batch: " + str(i) + "Loss: " + str(loss_tot/200.))
				loss_tot = 0

	def generate_examples(self, n_games):
		# Generate new training samples
		print("Generating Data")
		for i in range(n_games):
			print("Game " + str(i) + " / " + str(n_games))
			examples = self.current_agent.play_game_self()
			self.buffer.append(examples)
		print("Finished Generating Data")

		# Remove oldest entries from buffer if too long
		if len(self.buffer)>self.n_games_buffer:
			print("Buffer full. Deleting oldest samples.")
			while len(self.buffer) > self.n_games_buffer:
				del self.buffer[0]

	def run(self):
		self.test_vs_random()
		while True:
			self.generate_examples(self.n_games_per_generation)
			self.train_network()
			self.current_agent = MCTSAgent(self.current_net,
											board_width=self.board_width,
											board_height=self.board_height,
											n_in_row=self.n_in_row)
			self.test_vs_random()


if __name__ == '__main__':
	trainer = Trainer()
	trainer.run()
