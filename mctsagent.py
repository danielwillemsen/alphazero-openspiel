from connect4 import Game
from mcts import MCTS
import numpy as np
import warnings


class MCTSAgent:
	def __init__(self, net, **kwargs):
		self.net = net
		self.board_width = int(kwargs.get('board_width', 7))
		self.board_height = int(kwargs.get('board_height', 6))
		self.n_in_row = int(kwargs.get('n_in_row', 4))

	@staticmethod
	def select_move_noisy(policy):
		policy_noisy = 0.95 * np.array(policy) + 0.05 * np.random.dirichlet(0.3 * np.ones(len(policy)))
		return np.random.choice(len(policy), p=policy_noisy)

	@staticmethod
	def select_move(policy):
		policy = np.array(policy)
		return np.random.choice(len(policy), p=policy)

	@staticmethod
	def select_move_optimal(policy):
		return int(np.argmax(policy))

	def play_game_self(self):
		"""Play a game against itself using Monte-Carlo tree search

		@return:
		"""
		examples = []
		game = Game(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
		mcts = MCTS(self.net)
		while not game.is_terminal():
			game.invert_board()
			policy = mcts.search(game)
			examples.append([game.get_copy_board(), game.get_board_for_nn(), policy, None])
			action = self.select_move(policy)
			mcts.update_root(action)
			game.move(action, 1)

		if game.is_winner(1):
			value = 1
		elif game.is_winner(2):
			value = -1
			warnings.warn("Warning: player two has won, but player one made a move afterwards")
		else:
			value = 0

		for example in reversed(examples):
			example[3] = value
			value *= -1
		return examples

	def play_game_vs_random(self):
		"""Play a game using against a random move opponent, whilst selecting optimal moves based on MCTS + NN

		@return:
		"""
		game = Game(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
		mcts = MCTS(self.net)

		# Randomly select starting player
		if np.random.randint(0, 2) == 1:
			game.move_random(2)

		# Play game
		while not game.is_terminal():
			policy = mcts.search(game)
			action = self.select_move_optimal(policy)
			mcts.update_root(action)
			game.move(action, 1)
			if not game.is_terminal():
				action = game.move_random(2)
				mcts.update_root(action)
		if game.is_winner(1):
			return 1
		if game.is_winner(2):
			return -1
		else:
			return 0

	def play_game_vs_random_net_only(self):
		"""Play a game using against a random move opponent, whilst selecting optimal moves purely based on NN

		@return:
		"""
		game = Game(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
		mcts = MCTS(self.net)

		# Randomly select starting player
		if np.random.randint(0, 2) == 1:
			game.move_random(2)

		# Play game
		while not game.is_terminal():
			policy, value = self.net.predict(game)
			action = self.select_move_optimal(policy)
			game.move(action, 1)
			if not game.is_terminal():
				game.move_random(2)

		if game.is_winner(1):
			return 1
		if game.is_winner(2):
			return -1
		else:
			return 0
