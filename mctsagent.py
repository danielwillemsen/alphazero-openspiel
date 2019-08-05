from connect4 import Game
from mcts import MCTS
import numpy as np
import warnings


class MCTSAgent:
	def __init__(self, net, **kwargs):
		self.net = net

	@staticmethod
	def select_move_noisy(policy):
		policy_noisy = 0.75 * np.array(policy) + 0.25 * np.random.dirichlet(0.3 * np.ones(len(policy)))
		return np.random.choice(len(policy), p=policy_noisy)

	@staticmethod
	def select_move_optimal(policy):
		return int(np.argmax(policy))

	def play_game_self(self):
		"""Play a game against itself using Monte-Carlo tree search

		@return:
		"""
		examples = []
		game = Game()
		mcts = MCTS(self.net)
		while not game.is_terminal():
			game.invert_board()
			policy = mcts.search(game)
			examples.append([game.get_copy_board(), policy, None])
			action = self.select_move_noisy(policy)
			mcts.update_root(action)
			print(action)
			game.move(action, 1)

		if game.is_winner(1):
			value = 1
		elif game.is_winner(2):
			value = -1
			warnings.warn("Warning: player two has won, but player one made a move afterwards")
		else:
			value = 0

		for example in reversed(examples):
			example[2] = value
			value *= -1
		return examples
