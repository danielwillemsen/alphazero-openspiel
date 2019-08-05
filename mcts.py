"""Implementation of a monte-carly tree search for alphaZero

This code is inspired by the MCTS implementation of Junxiao Song
(https://github.com/junxiaosong/AlphaZero_Gomoku/blob/master/mcts_alphaZero.py)
"""

import numpy as np
import copy


class Node:
	"""MTCS Node
	"""
	def __init__(self, parent, prior_p):
		self.parent = parent
		self.children = []
		self.P = prior_p
		self.Q = 0
		self.N = 0

	def is_leaf(self):
		return self.children == []

	def is_root(self):
		return self.parent is None

	def select(self, c_puct):
		"""Select the child with highest values (non-noisy at this moment)

		@param c_puct: (float) coefficient for exploration.
		@return:
		"""
		value_list = [child.get_value(c_puct) for child in self.children]
		action = int(np.argmax(value_list))
		child = self.children[action]
		return child, np.argmax(value_list)

	def expand(self, prior_ps):
		"""Expand this node

		@param prior_ps: list of prior probabilities (currently only from neural net. In future also from simulation)
		@return:
		"""
		for action in range(len(prior_ps)):
			self.children.append(Node(self, prior_ps[action]))

	def get_value(self, c_puct):
		"""Calculates the value of the node

		@param c_puct: (float) coefficient for exploration.
		@return: Q plus bonus value (for exploration)
		"""
		return self.Q + c_puct*self.P*np.sqrt(self.parent.N)/(1+self.N)

	def update(self, value):
		self.Q = (self.N * self.Q + value) / (self.N + 1)
		self.N += 1

	def update_recursive(self, value):
		if not self.is_root():
			self.parent.update_recursive(-value)
		self.update(value)


class MCTS:
	"""Main Monte-Carlo tree class. Should be kept during the whole game.
	"""
	def __init__(self, net, **kwargs):
		self.c_puct = float(kwargs.get('c_puct', 1.0))
		self.n_playouts = int(kwargs.get('n_playouts', 500))

		self.root = Node(None, 0.0)
		self.net = net

	def playout(self, game):
		"""

		@param game: Should be a copy of the game as it is modified in place.
		@return:
		"""
		node = self.root

		# Selection
		while not node.is_leaf():
			node, action = node.select(self.c_puct)
			game.move(action, 1)
			game.invert_board()

		# Expansion
		if not game.is_terminal():
			# @todo add possiblity of using simulation instead of neural net prediction (for pure MCTS)
			prior_ps, leaf_value = self.net.predict(game)
			node.expand(prior_ps)
		else:
			if game.is_winner(1):
				leaf_value = 1
			elif game.is_winner(2):
				leaf_value = -1
			else:
				leaf_value = 0

		# Back propagation
		# @todo check if this minus sign here makes sense
		node.update_recursive(-leaf_value)
		return

	def get_action_probabilities(self):
		"""For now simply linear with the amount of visits.
		@todo check how this is done in the alphaZero paper

		@return:
		"""
		visits = [child.N for child in self.root.children]
		return [float(visit)/sum(visits) for visit in visits]

	def search(self, game):
		for i in range(self.n_playouts):
			game_copy = copy.deepcopy(game)
			self.playout(game_copy)
		return self.get_action_probabilities()

	def update_root(self, action):
		"""Updates root when new move has been performed.

		@param action: (int) action taht
		@return:
		"""
		self.root = self.root.children[action]
