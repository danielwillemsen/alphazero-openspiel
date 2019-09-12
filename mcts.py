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
	def __init__(self, policy_fn, **kwargs):
		self.c_puct = float(kwargs.get('c_puct', 1.0))
		self.n_playouts = int(kwargs.get('n_playouts', 100))
		self.use_dirichlet = bool(kwargs.get('use_dirichlet', True))
		self.root = Node(None, 0.0)
		self.policy_fn = policy_fn

	def playout(self, state):
		"""

		@param state: Should be a copy of the state as it is modified in place.
		@return:
		"""
		node = self.root

		# Selection
		current_player = state.current_player()
		while not node.is_leaf() and not state.is_terminal():
			node, action = node.select(self.c_puct)
			# @todo make this nicer
			while action not in state.legal_actions():
				action += 1
				if action == 7:
					action = 0
			current_player = state.current_player()
			state.apply_action(action)

		# Expansion
		if not state.is_terminal():
			# @todo add possibility of using simulation instead of neural net prediction (for pure MCTS)
			prior_ps, leaf_value = self.policy_fn(state)

			# Add dirichlet noise @todo check if this is the correct location for dirichlet noise
			if self.use_dirichlet:
				prior_ps = (0.8 * np.array(prior_ps) + 0.2 * np.random.dirichlet(0.3 * np.ones(len(prior_ps)))).tolist()

			node.expand(prior_ps)
		else:
			leaf_value = -state.player_return(current_player)

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

	def search(self, state):
		for i in range(self.n_playouts):
			state_copy = state.clone()
			self.playout(state_copy)
		return self.get_action_probabilities()

	def update_root(self, action):
		"""Updates root when new move has been performed.

		@param action: (int) action taht
		@return:
		"""
		if self.root.is_leaf():
			self.root = Node(None, 0.0)
		else:
			self.root = self.root.children[action]

