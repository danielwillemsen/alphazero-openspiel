import numpy as np
import copy


class Game:
	"""Main game class to play connect 4
	"""

	def __init__(self, **kwargs):
		"""

		@param kwargs:
		"""
		self.width = int(kwargs.get('width', 7))
		self.height = int(kwargs.get('height', 6))
		self.n_in_row = int(kwargs.get('n_in_row', 4))
		self.moves_max = kwargs.get('moves_max', 50)

		self.moves = []
		self.board = np.zeros((self.width,self.height))

	def is_terminal(self):
		"""Checks if game is terminal

		@return: (boolean) true if game is terminal
		"""
		if self.is_winner(1) or self.is_winner(2):
			return True
		if len(self.moves) >= self.moves_max:
			return True
		return False

	def is_winner(self, tile):
		"""Checks if player has won the game

		@param tile: (int) which player to check for if it as one (1/2)
		@return: (bool) true if the player has won
		"""
		# Check horizontal
		for y in range(self.height):
			for x in range(self.width - self.n_in_row + 1):
				winner_here = True
				for pos in range(self.n_in_row):
					if not self.board[y, x + pos] == tile:
						winner_here = False
				if winner_here:
					return True

		# Check vertical
		for y in range(self.height - self.n_in_row + 1):
			for x in range(self.width):
				winner_here = True
				for pos in range(self.n_in_row):
					if not self.board[y + pos, x] == tile:
						winner_here = False
				if winner_here:
					return True

		# Check diagonal \
		for y in range(self.height - self.n_in_row + 1):
			for x in range(self.width - self.n_in_row + 1):
				winner_here = True
				for pos in range(self.n_in_row):
					if not self.board[y + pos, x + pos] == tile:
						winner_here = False
				if winner_here:
					return True

		# Check diagonal /
		for y in range(self.height - self.n_in_row + 1):
			for x in range(self.n_in_row - 1, self.width):
				winner_here = True
				for pos in range(self.n_in_row):
					if not self.board[y + pos, x - pos] == tile:
						winner_here = False
				if winner_here:
					return True
		return False

	def get_copy_board(self):
		"""Creates a deep copy of the board

		@return:
		"""
		return copy.deepcopy(self.board)

	def invert_board(self):
		"""Inverts the board in place.

		@return:
		"""
		self.board = (self.board == 2)*1 + (self.board == 1) * 2
		return

	def get_board_inverted(self):
		"""Get an inverted copy of the board

		@return: Inverted copy of the board
		"""
		return (self.board == 2) * 1 + (self.board == 1) * 2

	def move(self, column, tile):
		"""

		@param column: (int) Column in which to drop the tile
		@param tile: (int) Color of player (either 1 or 2)
		@return:
		"""
		y = 0
		while y < self.height:
			if self.board[y, column] == 0:
				self.board[y, column] = tile
				self.moves.append(column)
				return
			y = y + 1
		self.moves.append(-1)		# indicating that an invalid move has been attempted
		return

	def move_random(self, tile):
		"""

		@param tile: (int) Color of player (either 1 or 2)
		@return:
		"""
		column = np.random.randint(0, self.width)
		self.move(column, tile)
		return
