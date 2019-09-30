import copy

import numpy as np


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
        self.moves_max = kwargs.get('moves_max', self.height * self.width)

        self.moves = []
        self.board = np.zeros((self.height, self.width))

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
        for x in range(self.width):
            for y in range(self.height):
                # Quick precheck
                if self.board[y, x] != tile:
                    continue

                # Check horizontal:
                if x <= self.width - self.n_in_row:
                    winner_here = True
                    for pos in range(self.n_in_row):
                        if not self.board[y, x + pos] == tile:
                            winner_here = False
                            break
                    if winner_here:
                        return True

                # Check vertical
                if y <= self.height - self.n_in_row:
                    winner_here = True
                    for pos in range(self.n_in_row):
                        if not self.board[y + pos, x] == tile:
                            winner_here = False
                            break
                    if winner_here:
                        return True

                # Check diagonal \
                if y <= self.height - self.n_in_row:
                    if x <= self.width - self.n_in_row:
                        winner_here = True
                        for pos in range(self.n_in_row):
                            if not self.board[y + pos, x + pos] == tile:
                                winner_here = False
                                break
                        if winner_here:
                            return True

                # Check diagonal /
                if y <= self.height - self.n_in_row:
                    if x >= self.n_in_row - 1:
                        winner_here = True
                        for pos in range(self.n_in_row):
                            if not self.board[y + pos, x - pos] == tile:
                                winner_here = False
                                break
                        if winner_here:
                            return True
        return False

    def get_copy_board(self):
        """Creates a deep copy of the board

        @return:
        """
        return copy.deepcopy(self.board)

    def get_board_for_nn(self):
        """Creates a board state which is more suitable for a neural network input

        @return: (np.array) board state consisting of 3 layers: free, occupied by player 1, occupied by player 2.
        """
        board_for_nn = np.zeros((3, self.height, self.width))
        board_for_nn[0, self.board == 0] = 1
        board_for_nn[1, self.board == 1] = 1
        board_for_nn[2, self.board == 2] = 1
        return board_for_nn

    def invert_board(self):
        """Inverts the board in place.

        @return:
        """
        self.board = (self.board == 2) * 1 + (self.board == 1) * 2
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
        ind = 0
        in_column = column
        while ind < self.width:
            y = 0
            while y < self.height:
                if self.board[y, column] == 0:
                    self.board[y, column] = tile
                    self.moves.append(column)
                    return
                y = y + 1
            ind = ind + 1
            column = (in_column + ind) % self.width
        # indicating that an invalid move has been attempted
        logger.info("Warning: Unable to do move, board is full.")
        logger.info(self.board)
        self.moves.append(-1)
        return

    def move_random(self, tile):
        """

        @param tile: (int) Color of player (either 1 or 2)
        @return:
        """
        column = np.random.randint(0, self.width)
        self.move(column, tile)
        return column


if __name__ == '__main__':
    import time

    game = Game()
    start = time.time()
    for i in range(12000):
        game.is_winner(1)
    logger.info(time.time() - start)
