import numpy as np
import pyspiel
from open_spiel.python.algorithms import mcts

from alphazerobot import AlphaZeroBot, NeuralNetBot
from connect4net import Net


def play_game(game, player1, player2):
    # Returns the reward of the first player
    state = game.new_initial_state()
    while not state.is_terminal():
        if len(state.history()) % 2 == 0:
            _, action = player1.step(state)
        else:
            _, action = player2.step(state)
        state.apply_action(action)
    return state.returns()[0]


def test_zero_vs_random(policy_fn):
    game = pyspiel.load_game('connect_four')

    # Alphazero first
    zero_bot = AlphaZeroBot(game, 0, policy_fn=policy_fn, use_dirichlet=False)
    random_bot = pyspiel.make_uniform_random_bot(game, 1, np.random.randint(0, 1000))
    score1 = play_game(game, zero_bot, random_bot)

    # Random bot first
    zero_bot = AlphaZeroBot(game, 1, policy_fn=policy_fn, use_dirichlet=False)
    random_bot = pyspiel.make_uniform_random_bot(game, 0, np.random.randint(0, 1000))
    score2 = -play_game(game, random_bot, zero_bot)
    return score1, score2


def test_zero_vs_mcts(policy_fn, max_search_nodes, **kwargs):
    game = pyspiel.load_game('connect_four')

    # Alphazero first
    zero_bot = AlphaZeroBot(game, 0, policy_fn=policy_fn, use_dirichlet=False, **kwargs)
    mcts_bot = mcts.MCTSBot(game, 1, 1,
                            max_search_nodes, mcts.RandomRolloutEvaluator(1))
    score1 = play_game(game, zero_bot, mcts_bot)

    # Random bot first
    zero_bot = AlphaZeroBot(game, 1, policy_fn=policy_fn, use_dirichlet=False, **kwargs)
    mcts_bot = mcts.MCTSBot(game, 0, 1,
                            max_search_nodes, mcts.RandomRolloutEvaluator(1))
    score2 = -play_game(game, mcts_bot, zero_bot)
    return score1, score2


def test_net_vs_mcts(policy_fn, max_search_nodes):
    game = pyspiel.load_game('connect_four')

    # Alphazero first
    zero_bot = NeuralNetBot(game, 0, policy_fn)
    mcts_bot = mcts.MCTSBot(game, 1, 1,
                            max_search_nodes, mcts.RandomRolloutEvaluator(1))
    score1 = play_game(game, zero_bot, mcts_bot)

    # Random bot first
    zero_bot = NeuralNetBot(game, 1, policy_fn)
    mcts_bot = mcts.MCTSBot(game, 0, 1,
                            max_search_nodes, mcts.RandomRolloutEvaluator(1))
    score2 = -play_game(game, mcts_bot, zero_bot)
    return score1, score2


def test_net_vs_random(policy_fn):
    game = pyspiel.load_game('connect_four')

    # Alphazero first
    zero_bot = NeuralNetBot(game, 0, policy_fn)
    random_bot = pyspiel.make_uniform_random_bot(game, 1, np.random.randint(0, 1000))
    score1 = play_game(game, zero_bot, random_bot)

    # Random bot first
    zero_bot = NeuralNetBot(game, 1, policy_fn)
    random_bot = pyspiel.make_uniform_random_bot(game, 0, np.random.randint(0, 1000))
    score2 = -play_game(game, random_bot, zero_bot)
    return score1, score2


def play_game_self(policy_fn):
    examples = []
    game = pyspiel.load_game('connect_four')
    state = game.new_initial_state()
    alphazero_bot = AlphaZeroBot(game, 0, policy_fn, self_play=True)
    while not state.is_terminal():
        policy, action = alphazero_bot.step(state)
        policy_dict = dict(policy)
        policy_list = []
        for i in range(7):
            # Create a policy list. To be used in the net instead of a list of tuples.
            policy_list.append(policy_dict.get(i, 0.0))
        examples.append([state.information_state(), Net.state_to_board(state), policy_list, None])
        state.apply_action(action)
    # Get return for starting player
    reward = state.returns()[0]
    for i in range(len(examples)):
        examples[i][3] = reward
        reward *= -1
    return examples
