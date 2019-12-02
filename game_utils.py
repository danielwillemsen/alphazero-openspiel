import numpy as np
import pyspiel
from open_spiel.python.algorithms import mcts
from connect4net import state_to_board
import copy
from alphazerobot import AlphaZeroBot, NeuralNetBot
from connect4net import Net


def play_game(game, player1, player2, generate_statistics=False):
    # Returns the reward of the first player
    statistics = dict()
    statistics["player1"] = []
    statistics["player2"] = []

    state = game.new_initial_state()
    while not state.is_terminal():
        if len(state.history()) % 2 == 0:
            _, action = player1.step(state)
        else:
            _, action = player2.step(state)
        state.apply_action(action)
        if generate_statistics:
            statistics["player1"].append({"root": copy.deepcopy(player1.mcts.root)})
            statistics["player2"].append({"root": copy.deepcopy(player2.mcts.root)})
    if generate_statistics:
        return state.returns()[0], statistics
    else:
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
    return score1, score2, None


def test_zero_vs_mcts(policy_fn, max_search_nodes, game_name, **kwargs):
    game = pyspiel.load_game(game_name)

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
    return score1, score2, None


def test_net_vs_mcts(policy_fn, max_search_nodes, game_name, **kwargs):
    game = pyspiel.load_game(game_name)

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
    return score1, score2, None


def test_net_vs_random(policy_fn, game_name, **kwargs):
    game = pyspiel.load_game(game_name)

    # Alphazero first
    zero_bot = NeuralNetBot(game, 0, policy_fn)
    random_bot = pyspiel.make_uniform_random_bot(game, 1, np.random.randint(0, 1000))
    score1 = play_game(game, zero_bot, random_bot)

    # Random bot first
    zero_bot = NeuralNetBot(game, 1, policy_fn)
    random_bot = pyspiel.make_uniform_random_bot(game, 0, np.random.randint(0, 1000))
    score2 = -play_game(game, random_bot, zero_bot)
    return score1, score2


def test_zero_vs_zero(policy_fn, max_search_nodes, game_name, policy_fn2=None, generate_statistics=False, **kwargs):
    settings1 = dict(kwargs.get("settings1", None))
    settings2 = dict(kwargs.get("settings2", None))
    statistics = {}

    if not policy_fn2:
        policy_fn2 = policy_fn
    game = pyspiel.load_game(game_name)

    # Alphazero first
    zero1_bot = AlphaZeroBot(game, 0, policy_fn=policy_fn, use_dirichlet=True, **settings1)
    zero2_bot = AlphaZeroBot(game, 1, policy_fn=policy_fn2, use_dirichlet=True, **settings2)

    if generate_statistics:
        score1, statistics_game1 = play_game(game, zero1_bot, zero2_bot, generate_statistics=generate_statistics)
        statistics["game1"] = statistics_game1
    else:
        score1 = play_game(game, zero1_bot, zero2_bot, generate_statistics=generate_statistics)

    # Random bot first
    zero1_bot = AlphaZeroBot(game, 1, policy_fn=policy_fn, use_dirichlet=True, **settings1)
    zero2_bot = AlphaZeroBot(game, 0, policy_fn=policy_fn2, use_dirichlet=True, **settings2)
    if generate_statistics:
        score2, statistics_game2 = play_game(game, zero2_bot, zero1_bot, generate_statistics=generate_statistics)
        statistics_game2["player1"], statistics_game2["player2"] = statistics_game2["player2"], statistics_game2["player1"]
        statistics["game2"] = statistics_game2
        score2 *= -1
    else:
        score2 = -play_game(game, zero2_bot, zero1_bot, generate_statistics=generate_statistics)

    return score1, score2, statistics


def play_game_self(policy_fn, game_name, **kwargs):
    examples = []
    game = pyspiel.load_game(game_name)
    state = game.new_initial_state()
    state_shape = game.information_state_normalized_vector_shape()
    num_distinct_actions = game.num_distinct_actions()
    alphazero_bot = AlphaZeroBot(game, 0, policy_fn, self_play=True, **kwargs)
    backup_type = str(kwargs.get("backup", "MC"))
    tree_strap = str(kwargs.get("tree_strap", False))
    while not state.is_terminal():
        policy, action = alphazero_bot.step(state)
        policy_dict = dict(policy)
        policy_list = []
        for i in range(num_distinct_actions):
            # Create a policy list. To be used in the net instead of a list of tuples.
            policy_list.append(policy_dict.get(i, 0.0))
        # MC
        if backup_type == "on-policy":
            if tree_strap:

                examples.append([state.information_state(), state_to_board(state, state_shape), policy_list,
                                 None , []])
                for action_temp, child_temp in alphazero_bot.mcts.root.children.items():
                    if child_temp.N > 10:  # The five is a tunable hyperparameter
                        copy_state = state.clone()
                        copy_state.apply_action(action_temp)
                        if not copy_state.is_terminal():
                            examples[-1][4].append(
                                [copy_state.information_state(), state_to_board(copy_state, state_shape), None,
                                 -child_temp.Q])
                # tree_strap_examples = get_examples_tree_strap_soft_Z(alphazero_bot.mcts.root, state.clone(), state_shape)
                # examples[-1][4] = tree_strap_examples
            else:
                examples.append([state.information_state(), state_to_board(state, state_shape), policy_list, None])

        # Soft-Z
        if backup_type == "soft-Z":
            if tree_strap:
                examples.append([state.information_state(), state_to_board(state, state_shape), policy_list,
                                 -alphazero_bot.mcts.root.Q, []])
                tree_strap_examples = get_examples_tree_strap_soft_Z(alphazero_bot.mcts.root, state.clone(), state_shape)
                examples[-1][4] = tree_strap_examples
            else:
                examples.append([state.information_state(), state_to_board(state, state_shape), policy_list,
                                 -alphazero_bot.mcts.root.Q])

        # TD
        if backup_type == "A0C":
            value = max([child.Q if child.N > 0 else -99.0 for child in alphazero_bot.mcts.root.children.values()])
            examples.append([state.information_state(), state_to_board(state, state_shape), policy_list, value])
        # diff:
        if backup_type == "off-policy":
            node = copy.deepcopy(alphazero_bot.mcts.root)
            value_mult = 1
            while not node.is_leaf():
                value = node.Q
                value_list = {action_temp: (child.N if child.N>0 else -99.0) for action_temp, child in node.children.items()}
                action_temp = max(value_list, key=value_list.get)
                node = node.children[action_temp]
                value_mult *= -1
            if node.N > 0:
                value = node.Q
                value_mult *=-1
            if tree_strap:
                examples.append([state.information_state(), state_to_board(state, state_shape), policy_list, value*value_mult, []])
                tree_strap_examples = get_examples_tree_strap_off_policy(alphazero_bot.mcts.root, state.clone(), state_shape)
                examples[-1][4] = tree_strap_examples
            else:
                examples.append([state.information_state(), state_to_board(state, state_shape), policy_list, value*value_mult])
        state.apply_action(action)
    # Get return for starting player
    if backup_type == "on-policy":
        reward = state.returns()[0]
        for i in range(len(examples)):
            examples[i][3] = reward
            reward *= -1
    return examples

def get_examples_tree_strap_off_policy(node, state, state_shape):
    examples = []
    if node.N > 20 and not state.is_terminal():
        for action_temp, child in node.children.items():
            if child.N > 20:
                temp_state = state.clone()
                temp_state.apply_action(action_temp)
                single_examples = get_examples_tree_strap_off_policy(child, temp_state, state_shape)
                for example in single_examples:
                    examples.append(example)
        value_mult = 1
        while not node.is_leaf():
            value = node.Q
            value_list = {action_temp: (child.N if child.N > 0 else -99.0) for action_temp, child in node.children.items()}
            action_temp = max(value_list, key=value_list.get)
            node = node.children[action_temp]
            value_mult *= -1
        examples.append([state.information_state(), state_to_board(state, state_shape), None, value*value_mult])
    return examples

def get_examples_tree_strap_soft_Z(node, state, state_shape):
    examples = []
    if node.N > 20 and not state.is_terminal():
        for action_temp, child in node.children.items():
            if child.N > 20:
                temp_state = state.clone()
                temp_state.apply_action(action_temp)
                single_examples = get_examples_tree_strap_soft_Z(child, temp_state, state_shape)
                for example in single_examples:
                    examples.append(example)
        value = node.Q
        examples.append([state.information_state(), state_to_board(state, state_shape), None, value*-1])
    return examples

# def get_examples_tree_strap_soft_Z(node, state, state_shape, num_distinct_actions):
#     examples = []
#     if node.N > 15 and not state.is_terminal():
#         for action_temp, child in node.children.items():
#             if child.N > 15:
#                 temp_state = state.clone()
#                 temp_state.apply_action(action_temp)
#                 single_examples = get_examples_tree_strap_soft_Z(child, temp_state, state_shape, num_distinct_actions)
#                 for example in single_examples:
#                     examples.append(example)
#         value = node.Q
#         visits = [node.children[i].N if i in node.children else 0 for i in range(num_distinct_actions)]
#         normalized_visit_counts = [float(visit) / sum(visits) for visit in visits]
#         legal_actions = state.legal_actions(state.current_player())
#         action_probabilities = remove_illegal_actions(np.array(normalized_visit_counts), legal_actions)
#         policy = []
#         for act in legal_actions:
#             policy.append(action_probabilities[act])
#         examples.append([state.information_state(), state_to_board(state, state_shape), action_probabilities.tolist(), value*-1])
#     return examples