import numpy as np
import pyspiel
from open_spiel.python.algorithms import mcts
from network import state_to_board
import copy
from alphazerobot import AlphaZeroBot, NeuralNetBot
from network import Net
from alphazerobot import remove_illegal_actions

class HumanBot():
    def step(self, state):
        print(state)
        action = int(input("Give Action"))
        return 0, action

def play_game(game, player1, player2, generate_statistics=False):
    # Returns the reward of the first player
    statistics = dict()
    statistics["player1"] = []
    statistics["player2"] = []

    state = game.new_initial_state()
    while not state.is_terminal():
        if len(state.history()) % 2 == 0:
            action = player1.step(state)
        else:
            action = player2.step(state)
        state.apply_action(action)
        if generate_statistics:
            statistics["player1"].append({"root": copy.deepcopy(player1.mcts.root)})
            statistics["player2"].append({"root": copy.deepcopy(player2.mcts.root)})
    if generate_statistics:
        return state.returns()[0], statistics
    else:
        return state.returns()[0]

def test_zero_vs_human(policy_fn):
    game = pyspiel.load_game('connect_four')

    # Alphazero first
    zero_bot = AlphaZeroBot(game, 0, policy_fn=policy_fn, use_dirichlet=False)
    human_bot = HumanBot()
    score1 = play_game(game, zero_bot, human_bot)

    # Random bot first
    zero_bot = AlphaZeroBot(game, 1, policy_fn=policy_fn, use_dirichlet=False)
    random_bot = pyspiel.make_uniform_random_bot(game, 0, np.random.randint(0, 1000))
    score2 = -play_game(game, human_bot, zero_bot)
    return score1, score2, None

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
    mcts_bot = mcts.MCTSBot(game, 1,
                            max_search_nodes, mcts.RandomRolloutEvaluator(1))
    score1 = play_game(game, zero_bot, mcts_bot)

    # Random bot first
    zero_bot = AlphaZeroBot(game, 1, policy_fn=policy_fn, use_dirichlet=False, **kwargs)
    mcts_bot = mcts.MCTSBot(game, 0,
                            max_search_nodes, mcts.RandomRolloutEvaluator(1))
    score2 = -play_game(game, mcts_bot, zero_bot)
    return score1, score2, None


def test_net_vs_mcts(policy_fn, max_search_nodes, game_name, **kwargs):
    game = pyspiel.load_game(game_name)

    # Alphazero first
    zero_bot = NeuralNetBot(game, 0, policy_fn)
    mcts_bot = mcts.MCTSBot(game, 1,
                            max_search_nodes, mcts.RandomRolloutEvaluator(1))
    score1 = play_game(game, zero_bot, mcts_bot)

    # Random bot first
    zero_bot = NeuralNetBot(game, 1, policy_fn)
    mcts_bot = mcts.MCTSBot(game, 0,
                            max_search_nodes, mcts.RandomRolloutEvaluator(1))
    score2 = -play_game(game, mcts_bot, zero_bot)
    return score1, score2, None


def test_net_vs_random(policy_fn, game_name, **kwargs):
    game = pyspiel.load_game(game_name)

    # Alphazero first
    zero_bot = NeuralNetBot(game, 0, policy_fn)
    random_bot = pyspiel.make_uniform_random_bot(1, np.random.randint(0, 1000))
    score1 = play_game(game, zero_bot, random_bot)

    # Random bot first
    zero_bot = NeuralNetBot(game, 1, policy_fn)
    random_bot = pyspiel.make_uniform_random_bot(0, np.random.randint(0, 1000))
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
    action_was_greedy_list = []
    action_was_greedy_list_N = []

    if game_name == "toy":
        from toy import ToyGame
        from toy import state_to_board
        l = int(kwargs.get("length", 7))
        game = ToyGame(l)
    else:
        from network import state_to_board
        game = pyspiel.load_game(game_name)
    state = game.new_initial_state()
    initial_sequence = kwargs.get("initial_sequence", None)
    if initial_sequence:
        for act in initial_sequence:
            state.apply_action(act)
    state_shape = game.observation_tensor_shape()
    num_distinct_actions = game.num_distinct_actions()
    alphazero_bot = AlphaZeroBot(game, 0, policy_fn, self_play=True, **kwargs)
    backup_types = kwargs.get("backup_types", None)
    main_backup = kwargs.get("backup", "on-policy")

    if not backup_types:
        backup_types = {main_backup}
    #print(main_backup, backup_types)
    assert main_backup in backup_types, "Main backup should be in backup types."

    while not state.is_terminal():

        # Select action, get policy training target
        policy, action = alphazero_bot.step(state, return_policy=True)


        # Create a policy list. To be used in the net instead of a list of tuples.
        policy_dict = dict(policy)
        policy_list = []
        for i in range(num_distinct_actions):
            policy_list.append(policy_dict.get(i, 0.0))

        # Create target dict
        targets = {}
        target = None

        # Add samples to store in replay buffer. Note this is the only difference between the algorithms in the paper.
        # On-policy
        # Cannot determine target yet

        # Soft-Z
        if "soft-Z" in backup_types:
            target = -alphazero_bot.mcts.root.Q
            targets["soft-Z"] = target

        # A0C
        if "A0C" in backup_types:
            target = max([child.Q if child.N > 0 else -99.0 for child in alphazero_bot.mcts.root.children.values()])
            targets["A0C"] = target

        # A0GB:
        if "off-policy" or "greedy-forward" or "greedy-forward-N" in backup_types:
            node = copy.deepcopy(alphazero_bot.mcts.root)
            value_mult = 1.0
            while not node.is_leaf():
                value = node.Q
                value_list = {action_temp: (child.N+child.P if child.N>0 else -99.0) for action_temp, child in node.children.items()}
                action_temp = max(value_list, key=value_list.get)
                node = node.children[action_temp]
                value_mult *= -1.0
            if node.N > 0:
                value = node.Q
                value_mult *=-1.0
            target = value*value_mult
            if "off-policy" in backup_types:
                targets["off-policy"] = target
            if "greedy-forward" in backup_types:
                targets["greedy-forward"] = target
                root_Q = {key: (child.Q if child.N > 0 else -99.0) for (key, child) in alphazero_bot.mcts.root.children.items()}
                greedy_action_value = max(root_Q)
                action_was_greedy_list.append(root_Q[action] >= greedy_action_value - 0.0001)
            if "greedy-forward-N" in backup_types:
                targets["greedy-forward-N"] = target
                root_N = {key: (child.N if child.N > 0 else -99.0) for (key, child) in alphazero_bot.mcts.root.children.items()}
                greedy_action_value = max(root_N)
                action_was_greedy_list_N.append(root_N[action] == greedy_action_value)

        examples.append([state.information_state_string(), state_to_board(state, state_shape), policy_list, None, targets])

        # Take the actual action in the environment
        state.apply_action(action)

    if "greedy-forward" in backup_types:
        for i in reversed(range(len(examples)-1)):
            if action_was_greedy_list[i]:
                examples[i][4]["greedy-forward"] = -examples[i+1][4]["greedy-forward"]

    if "greedy-forward-N" in backup_types:
        for i in reversed(range(len(examples)-1)):
            if action_was_greedy_list_N[i]:
                examples[i][4]["greedy-forward-N"] = -examples[i+1][4]["greedy-forward-N"]

    # For on-policy, the return needs to be set after finishing the game.
    if "on-policy" in backup_types:
        reward = state.returns()[0]
        if initial_sequence:
            reward *= (-1)**len(initial_sequence)
        for i in range(len(examples)):
            examples[i][4]["on-policy"] = reward
            reward *= -1

    for ex in examples:
        ex[3] = ex[4][main_backup]

    return examples
