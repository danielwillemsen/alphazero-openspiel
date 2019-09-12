import torch
from connect4net import Net
from open_spiel.python import rl_agent
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent
import numpy as np
import pyspiel
from mcts import MCTS

from open_spiel.python.algorithms import mcts


def remove_illegal_actions(action_probabilities, legal_actions):
    legal_actions_arr = np.zeros(action_probabilities.shape, dtype=bool)
    legal_actions_arr[legal_actions] = True
    action_probabilities[~legal_actions_arr] = 0.0

    # Check if any of the legal actions actually does have a probability >0.
    if sum(action_probabilities) > 1e-6:
        action_probabilities = action_probabilities / sum(action_probabilities)
    else:
        action_probabilities = np.zeros(len(action_probabilities))
        action_probabilities[legal_actions] = 1./len(legal_actions)
    return action_probabilities


class AlphaZeroBot(pyspiel.Bot):
    """Bot which uses a combination of MCTS and a policy value net to calculate a move.

    """
    def __init__(self, game, player, policy_fn, self_play=False, keep_search_tree=True, **kwargs):
        super(AlphaZeroBot, self).__init__(game, player)
        self.policy_fn = policy_fn
        self.kwargs = kwargs
        self.mcts = MCTS(self.policy_fn, **kwargs)
        self.self_play = self_play
        self.keep_search_tree = keep_search_tree

    def step(self, state):
        # Defaults to keep the tree during a game
        if self.keep_search_tree:
            action_history = state.history()
            if self.self_play:
                # Update root of the tree with last action
                if action_history:
                    self.mcts.update_root(action_history[-1])
            else:
                # Update root of the tree with last two actions (last opponent and own moves)
                if len(action_history) >= 2:
                    self.mcts.update_root(action_history[-2])
                    self.mcts.update_root(action_history[-1])
        # Create a new MCTS search tree
        else:
            self.mcts = MCTS(self.policy_fn, **self.kwargs)

        # Perform the MCTS
        action_probabilities = np.array(self.mcts.search(state))

        # Remove illegal actions
        legal_actions = state.legal_actions(state.current_player())
        action_probabilities = remove_illegal_actions(action_probabilities, legal_actions)

        # Select the action, either probabilitically or simply the best.
        if self.self_play:
            action = np.random.choice(len(action_probabilities), p=action_probabilities)
        else:
            action = np.argmax(action_probabilities)

        # This format is needed for the bot API
        policy = []
        for act in legal_actions:
            policy.append((act, action_probabilities[act]))

        return policy, action


class NeuralNetBot(pyspiel.Bot):
    """Bot which uses a combination of MCTS and a policy value net to calculate a move.

    """
    def __init__(self, game, player, net):
        super(NeuralNetBot, self).__init__(game, player)
        self.net = net
        self.net.eval()

    def step(self, state):
        ps, v = self.net.predict(state)
        action_probabilities = np.array(ps)

        # Remove illegal actions
        legal_actions = state.legal_actions(state.current_player())
        action_probabilities = remove_illegal_actions(action_probabilities, legal_actions)
        action = np.argmax(action_probabilities)

        # This format is needed for the bot API
        policy = []
        for act in legal_actions:
            policy.append((act, action_probabilities[act]))

        return policy, action

def test():
    game = pyspiel.load_game('connect_four')
    state = game.new_initial_state()
    print("Initial state: ", str(state))
    zero_num = 0
    uct_c = 2
    max_search_nodes = 300
    # Create MCTS bot
    evaluator = mcts.RandomRolloutEvaluator(1)
    mcts_bot = mcts.MCTSBot(game, 1-zero_num, uct_c,
                            max_search_nodes, evaluator)
    # Create random bot
    zero_bot = pyspiel.make_uniform_random_bot(game, zero_num, 123)
    connect_four_net = Net()
    connect_four_net.load_state_dict(torch.load("models/bigwoskipping350.pth", map_location='cpu'))
    #mcts_bot = NeuralNetBot(game, 1-zero_num, connect_four_net)

    zero_bot = AlphaZeroBot(game, zero_num, connect_four_net)

    if zero_num == 1:
        bots = [mcts_bot, zero_bot]
    else:
        bots = [zero_bot, mcts_bot]

    while not state.is_terminal():
        # Decision node: sample action for the single current player
        policy, action = bots[state.current_player()].step(state)
        print("Player ", state.current_player(), ", randomly sampled action: ",
                  state.action_to_string(state.current_player(), action))
        state.apply_action(action)
        print("Next state: ", str(state))

    # Game is now done. Print return for each player
    returns = state.returns()
    for pid in range(game.num_players()):
        print("Return for player {} is {}".format(pid, returns[pid]))
    print("return for alphaZero: " + str(zero_num+1) + "amount:"+ str(returns[zero_num]))


if __name__ == "__main__":
    test()
    # game = pyspiel.load_game('connect_four')
    # state = game.new_initial_state()
    # print("Initial state: ", str(state))
    # mcts_player = 0
    # uct_c = 2
    # max_search_nodes = 100
    # # Create MCTS bot
    # evaluator = mcts.RandomRolloutEvaluator(10)
    # mcts_bot = mcts.MCTSBot(game, mcts_player, uct_c,
    #                         max_search_nodes, evaluator)
    # random_bot = NetBot(game, 0)
    #
    # # Create random bot
    # # random_bot = pyspiel.make_uniform_random_bot(game, 1 - mcts_player, 123)
    # state.apply_action(0)
    # state.apply_action(1)
    # state.apply_action(0)
    # state.apply_action(1)
    # state.apply_action(0)
    # _, action = random_bot.step(state)
    # print("Player ", state.current_player(), ", randomly sampled action: ",
    #       state.action_to_string(state.current_player(), action))
    # state.apply_action(action)
    # print(str(state))
    # _, action = random_bot.step(state)
    # print("Player ", state.current_player(), ", randomly sampled action: ",
    #       state.action_to_string(state.current_player(), action))
    # state.apply_action(action)
    # print(str(state))
    #
    #
