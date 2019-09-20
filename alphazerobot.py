import numpy as np
import pyspiel

from mcts import MCTS


def remove_illegal_actions(action_probabilities, legal_actions):
    legal_actions_arr = np.zeros(action_probabilities.shape, dtype=bool)
    legal_actions_arr[legal_actions] = True
    action_probabilities[~legal_actions_arr] = 0.0

    # Check if any of the legal actions actually does have a probability >0.
    if sum(action_probabilities) > 1e-6:
        action_probabilities = action_probabilities / sum(action_probabilities)
    else:
        action_probabilities = np.zeros(len(action_probabilities))
        action_probabilities[legal_actions] = 1. / len(legal_actions)
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
