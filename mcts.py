"""Implementation of a monte-carly tree search for alphaZero

This code is inspired by the MCTS implementation of Junxiao Song
(https://github.com/junxiaosong/AlphaZero_Gomoku/blob/master/mcts_alphaZero.py)
"""

import numpy as np
import math

class Node:
    """MTCS Node
    """

    def __init__(self, parent, prior_p, use_puct=True):
        self.parent = parent
        self.children = dict()
        self.P = prior_p
        self.Q = 0
        self.N = 0
        self.use_puct = use_puct

    def is_leaf(self):
        return self.children == {}

    def is_root(self):
        return self.parent is None

    def select(self, c_puct, legal_actions):
        """Select the child with highest values (non-noisy at this moment)

        @param c_puct: (float) coefficient for exploration.
        @param legal_actions: (list) of all legal actions
        @return:
        """
        value_list = {action: child.get_value(c_puct) for action, child in self.children.items()}
        action = max(value_list, key=value_list.get)
        child = self.children[action]
        return child, action

    def expand(self, prior_ps, legal_actions):
        """Expand this node

        @param prior_ps: list of prior probabilities (currently only from neural net. In future also from simulation)
        @return:
        """
        for action in legal_actions:
            self.children[action] = Node(self, prior_ps[action], use_puct=self.use_puct)

    def get_value(self, c_puct):
        """Calculates the value of the node

        @param c_puct: (float) coefficient for exploration.
        @return: Q plus bonus value (for exploration)
        """
        if self.use_puct:
            return self.Q + c_puct * self.P * np.sqrt(self.parent.N) / (self.N+1)
        else:
            return float('inf') if self.N == 0 else self.Q + c_puct * self.P * np.sqrt(math.log(self.parent.N) / (self.N))

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

    def __init__(self, policy_fn, num_distinct_actions, **kwargs):
        self.num_distinct_actions = num_distinct_actions
        self.c_puct = float(kwargs.get('c_puct', 1.0))
        self.n_playouts = int(kwargs.get('n_playouts', 100))
        self.use_dirichlet = bool(kwargs.get('use_dirichlet', True))
        self.use_puct = bool(kwargs.get('use_puct', True))
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
            current_player = state.current_player()
            node, action = node.select(self.c_puct, state.legal_actions(current_player))
            state.apply_action(action)

        # Expansion
        if not state.is_terminal():
            prior_ps, leaf_value = self.policy_fn(state)
            node.expand(prior_ps, state.legal_actions(state.current_player()))
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
        visits = [self.root.children[i].N if i in self.root.children else 0 for i in range(self.num_distinct_actions)]
        return [float(visit) / sum(visits) for visit in visits]

    def search(self, state):
        # Expand the root with dirichlet noise if this is the first move of the game
        if self.use_dirichlet and not state.history():
            self.expand_root_dirichlet(state)

        for i in range(self.n_playouts):
            state_copy = state.clone()
            self.playout(state_copy)
        return self.get_action_probabilities()

    def expand_root_dirichlet(self, state):
        prior_ps, leaf_value = self.policy_fn(state)
        legal_actions = state.legal_actions(state.current_player())
        if self.use_dirichlet:
            prior_ps = (0.75 * np.array(prior_ps))
            dirichlet = list(np.random.dirichlet(0.3 * np.ones(len(legal_actions))))
            prior_ps = [prior_ps[i] + 0.25*dirichlet[legal_actions.index(i)] if i in legal_actions else 0.0 for i in range(len(prior_ps))]
        self.root.expand(prior_ps, legal_actions)

    def update_root(self, action):
        """Updates root when new move has been performed.

        @param action: (int) action taht
        @return:
        """
        if self.root.is_leaf():
            self.root = Node(None, 0.0, use_puct=self.use_puct)
        else:
            self.root = self.root.children[action]
            self.root.parent = None

    def random_rollout(self, state):
        working_state = state.clone()
        starting_player = working_state.current_player()
        while not working_state.is_terminal():
            action = np.random.choice(working_state.legal_actions())
            working_state.apply_action(action)
        leaf_value = working_state.player_return(starting_player)
        prior_ps = np.ones(self.num_distinct_actions)
        return prior_ps, leaf_value
