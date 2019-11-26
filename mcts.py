"""Implementation of a monte-carly tree search for alphaZero

This code is inspired by the MCTS implementation of Junxiao Song
(https://github.com/junxiaosong/AlphaZero_Gomoku/blob/master/mcts_alphaZero.py)
"""

import numpy as np
import math

class Node:
    """MTCS Node containing prior policy, Q-value and visit count.
    """

    def __init__(self, parent, prior_p, use_puct=True):
        self.parent = parent
        self.children = dict()
        self.P = prior_p
        self.Q = 0
        self.N = 0
        self.use_puct = use_puct

    def is_leaf(self):
        """ Check if this node is a leaf

        Returns: Boolean whether node is leaf or not

        """
        return self.children == {}

    def is_root(self):
        """ Check whether this node is a root node

        Returns: Boolean whether node is root or not

        """
        return self.parent is None

    def select(self, c_puct):
        """Select the child with highest values (non-noisy at this moment)

        Args:
            c_puct: (float) coefficient for exploration.

        Returns:
            child: selected child node
            action: action associated with child node

        """
        value_list = {action: child.get_value(c_puct) for action, child in self.children.items()}
        action = max(value_list, key=value_list.get)
        child = self.children[action]
        return child, action

    def expand(self, prior_ps, legal_actions):
        """Expand this node

        Args:
            prior_ps: list of prior probabilities (currently only from neural net. In future also from simulation)
            legal_actions: list of legal actions

        """
        for action in legal_actions:
            if action not in self.children:
                self.children[action] = Node(self, prior_ps[action], use_puct=self.use_puct)
            else:
                self.children[action].P = prior_ps[action]

    def get_value(self, c_puct):
        """Calculates the value of the node according to PUCT or another modified UCT

        Args:
            c_puct: (float) coefficient for exploration.

        Returns: float of the value

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

    def __init__(self, policy_fn, num_distinct_actions,
                 c_puct=2.5, n_playouts=100, use_dirichlet=True, use_puct=True):
        """Initializes the MCTS search tree

        Args:
            policy_fn: function that is used to get the value and prior policy estimate.
            num_distinct_actions: Amount of distinct actions for this game

        Keyword Args:
            c_puct: exploration coefficient for the search
            n_playouts: number of simulations to do for a single MCTS search
            use_dirichlet: whether to add dirichlet noise to root nodes
            use_puct: use alphaZero style PUCT or use a different UCT formula.

        """
        self.num_distinct_actions = num_distinct_actions
        self.c_puct = c_puct
        self.n_playouts = n_playouts
        self.use_dirichlet = use_dirichlet
        self.use_puct = use_puct
        self.root = Node(None, 0.0)
        self.policy_fn = policy_fn

    def playout(self, state):
        """Do a single MCTS simulation

        Performs the four steps of MCTS once: selection, expansion, simulation and backpropagation.

        Args:
            state: current root state. Should be a copy as it is modified in-place

        """
        node = self.root

        # Selection
        current_player = state.current_player()
        while not node.is_leaf() and not state.is_terminal():
            current_player = state.current_player()
            node, action = node.select(self.c_puct)
            state.apply_action(action)

        # Expansion & "Simulation (policy-value function evaluation"
        if not state.is_terminal():
            prior_ps, leaf_value = self.policy_fn(state)
            node.expand(prior_ps, state.legal_actions(state.current_player()))
        else:
            leaf_value = -state.player_return(current_player)

        # Back propagation
        node.update_recursive(-leaf_value)
        return

    def get_normalized_visit_counts(self):
        """Get the normalized visit counts of the root node of this MCTS tree

        Returns: list of normalized visit counts

        """
        visits = [self.root.children[i].N if i in self.root.children else 0 for i in range(self.num_distinct_actions)]
        return [float(visit) / sum(visits) for visit in visits]

    def search(self, state):
        """Do a full MCTS search for the given state

        Args:
            state: current state for which the MCTS search needs to be performed

        Returns: list of normalized visit counts. Can be interpreted as action probabilities.

        """
        # Expand the root with dirichlet noise if this is the first move of the game
        if self.use_dirichlet:
            self.expand_root_dirichlet(state)

        for i in range(self.n_playouts):
            state_copy = state.clone()
            self.playout(state_copy)
        return self.get_normalized_visit_counts()

    def expand_root_dirichlet(self, state):
        prior_ps, leaf_value = self.policy_fn(state)
        legal_actions = state.legal_actions(state.current_player())
        if self.use_dirichlet:
            prior_ps = (0.75 * np.array(prior_ps))
            dirichlet = list(np.random.dirichlet(0.3 * np.ones(len(legal_actions))))
            for i, action in enumerate(legal_actions):
                prior_ps[action] = prior_ps[action] + 0.25*dirichlet[i]
        self.root.expand(prior_ps, legal_actions)

    def update_root(self, action):
        """Updates root when new move has been performed.

        Args:
            action: (int) action that should be performed to update the root
        @return:
        """
        if self.root.is_leaf():
            self.root = Node(None, 0.0, use_puct=self.use_puct)
        else:
            self.root = self.root.children[action]
            self.root.parent = None

    def random_rollout(self, state):
        """Perform a random rollout as an alternative to a policy-value function

        Args:
            state: state from which to run the rollout

        Returns:
            prios_ps: a uniform prior policy
            leaf_value: value from the random rollout

        """
        working_state = state.clone()
        starting_player = working_state.current_player()
        while not working_state.is_terminal():
            action = np.random.choice(working_state.legal_actions())
            working_state.apply_action(action)
        leaf_value = working_state.player_return(starting_player)
        prior_ps = np.ones(self.num_distinct_actions)
        return prior_ps, leaf_value
