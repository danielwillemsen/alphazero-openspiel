import numpy as np
import copy
from mcts import MCTS
from alphazerobot import AlphaZeroBot


class ToyGame:
    def __init__(self, length):
        self.length = length
        return

    def num_distinct_actions(self):
        return 4
    def new_initial_state(self):
        return State(0, 0, self.length)

class State:
    def __init__(self, location1, location2, length):
        self.location1 = location1
        self.location2 = location2
        self.length = length
        self.history_list = []
        self.terminal = False
        self.reward = 0

    def is_terminal(self):
        return self.terminal

    def current_player(self):
        if len(self.history_list) % 2 == 0:
            return 0
        else:
            return 1
    def history(self):
        return self.history_list

    def legal_actions(self, player):
        if self.current_player()==0:
            return [0, 1, 2, 3]
        else:
            return [0]

    def information_state(self):
        return ""

    def apply_action(self, action):
        if len(self.history()) > length*2:
            self.terminal = True
            self.reward = -1
            return
        if action in [0, 2]:
            if self.current_player() == 0:
                if action == 0:
                    self.terminal = True
                    self.reward = -1
                if action == 2:
                    self.terminal = True
                    self.reward = 0
        if action == 1:
            if self.current_player() == 0:
                self.location1 += 1
                if self.location1 == self.length:
                    self.terminal = True
                    self.reward = .1
        if action == 3:
            if self.current_player() == 0:
                if self.location1 > 0:
                    self.location1 -= 1
        self.history_list.append(action)


    def player_return(self, player):
        if player == 0:
            return self.reward
        else:
            return -self.reward

    def returns(self):
        return [self.reward, -self.reward]

    def clone(self):
        return copy.deepcopy(self)

def play_game_self(policy_fn, length, **kwargs):
    examples = []
    game = ToyGame(length)
    state = game.new_initial_state()
    num_distinct_actions = game.num_distinct_actions()
    alphazero_bot = AlphaZeroBot(game, 0, policy_fn, self_play=True, **kwargs)
    backup_type = "off-policy"
    while not state.is_terminal():
        policy, action = alphazero_bot.step(state)
        policy_dict = dict(policy)
        policy_list = []
        for i in range(num_distinct_actions):
            # Create a policy list. To be used in the net instead of a list of tuples.
            policy_list.append(policy_dict.get(i, 0.0))
        # MC
        if backup_type == "on-policy":
            examples.append([state.information_state(), state.clone(), policy_list, None])

        # Soft-Z
        if backup_type == "soft-Z":
            examples.append([state.information_state(), state.clone(), policy_list,
                            -alphazero_bot.mcts.root.Q])
        # A0C-2
        if backup_type == "A0C-2":
            node = copy.deepcopy(alphazero_bot.mcts.root)
            value_list = {action_temp: (child.Q if child.N > 0 else -99.0) for action_temp, child in
                          node.children.items()}
            action_temp = max(value_list, key=value_list.get)
            value = - node.children[action_temp].V
            examples.append([state.information_state(), state.clone(), policy_list,
                            value, copy.deepcopy(alphazero_bot.mcts.root)])
        # TD
        if backup_type == "A0C":
            value = max([child.Q if child.N > 0 else -99.0 for child in alphazero_bot.mcts.root.children.values()])
            examples.append([state.information_state(), state.clone(), policy_list, value])
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
            examples.append([state.information_state(), state.clone(), policy_list, value*value_mult, copy.deepcopy(alphazero_bot.mcts.root)])
        state.apply_action(action)
    # Get return for starting player
    if backup_type == "on-policy":
        reward = state.returns()[0]
        for i in range(len(examples)):
            examples[i][3] = reward
            reward *= -1
    return examples

class PVTable:
    def __init__(self, length):
        self.values = np.zeros((length, length, 2))
        self.policy = np.zeros((length, length, 2, 4)) + 0.25
        self.visits = np.zeros((length, length, 2))

    def policy_fn(self, state):
        loc1 = state.location1
        loc2 = state.location2
        player = state.current_player()
        return list(self.policy[loc1, loc2, player, :]), self.values[loc1, loc2, player]

length = 4
num_distinct_actions = 4
pvtable = PVTable(length)

for i_game in range(50000):
    examples = play_game_self(pvtable.policy_fn, length, keep_search_tree=False, n_playouts=25)
    for example in examples:
        player = example[1].current_player()
        loc1 = example[1].location1
        loc2 = example[1].location2

        policy = np.array(example[2])
        value = float(example[3])
        pvtable.values[loc1, loc2, player] = 0.95*pvtable.values[loc1, loc2, player] + 0.05 * value
        pvtable.visits[loc1, loc2, player] += 1
        pvtable.policy[loc1, loc2, player, :] = 0.95*pvtable.policy[loc1, loc2, player] + 0.05 * policy
    if i_game%100 == 0:
        #print(str(pvtable.policy[1,0,0]) + str(pvtable.values[1,0,0]) + str(pvtable.policy[3,0,0]) + str(pvtable.values[6,0,0])
        print(str(pvtable.values[:,0,0]))
        print(str(pvtable.visits[:,0,0]))
# length_cliff = 10
# start = [1,0]
#
# rewards = np.zeros((height, length_cliff))
# rewards[0,:] = -1
# rewards[0,-1] = 1
# rewards[-1,:] = 0.5
# print(rewards)

