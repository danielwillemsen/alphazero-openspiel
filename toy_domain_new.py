import numpy as np
import copy
from mcts import MCTS
from alphazerobot import AlphaZeroBot
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import pickle as p
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 13}
matplotlib.rc('font', **font)

final_visits = []
from game_utils import play_game_self

class PVTable:
    def __init__(self, length):
        self.values = np.zeros((length, 2))

        self.policy = np.zeros((length, 2, 4)) + 0.25
        self.visits = np.zeros((length, 2))

        #Value table uses exponentially weighted moving average. Correction "extra" corrects for the lower number of samples at the start.
        self.extra = np.zeros((length, 2)) + 0.999

    def policy_fn(self, state):
        loc1 = state.location1
        player = state.current_player()
        return list(self.policy[loc1, player, :]), self.values[loc1, player]/(1.0-self.extra[loc1, player])

def update_pvtables(example, pvtables):
    player = example[1].current_player()
    loc1 = example[1].location1

    policy = np.array(example[2])
    for key, pvtable in pvtables.items():
        value = float(example[4][key])
        pvtable.extra[loc1, player] = (1 - alpha) * pvtable.extra[loc1, player]
        pvtable.visits[loc1, player] += 1

        alpha_p = 0.025
        pvtable.values[loc1, player] = (1 - alpha) * pvtable.values[loc1, player] + alpha * value
        pvtable.policy[loc1, player, :] = (1 - alpha_p) * pvtable.policy[loc1, player] + alpha_p * policy

backup_types = ["on-policy", "soft-Z", "A0C", "off-policy", "greedy-forward"]

length = 5
n_games = 5000
num_distinct_actions = 4
pvtables = {backup_type: PVTable(length) for backup_type in backup_types}
pvtable = PVTable(length)
alpha = 0.025
backup_res = {backup_type: [] for backup_type in backup_types}

for i_game in range(n_games):
    examples = play_game_self(pvtables["off-policy"].policy_fn, "toy",
                              keep_search_tree=False,
                              n_playouts=10,
                              c_puct=2.5,
                              dirichlet_ratio=0.25,
                              backup="off-policy",
                              backup_types=backup_types,
                              length=length)
    for example in examples:
        update_pvtables(example, pvtables)

        #For further data visualization
        for key, pvtable in pvtables.items():
            backup_res[key].append(pvtable.values[4, 0])

    if i_game%200 == 0:
        print("Game_no:", i_game, "/", n_games)

for key, val in backup_res.items():
    plt.plot(val, label=key)
plt.legend()
plt.show()
