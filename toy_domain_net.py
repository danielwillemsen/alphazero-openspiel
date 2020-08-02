import numpy as np
import copy
from mcts import MCTS
from alphazerobot import AlphaZeroBot
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import pyspiel
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

class PVTable_tictactoe:
    def __init__(self, length, use_random=False, tables=[]):
        self.values = dict()
        self.use_random = use_random
        self.tables = tables
        self.policy = dict()
        self.visits = dict() #np.zeros((length, 2))

        #Value table uses exponentially weighted moving average. Correction "extra" corrects for the lower number of samples at the start.
        self.extra = dict() #np.zeros((length, 2)) + 0.999

    def policy_fn(self, state):
        string = state.information_state_string()
        if string in self.values.keys():
            pol = self.policy[string]
            val = self.values[string]
        else:
            # Initialize all tables to the same, random, value
            if self.use_random:
                pol = np.zeros(9) + 1./9.
                val = np.random.randn()*0.25
                for table in self.tables:
                    table.policy[string] = pol
                    table.values[string] = val
                    table.visits[string] = 1
            else:
                pol = np.zeros(9) + 1./9.
                val = 0.


        return list(pol), val#/(1.0-self.extra[loc1, player])

def update_pvtables(example, pvtables):
    player = example[1].current_player()
    loc1 = example[1].location1

    policy = np.array(example[2])
    for key, pvtable in pvtables.items():
        value = float(example[4][key])
        pvtable.extra[loc1, player] = (1 - alpha) * pvtable.extra[loc1, player]
        pvtable.visits[loc1, player] += 1

        alpha_p = 0.1
        pvtable.values[loc1, player] = (1 - alpha) * pvtable.values[loc1, player] + alpha * value
        pvtable.policy[loc1, player, :] = (1 - alpha_p) * pvtable.policy[loc1, player] + alpha_p * policy

def update_pvtables_tictactoe(example, pvtables):
    state = example[0]
    policy = np.array(example[2])
    for key, pvtable in pvtables.items():
        value = float(example[4][key])
        alpha_p = 0.1
        if state in pvtable.values.keys():
            pvtable.values[state] = (1 - alpha) * pvtable.values[state] + alpha * value
            pvtable.policy[state] = (1 - alpha_p) * pvtable.policy[state] + alpha_p * policy
            pvtable.visits[state] += 1

        else:
            pvtable.values[state] = alpha * value
            pvtable.policy[state] = alpha_p * policy + 1./9.
            pvtable.visits[state] = 1

backup_types = ["on-policy", "soft-Z", "A0C", "off-policy"]#, "greedy-forward", "greedy-forward-N"]

game = pyspiel.load_game("connect_four")
length = 5
n_games = 2500
num_distinct_actions = 4
pvtables = {backup_type: PVTable_tictactoe(length, use_random=True) for backup_type in backup_types}
alpha = 0.025
backup_res = {backup_type: [] for backup_type in backup_types}

tables = [tab for tab in pvtables.values()]
use_table = pvtables["off-policy"]
use_table.tables = tables
for i_game in range(n_games):
    examples = play_game_self(pvtables["off-policy"].policy_fn, "tic_tac_toe",
                              keep_search_tree=False,
                              n_playouts=100,
                              c_puct=2.5,
                              dirichlet_ratio=0.25,
                              backup="off-policy",
                              backup_types=backup_types,
                              length=length,
                              initial_sequence=[])# 0 4 2
    for example in examples:
        update_pvtables_tictactoe(example, pvtables)

        #For further data visualization
        for key, pvtable in pvtables.items():
            #backup_res[key].append(np.copy(pvtable.values))
            backup_res[key].append(np.copy(pvtable.values[""]))

    # if backup_res["off-policy"][-1][3,0] > 0.0001:
    #     a=2
    if i_game%2 == 0:
        print("Game_no:", i_game, "/", n_games)

backup_plot_res = {backup_type: [] for backup_type in backup_types}
for key, val in backup_res.items():
    backup_res[key] = np.stack(val, axis=0)

for key, val in backup_res.items():
    plt.plot(val, label=key)
    plt.xlabel("Backup Number")
    plt.ylabel("Value estimate of state (Tieable in 6 turns)")

#    plt.plot(val[:, 3, 0], label=key)
plt.legend()
plt.show()

labels = [type_name.title() for type_name in backup_types]
cmap = cm.get_cmap('RdYlGn', 30)

i = 0
fig, axes = plt.subplots(nrows=len(backup_types), ncols=1, figsize=(8,8), tight_layout=False, constrained_layout=True)
axes[0].set_title(f"Progress over {n_games} games")

fig.set_tight_layout(False)
for ax in axes.flat:
    #ax.set_axis_off()
#     im = ax.imshow(np.expand_dims(values_total[i],0), label=labels[i], cmap=cmap, vmin=-0.1, vmax=0.1)
    im = ax.imshow(backup_res[backup_types[i]][:,:,0], label=backup_types[i], cmap=cmap, vmin=-0.1, vmax=0.1, aspect='auto', interpolation='none')
    ax.set_ylabel(labels[i])
    ax.get_yaxis().set_ticks([])
    ax.set_xticks(np.arange(length))
    ax.set_xticklabels([str(i) for i in range(length)])
    i+=1

axes[-1].annotate("State", xy=(0.5, -0.3), xycoords=axes[-1].get_window_extent,
                  xytext=(0,0), textcoords="offset points", ha='center', va='bottom')

#fig.subplots_adjust(bottom=0.1, top=0.9, left=0.0, right=0.7,
#                    wspace=0.02, hspace=0.02)

#cb_ax = fig.add_axes([0.87, 0.1, 0.02, 0.8])
cbar = fig.colorbar(im, ax=axes)
cbar.set_label('Value', rotation=270, labelpad=10.5)

plt.show()