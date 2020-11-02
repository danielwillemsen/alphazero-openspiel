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
from network import Net
from train import Trainer
import torch
import pickle
import time
import sys
torch.set_num_threads(1)

class PVTable:
    def __init__(self, length):
        self.values = np.zeros((length, 2))

        self.policy = np.zeros((length, 2, 4)) + 0.25
        self.visits = np.zeros((length, 2))

        # Value table uses exponentially weighted moving average. Correction "extra" corrects for the lower number of samples at the start.
        self.extra = np.zeros((length, 2)) + 0.999

    def policy_fn(self, state):
        loc1 = state.location1
        player = state.current_player()
        return list(self.policy[loc1, player, :]), self.values[loc1, player] / (1.0-self.extra[loc1, player])

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
#backup_types = ["A0C", "off-policy"]#, "greedy-forward", "greedy-forward-N"]

game = pyspiel.load_game("tic_tac_toe")
length = 50
n_games = 20001
#num_distinct_actions = 4
state_shape = game.observation_tensor_shape()
num_distinct_actions = game.num_distinct_actions()
alpha = 0.025
backup_res = {backup_type: [] for backup_type in backup_types}
net = Net(state_shape, num_distinct_actions)
#tables = [tab for tab in pvtables.values()]
#use_table = pvtables["off-policy"]
#use_table.tables = tables
state = game.new_initial_state()
initial_sequence = []
best_next_actions = [3, 4, 6]
bad_next_actions = [2,5,7,8]

state2 = game.new_initial_state()

save = True

for act in initial_sequence:
    state.apply_action(act)

good_states = []
for act in best_next_actions:
    state2 = state.clone()
    state2.apply_action(act)
    good_states.append(state2)

bad_states = []
for act in bad_next_actions:
    state2 = state.clone()
    state2.apply_action(act)
    bad_states.append(state2)

n_runs = 1
runid = "1"
tot_games_played = 0
time_start = time.time()
# runs = [1]
runs = [int(sys.argv[1])]
print("runs:" + str(runs))
for type in ["NN"]:#, "Tabular, random init"]#, "NN"]:
    tot_games = n_games*len(runs)*len(backup_types)

    val_dict = {}
    correct_dict = {}
    correct_perc_dict = {}
    advantage_dict = {}
    for backup_type in backup_types:
        val_lists = []
        correct_lists = []
        correct_perc_lists = []
        advantage_lists = []
        for run in runs:
            if type == "NN":
                trainer = Trainer(game="tic_tac_toe")
                trainer.batch_size = 256
                trainer.n_batches_per_generation = 500
                trainer.n_games_buffer = 999999
                torch.manual_seed(0)
                net = Net(state_shape, num_distinct_actions)
                trainer.current_net = net
                trainer.current_net.eval()

                trainer.optimizer = torch.optim.Adam(trainer.current_net.parameters(), lr=trainer.lr, weight_decay=0.0001)
            elif type == "Tabular":
                pvtables = {backup_type: PVTable_tictactoe(length, use_random=False) for backup_type in backup_types}
            else:
                pvtables = {backup_type: PVTable_tictactoe(length, use_random=True) for backup_type in backup_types}



            val_list = []
            correct_list = []
            correct_perc_list = []
            advantage_list = []
            for i_game in range(n_games + 1):
                tot_games_played += 1
                if type == "NN":
                    examples = play_game_self(trainer.current_net.predict, "tic_tac_toe",
                                              keep_search_tree=False,
                                              n_playouts=100,
                                              c_puct=2.5,
                                              dirichlet_ratio=0.25,
                                              backup=backup_type,
                                              backup_types=backup_types,
                                              length=length,
                                              initial_sequence=initial_sequence)# 0 4 2
                    # train_net()
                    trainer.buffer.append(examples)
                # for example in examples:
                #     trainer.buffer.append(example)
                    if i_game % 500 == 499:
                        trainer.train_network()
                    pol, val = trainer.current_net.predict(state)
                else:
                    examples = play_game_self(pvtables[backup_type].policy_fn, "tic_tac_toe",
                                              keep_search_tree=False,
                                              n_playouts=100,
                                              c_puct=2.5,
                                              dirichlet_ratio=0.25,
                                              backup=backup_type,
                                              backup_types=backup_types,
                                              length=length,
                                              initial_sequence=initial_sequence)
                    for example in examples:
                        update_pvtables_tictactoe(example, pvtables)

                    pol, val = pvtables[backup_type].policy_fn(state)

                    # Advantage computation
                    bad_vals = []
                    for bad_state in bad_states:
                        _, v = pvtables[backup_type].policy_fn(bad_state)
                        bad_vals.append(v)

                    good_vals = []
                    for good_state in good_states:
                        _, v = pvtables[backup_type].policy_fn(good_state)
                        good_vals.append(v)

                    advantage = np.mean([good_vals]) - np.mean([bad_vals])
                if save:
                    if i_game%500 == 0:
                        if type == "NN":
                            torch.save(trainer.current_net.state_dict(),
                                       "/export/scratch2/jdw/models/toynets_NN_meteor_bignet" + "/" + runid + "_" + backup_type + "_" + str(run) + "_" + str(i_game) + ".pth")
                        else:
                            pickle.dump(pvtables[backup_type], open("/export/scratch2/jdw/models/toynets_meteor" + "/" + runid + "_" + backup_type + "_" + str(run) + "_" + str(i_game) + ".p", "wb"))

                # advantage_list.append(advantage)
                val_list.append(val)
                correct_list.append(np.argmax(pol) in best_next_actions)
                correct_perc_list.append(np.sum(np.array(pol)[best_next_actions]))
                    # update_pvtables_tictactoe(example, pvtables)

                    # #For further data visualization
                    # for key, pvtable in pvtables.items():
                    #     #backup_res[key].append(np.copy(pvtable.values))
                    #     backup_res[key].append(np.copy(pvtable.values[""]))
            # if backup_res["off-policy"][-1][3,0] > 0.0001:
            #     a=2

                # if i_game%2 == 0:
                #     print("Game_no:", i_game, "/", n_games)
                if i_game%200==0:
                    pol, val = trainer.current_net.predict(state2)
                    print(val)
                    time_current = time.time()
                    print("Games played: " + str(tot_games_played) + " / " + str(tot_games))
                    print("Time elapsed: " + str(time_current - time_start) + " / " + str((time_current - time_start)/tot_games_played*tot_games))
        #
        #     val_lists.append(val_list)
        #     correct_lists.append(correct_list)
        #     correct_perc_lists.append(correct_perc_list)
        #     advantage_lists.append(advantage_list)
        #
        # val_dict[backup_type] = val_lists
        # correct_dict[backup_type] = correct_lists
        # correct_perc_dict[backup_type] = correct_perc_lists
        # advantage_dict[backup_type] = advantage_lists
    #
    # plt.figure()
    # for key, val in val_dict.items():
    #     val_flat = [np.mean(vals) for vals in zip(*val)]
    #     plt.plot(val_flat, label=key)
    #     plt.xlabel("Games played")
    #     plt.ylabel("Value estimate of state (" + type +") (Winnable in 5 turns)")
    #     plt.legend()
    #     plt.savefig("./plots/" + runid + type + "value" +".png")
    #
    # plt.figure()
    # for key, val in advantage_dict.items():
    #     val_flat = [np.mean(vals) for vals in zip(*val)]
    #     plt.plot(val_flat, label=key)
    #     plt.xlabel("Games played")
    #     plt.ylabel("Advantage estimate of state (" + type +") (Winnable in 5 turns)")
    #     plt.legend()
    #     plt.savefig("./plots/" + runid + type + "advantage" +".png")
    #
    # plt.figure()
    # for key, val in correct_perc_dict.items():
    #     val_flat = [np.mean(vals) for vals in zip(*val)]
    #     plt.plot(val_flat, label=key)
    #     plt.xlabel("Games played")
    #     plt.ylabel("Policy of correct move (" + type +") (Winnable in 5 turns)")
    #     plt.legend()
    #     plt.savefig("./plots/" + runid + type + "policy" +".png")
    #
    # plt.figure()
    # for key, val in correct_dict.items():
    #     val_flat = [np.mean(vals) for vals in zip(*val)]
    #     plt.plot(val_flat, label=key)
    #     plt.xlabel("Games played")
    #     plt.ylabel("Percentage correct move selected (" + type +") (Winnable in 5 turns)")
    #
    #     plt.legend()
    #     plt.savefig("./plots/" + runid + type + "moveselect"  +".png")
#plt.show()
#1]
# best_next_actions = [3, 4, 6]
# backup_plot_res = {backup_type: [] for backup_type in backup_types}
# for key, val in backup_res.items():
#     backup_res[key] = np.stack(val, axis=0)
#
# for key, val in backup_res.items():
#     plt.plot(val, label=key)
#     plt.xlabel("Backup Number")
#     plt.ylabel("Value estimate of state (Tieable in 6 turns)")
#
# #    plt.plot(val[:, 3, 0], label=key)
# plt.legend()
# plt.show()
#
# labels = [type_name.title() for type_name in backup_types]
# cmap = cm.get_cmap('RdYlGn', 30)
#
# i = 0
# fig, axes = plt.subplots(nrows=len(backup_types), ncols=1, figsize=(8,8), tight_layout=False, constrained_layout=True)
# axes[0].set_title(f"Progress over {n_games} games")
#
# fig.set_tight_layout(False)
# for ax in axes.flat:
#     #ax.set_axis_off()
# #     im = ax.imshow(np.expand_dims(values_total[i],0), label=labels[i], cmap=cmap, vmin=-0.1, vmax=0.1)
#     im = ax.imshow(backup_res[backup_types[i]][:,:,0], label=backup_types[i], cmap=cmap, vmin=-0.1, vmax=0.1, aspect='auto', interpolation='none')
#     ax.set_ylabel(labels[i])
#     ax.get_yaxis().set_ticks([])
#     ax.set_xticks(np.arange(length))
#     ax.set_xticklabels([str(i) for i in range(length)])
#     i+=1
#
# axes[-1].annotate("State", xy=(0.5, -0.3), xycoords=axes[-1].get_window_extent,
#                   xytext=(0,0), textcoords="offset points", ha='center', va='bottom')
#
# #fig.subplots_adjust(bottom=0.1, top=0.9, left=0.0, right=0.7,
# #                    wspace=0.02, hspace=0.02)
#
# #cb_ax = fig.add_axes([0.87, 0.1, 0.02, 0.8])
# cbar = fig.colorbar(im, ax=axes)
# cbar.set_label('Value', rotation=270, labelpad=10.5)
#
# plt.show()