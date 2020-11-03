from os import listdir
from os.path import isfile, join
import pickle
from game_utils import test_net_vs_net
from toy_domain_net import PVTable_tictactoe
import numpy as np
import matplotlib.pyplot as plt
from train import Trainer
from torch import multiprocessing
import torch
from network import Net #SmallNet as Net
import pyspiel
import random
from tic_tac_toe_readin import get_move_dict
from scipy import stats

#Input path with all stored networks
mypath = "./models/toynets_NN_meteor_bignet/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print(len(onlyfiles))
runid = "1"
game = pyspiel.load_game("tic_tac_toe")



state_shape = game.observation_tensor_shape()
num_distinct_actions = game.num_distinct_actions()
backup_types = ["on-policy", "soft-Z", "A0C", "off-policy"]

paths = []
path_str = []
moves = [j for j in range(9)]
random.seed(0)
while len(paths) < 250:
    pathlength = random.randint(0,9)
    path = random.sample(moves, pathlength)
    state = game.new_initial_state()
    for action in path:
        state.apply_action(action)
        if state.is_terminal():
            break
    if not state.is_terminal() and str(state) not in path_str and len(state.legal_actions()) > 1:
        paths.append(path)
        path_str.append(str(state))

print(len(paths))
print("Loading state dict")
state_dict = get_move_dict()
print("state dict loaded")
state_dict_correct_moves = {}
state_dict_all_moves = {}
for key, val in state_dict.items():
    max_val = np.max([v for v in val.values()])
    correct_moves = []
    all_moves = []
    for move, v2 in val.items():
        if v2 == max_val:
            correct_moves.append(move)
        all_moves.append(move)
    state_dict_correct_moves[key] = correct_moves
    state_dict_all_moves[key] = all_moves

print("state dict made")
if "NN" in mypath:
    types = ["NN"]
else:
    types = ["Tabular"]
game_no = [i for i in range(0, 10001, 500)]
for type in types:
    kwargs = {"settings1": {}, "settings2": {}}
    plot_data = {}
    plot_data_pol = {}
    plot_data_val_diff = {}
    for backup_type in backup_types:
        correct_percentage = []
        correct_pol = []
        val_diff = []
        for run in range(10):
            print(str(run) + backup_type)

            correct_percentage_single_run = []
            correct_pol_single_run = []
            val_diff_single_run = []
            for i_game in game_no:
                corrects = []
                pols = []
                val_diffs = []
                if type == "NN":
                    trainer_self = Trainer()
                    trainer_self.current_net = Net(state_shape, num_distinct_actions)

                    trainer_self.current_net.load_state_dict(
                        torch.load(mypath + runid + "_" + backup_type + "_" + str(run) + "_" + str(i_game) + ".pth"))
                    trainer_self.current_net.eval()
                    policy_fn_self = trainer_self.current_net.predict
                else:
                    policy_fn_self = pickle.load(open(mypath + runid + "_" + backup_type + "_" + str(run) + "_" + str(i_game) + ".p", "rb")).policy_fn
                for path in paths:
                    state = game.new_initial_state()
                    for action in path:
                        state.apply_action(action)
                    pol, val = policy_fn_self(state)

                    # Fig 12
                    found = False
                    while not found:
                        act = np.argmax(pol)
                        if act in state_dict_all_moves[str(state)]:
                            found = True
                            if act in state_dict[str(state)].keys() and state_dict[str(state)][act] == max(state_dict[str(state)].values()):
                                corrects.append(1.0)
                            else:
                                corrects.append(0.0)
                        else:
                            pol[act] = -99.

                    # Fig 11
                    pol_cor = 0.0
                    pol_tot = 0.0
                    for act, p in enumerate(pol):
                        if act in state_dict_correct_moves[str(state)]:
                            pol_cor += p
                        if act in state_dict_all_moves[str(state)]:
                            pol_tot += p
                    pols.append(pol_cor/pol_tot)


                    # Fig 10
                    v_d = np.abs(val - max(state_dict[str(state)].values()))
                    val_diffs.append(v_d)
                val_diff_single_run.append(np.mean(val_diffs))
                correct_pol_single_run.append(np.mean(pols))
                correct_percentage_single_run.append(np.mean(corrects))
            val_diff.append(val_diff_single_run)
            correct_percentage.append(correct_percentage_single_run)
            correct_pol.append(correct_pol_single_run)
        plot_data[backup_type] = correct_percentage
        plot_data_pol[backup_type] = correct_pol
        plot_data_val_diff[backup_type] = val_diff

markers = ["o", "x", "+", "v"]
linestyles = ["-", "--", "-.", "dotted"]
labelnames = {"on-policy":  "AlphaZero", "soft-Z": "Soft-Z", "A0C": "A0C", "off-policy": "A0GB"}

i=0
plt.figure()
pickle.dump(plot_data, open("../" + str(types[0]) + "_plot_data.p", "wb"))

for key, val in plot_data.items():
    mean_scores = np.mean(val, axis=0)
    std_scores = stats.sem(val, axis=0)
    plt.plot(game_no, mean_scores, label=labelnames[key], marker=markers[i], linestyle=linestyles[i])
    plt.fill_between(game_no, (mean_scores - std_scores), (mean_scores + std_scores), alpha=0.25)
    i+=1
plt.legend()
plt.ylabel("Chance of selecting correct move")
plt.xlabel("Games played")
plt.grid()

# plt.ylim(0.,1.)
plt.show()
i=0

plt.figure()
pickle.dump(plot_data_pol, open("../" + str(types[0]) + "_plot_data_pol.p", "wb"))
for key, val in plot_data_pol.items():
    mean_scores = np.mean(val, axis=0)
    std_scores = stats.sem(val, axis=0)
    plt.plot(game_no, mean_scores, label=labelnames[key], marker=markers[i], linestyle=linestyles[i])
    plt.fill_between(game_no, (mean_scores - std_scores), (mean_scores + std_scores), alpha=0.25)
    i+=1

plt.legend()
plt.ylabel("Total policy of correct moves")
plt.xlabel("Games played")
plt.grid()

# plt.ylim(0.,1.)
plt.show()

i=0

plt.figure()
pickle.dump(plot_data_val_diff, open("../" + str(types[0]) + "_plot_data_val_diff.p", "wb"))

for key, val in plot_data_val_diff.items():
    mean_scores = np.mean(val, axis=0)
    std_scores = stats.sem(val, axis=0)
    plt.plot(game_no, mean_scores, label=labelnames[key], marker=markers[i], linestyle=linestyles[i])
    plt.fill_between(game_no, (mean_scores - std_scores), (mean_scores + std_scores), alpha=0.25)
    i+=1

plt.legend()
plt.ylabel("Mean absolute error in value estimation")
plt.xlabel("Games played")
plt.grid()

# plt.ylim(0.,1.5)
plt.show()