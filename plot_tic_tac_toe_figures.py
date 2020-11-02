from os import listdir
from os.path import isfile, join
import pickle
from game_utils import test_net_vs_net
from toy_domain_net import PVTable_tictactoe
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from train import Trainer
from torch import multiprocessing
import torch
from network import Net #SmallNet as Net
import pyspiel
import random
from scipy import stats
import time
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)
# plt.rcParams["figure.figsize"] = [5,4]
markers = ["o", "x", "+", "v"]
linestyles = ["-", "--", "-.", "dotted"]
labelnames = {"on-policy":  "AlphaZero", "soft-Z": "Soft-Z", "A0C": "A0C", "off-policy": "A0GB"}

## TABULAR
game_no = [i for i in range(0, 50001, 2500)]
types = ["Tabular"]


# Fig 10 a
plt.figure()
plot_data_val_diff = pickle.load(open("../" + str(types[0]) + "_plot_data_val_diff.p", "rb"))
i=0
for key, val in plot_data_val_diff.items():
    mean_scores = np.mean(val, axis=0)
    std_scores = stats.sem(val, axis=0)
    plt.plot(game_no, mean_scores, label=labelnames[key], marker=markers[i], linestyle=linestyles[i])
    plt.fill_between(game_no, (mean_scores - std_scores), (mean_scores + std_scores), alpha=0.25)
    i+=1
plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.14)
plt.legend()
plt.ylabel("MAE of value estimates")
plt.xlabel("Games played")
plt.grid()
plt.savefig("../fig10a.png")

plt.show()

# Fig 11 a
plt.figure()
plot_data_pol = pickle.load(open("../" + str(types[0]) + "_plot_data_pol.p", "rb"))
i = 0
for key, val in plot_data_pol.items():
    mean_scores = np.mean(val, axis=0)
    std_scores = stats.sem(val, axis=0)
    plt.plot(game_no, mean_scores, label=labelnames[key], marker=markers[i], linestyle=linestyles[i])
    plt.fill_between(game_no, (mean_scores - std_scores), (mean_scores + std_scores), alpha=0.25)
    i+=1
plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.14)
plt.legend()
plt.ylabel("Total policy of correct moves")
plt.xlabel("Games played")
plt.grid()
plt.savefig("../fig11a.png")

plt.show()

# Fig 12 a
i=0
plt.figure()
plot_data = pickle.load(open("../" + str(types[0]) + "_plot_data.p", "rb"))

for key, val in plot_data.items():
    mean_scores = np.mean(val, axis=0)
    std_scores = stats.sem(val, axis=0)
    plt.plot(game_no, mean_scores, label=labelnames[key], marker=markers[i], linestyle=linestyles[i])
    plt.fill_between(game_no, (mean_scores - std_scores), (mean_scores + std_scores), alpha=0.25)
    i+=1
plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.14)

plt.legend()
plt.ylabel("Chance of selecting correct move")
plt.xlabel("Games played")
plt.grid()
plt.savefig("../fig12a.png")

plt.show()


# Fig 13 a

plot_data = pickle.load(open("../tournament_Tabular.p", "rb"))

plt.figure()
i=0
for key, val in plot_data.items():
    mean_scores = np.mean(val, axis=0)
    std_scores = stats.sem(val, axis=0)
    plt.plot(game_no, mean_scores, label=labelnames[key], marker=markers[i], linestyle=linestyles[i])
    plt.fill_between(game_no, (mean_scores - std_scores), (mean_scores + std_scores), alpha=0.25)
    i += 1
plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.14)

plt.legend()
plt.xlabel("Games played")
plt.ylabel("Win rate vs baselines")

plt.grid()
plt.ylim(0.,1.)
plt.savefig("../fig13a.png")
plt.show()

# TABULAR
game_no = [i for i in range(0, 10001, 500)]
types = ["NN"]
# Fig 10 b
plt.figure()
plot_data_val_diff = pickle.load(open("../" + str(types[0]) + "_plot_data_val_diff.p", "rb"))
i=0
for key, val in plot_data_val_diff.items():
    mean_scores = np.mean(val, axis=0)
    std_scores = stats.sem(val, axis=0)
    plt.plot(game_no, mean_scores, label=labelnames[key], marker=markers[i], linestyle=linestyles[i])
    plt.fill_between(game_no, (mean_scores - std_scores), (mean_scores + std_scores), alpha=0.25)
    i+=1
plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.14)
plt.legend()
plt.ylim(0.,0.95)
plt.ylabel("MAE of value estimates")
plt.xlabel("Games played")
plt.grid()
plt.savefig("../fig10b.png")

plt.show()

# Fig 11 b
plt.figure()
plot_data_pol = pickle.load(open("../" + str(types[0]) + "_plot_data_pol.p", "rb"))
i = 0
for key, val in plot_data_pol.items():
    mean_scores = np.mean(val, axis=0)
    std_scores = stats.sem(val, axis=0)
    plt.plot(game_no, mean_scores, label=labelnames[key], marker=markers[i], linestyle=linestyles[i])
    plt.fill_between(game_no, (mean_scores - std_scores), (mean_scores + std_scores), alpha=0.25)
    i+=1
plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.14)
plt.legend()
plt.ylabel("Total policy of correct moves")
plt.xlabel("Games played")
plt.grid()
plt.savefig("../fig11b.png")

plt.show()

# Fig 12 b
i=0
plt.figure()
plot_data = pickle.load(open("../" + str(types[0]) + "_plot_data.p", "rb"))

for key, val in plot_data.items():
    mean_scores = np.mean(val, axis=0)
    std_scores = stats.sem(val, axis=0)
    plt.plot(game_no, mean_scores, label=labelnames[key], marker=markers[i], linestyle=linestyles[i])
    plt.fill_between(game_no, (mean_scores - std_scores), (mean_scores + std_scores), alpha=0.25)
    i+=1
plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.14)

plt.legend()
plt.ylabel("Chance of selecting correct move")
plt.xlabel("Games played")
plt.grid()
plt.savefig("../fig12b.png")

plt.show()

# Fig 13 b


plot_data = pickle.load(open("../tournament_NN.p", "rb"))

plt.figure()
i=0
for key, val in plot_data.items():
    mean_scores = np.mean(val, axis=0)
    std_scores = stats.sem(val, axis=0)
    plt.plot(game_no, mean_scores, label=labelnames[key], marker=markers[i], linestyle=linestyles[i])
    plt.fill_between(game_no, (mean_scores - std_scores), (mean_scores + std_scores), alpha=0.25)
    i += 1
plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.14)

plt.legend()
plt.xlabel("Games played")
plt.ylabel("Win rate vs baselines")

plt.grid()
plt.ylim(0.,1.)
plt.savefig("../fig13b.png")

#plt.savefig("/export/scratch2/jdw/tournament.png")
plt.show()