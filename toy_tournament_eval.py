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
from scipy import stats
import time

torch.set_num_threads(1)

mypath = "./models/toynets_meteor/"#models/toynets_NN_meteor_bignet/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and int(f.split("_")[-1].split(".")[0])<=50000]
print(len(onlyfiles))

#500 opponents
opponents = random.sample(onlyfiles, 500)

print(len(onlyfiles))
runid = "1"
game = pyspiel.load_game("tic_tac_toe")
length = 50
n_games = 200
# num_distinct_actions = 4
state_shape = game.observation_tensor_shape()
num_distinct_actions = game.num_distinct_actions()
backup_types = ["on-policy", "soft-Z", "A0C", "off-policy"]#, "greedy-forward", "greedy-forward-N"]
print(len(onlyfiles))
gamerange = [i for i in range(0, 50001, 2500)]
n_runs = 10
n_games_total = len(gamerange)*len(backup_types)*n_runs*len(opponents)
time_start = time.time()
tot_games_played = 0
for type in ["Tabular"]:
    if type == "Tabular":
        opponents_loaded = []
        for opponent in opponents:
            opponents_loaded.append(pickle.load(open(mypath + opponent, "rb")).policy_fn)
    kwargs = {"settings1": {}, "settings2": {}}
    plot_data = {}
    for backup_type in backup_types:
        print("Backup: " + str(backup_type))
        mean_scores = []
        for run in range(n_runs):
            tot_games_played += 1
            print("Run: " + str(run))
            mean_scores_single_run = []
            for i_game in gamerange:
                print(i_game)
                scores = []
                time_current = time.time()
                print("Games played: " + str(tot_games_played) + " / " + str(n_games_total))
                print("Time elapsed: " + str(time_current - time_start) + " / " + str(
                    (time_current - time_start) / tot_games_played * n_games_total))
                tot_games_played += len(opponents)
                if type == "NN":
                    trainer_self = Trainer()
                    trainer_self.current_net = Net(state_shape, num_distinct_actions)

                    trainer_self.current_net.load_state_dict(
                        torch.load(mypath + runid + "_" + backup_type + "_" + str(run) + "_" + str(i_game) + ".pth"))
                    trainer_self.current_net.eval()

                    policy_fn_self = trainer_self.current_net.predict
                else:
                    policy_fn_self = pickle.load(open(mypath + runid + "_" + backup_type + "_" + str(run) + "_" + str(i_game) + ".p", "rb")).policy_fn
                for i_op, opponent in enumerate(opponents):
                    if type == "NN":
                        trainer_opponent = Trainer()
                        trainer_opponent.current_net = Net(state_shape, num_distinct_actions)

                        trainer_opponent.current_net.load_state_dict(
                            torch.load(mypath+opponent))
                        trainer_opponent.current_net.eval()

                        policy_fn_opponent = trainer_opponent.current_net.predict
                    else:
                        policy_fn_opponent = opponents_loaded[i_op]#pickle.load(open(mypath+opponent, "rb")).policy_fn

                    score1, score2, _ = test_net_vs_net(policy_fn_self, 100, "tic_tac_toe", policy_fn2=policy_fn_opponent, **kwargs)
                    scores.append(score1)
                    scores.append(score2)
                mean_scores_single_run.append(np.mean(scores)*0.5+0.5)
            mean_scores.append(mean_scores_single_run)
        plot_data[backup_type] = mean_scores

markers = ["o", "x", "+", "v"]
linestyles = ["-", "--", "-.", "dotted"]
labelnames = {"on-policy":  "AlphaZero", "soft-Z": "Soft-Z", "A0C": "A0C", "off-policy": "A0GB"}
pickle.dump(plot_data, open("../tournament_Tabular.p", "wb"))

plt.figure()
i=0
for key, val in plot_data.items():
    mean_scores = np.mean(val, axis=0)
    std_scores = stats.sem(val, axis=0)
    plt.plot(gamerange, mean_scores, label=labelnames[key], marker=markers[i], linestyle=linestyles[i])
    plt.fill_between(gamerange, (mean_scores - std_scores), (mean_scores + std_scores), alpha=0.25)
    i += 1
plt.legend()
plt.xlabel("Games played")
plt.ylabel("Win rate vs baselines")

plt.grid()
plt.ylim(0.,1.)
#plt.savefig("/export/scratch2/jdw/tournament.png")
plt.show()