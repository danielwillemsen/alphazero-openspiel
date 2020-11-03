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
        self.visits = dict()

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


        return list(pol), val

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

backup_types = ["on-policy", "soft-Z", "A0C", "off-policy"]
game = pyspiel.load_game("tic_tac_toe")
length = 50
n_games = 20001

state_shape = game.observation_tensor_shape()
num_distinct_actions = game.num_distinct_actions()
alpha = 0.025
backup_res = {backup_type: [] for backup_type in backup_types}
net = Net(state_shape, num_distinct_actions)

state = game.new_initial_state()
initial_sequence = []

save = True

for act in initial_sequence:
    state.apply_action(act)

n_runs = 1
runid = "1"
tot_games_played = 0
time_start = time.time()
runs = [int(sys.argv[1])]
print("runs:" + str(runs))


types_alphazero = ["NN"] #or ["Tabular"]

for type in types_alphazero:
    tot_games = n_games*len(runs)*len(backup_types)

    for backup_type in backup_types:
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
                                              initial_sequence=initial_sequence)
                    trainer.buffer.append(examples)
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

                if save:
                    if i_game%500 == 0:
                        if type == "NN":
                            torch.save(trainer.current_net.state_dict(),
                                       "/export/scratch2/jdw/models/toynets_NN_meteor_bignet" + "/" + runid + "_" + backup_type + "_" + str(run) + "_" + str(i_game) + ".pth")
                        else:
                            pickle.dump(pvtables[backup_type], open("/export/scratch2/jdw/models/toynets_meteor" + "/" + runid + "_" + backup_type + "_" + str(run) + "_" + str(i_game) + ".p", "wb"))

                if i_game%200==0:
                    pol, val = trainer.current_net.predict(state2)
                    print(val)
                    time_current = time.time()
                    print("Games played: " + str(tot_games_played) + " / " + str(tot_games))
                    print("Time elapsed: " + str(time_current - time_start) + " / " + str((time_current - time_start)/tot_games_played*tot_games))
