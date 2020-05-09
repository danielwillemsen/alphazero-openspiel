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
class ToyGame:
    def __init__(self, length):
        self.length = length
        return

    def num_distinct_actions(self):
        return 4
    def new_initial_state(self):
        return State(0, self.length)

class State:
    def __init__(self, location1, length):
        self.location1 = location1
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
            return [0, 1, 2]
        else:
            return [0]

    def information_state(self):
        return str(self.location1)

    def apply_action(self, action):
        if self.current_player() == 0:
            if len(self.history()) > length * 5:
                self.terminal = True
                self.reward = 0.
                return
            if action == 0:
                self.reward = -1.
                self.terminal = True
            if action == 1:
                self.location1 += 1
                if self.location1 == self.length:
                    self.reward = 0.1
                    self.terminal = True
            if action == 2:
                self.reward = 0.
                self.terminal = True
            if action == 3:
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


def play_game_self(policy_fn, length, backup_type="on-policy", **kwargs):
    examples = []
    game = ToyGame(length)
    state = game.new_initial_state()
    num_distinct_actions = game.num_distinct_actions()
    alphazero_bot = AlphaZeroBot(game, 0, policy_fn, self_play=True, **kwargs)
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
                value_list = {action_temp: (child.N+child.P if child.N>0 else -99.0) for action_temp, child in node.children.items()}
                action_temp = max(value_list, key=value_list.get)
                node = node.children[action_temp]
                value_mult *= -1
            if node.N > 0:
                value = node.Q
                value_mult *=-1
            examples.append([state.information_state(), state.clone(), policy_list,
                             value * value_mult,
                             copy.deepcopy(alphazero_bot.mcts.root)])
            #examples.append([state.information_state(), state.clone(), policy_list,  value * value_mult * 5.0**(-abs(value *value_mult + alphazero_bot.mcts.root.Q)), copy.deepcopy(alphazero_bot.mcts.root)])
        if backup_type == "off-policy2":
            node = copy.deepcopy(alphazero_bot.mcts.root)
            value_mult = 1
            value = node.Q
            value_list = {action_temp: (child.N if child.N > 0 else -99.0) for action_temp, child in
                          node.children.items()}
            action_temp = max(value_list, key=value_list.get)
            node = node.children[action_temp]
            value_mult *= -1
            while not node.is_leaf():
                value = node.Q
                value_list = {action_temp: (child.P) for action_temp, child in node.children.items()}
                action_temp = max(value_list, key=value_list.get)
                if node.children[action_temp].N>0:
                    node = node.children[action_temp]
                    value_mult *= -1
                else:
                    break
            if node.N > 0:
                value = node.Q
                value_mult *= -1
            examples.append([state.information_state(), state.clone(), policy_list, value*value_mult, copy.deepcopy(alphazero_bot.mcts.root)])
        if backup_type == "off-policy-lambda":
            node = copy.deepcopy(alphazero_bot.mcts.root)
            value_mult = 1
            lambda_value = 0.0
            step = 1
            value_tot = 0.0
            while not node.is_leaf():
                value_mult *= -1
                value = node.Q
                value_list = {action_temp: (child.N+child.P if child.N>0 else -99.0) for action_temp, child in node.children.items()}
                action_temp = max(value_list, key=value_list.get)
                node = node.children[action_temp]

                value_tot += value * value_mult * (1-lambda_value)*lambda_value**(step-1)
                step += 1

            if node.N > 0:
                value = node.Q
                value_mult *=-1
                value_tot += value * value_mult * (1-lambda_value)*lambda_value**(step-1)
                step += 1
            value_tot -= value * value_mult * (1-lambda_value)*lambda_value**(step-2)
            value_tot += value * value_mult * lambda_value**(step-2)
            examples.append([state.information_state(), state.clone(), policy_list, value_tot])
        if backup_type == "off-policy-lambda-2":
            node = copy.deepcopy(alphazero_bot.mcts.root)
            value_mult = 1
            lambda_value = 0.75
            step = 1
            value_tot = 0.0
            while not node.is_leaf():
                value_mult *= -1
                value = node.value
                value_list = {action_temp: (child.N+child.P if child.N>0 else -99.0) for action_temp, child in node.children.items()}
                action_temp = max(value_list, key=value_list.get)
                node = node.children[action_temp]

                value_tot += value * value_mult * (1-lambda_value)*lambda_value**(step-1)
                step += 1

            if node.N > 0:
                value = node.value
                value_mult *=-1
                value_tot += value * value_mult * (1-lambda_value)*lambda_value**(step-1)
                step += 1
            value_tot -= value * value_mult * (1-lambda_value)*lambda_value**(step-2)
            value_tot += value * value_mult * lambda_value**(step-2)
            examples.append([state.information_state(), state.clone(), policy_list, value_tot, copy.deepcopy(alphazero_bot.mcts.root)])

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
        self.values = np.zeros((length, 2))
        #self.values = np.random.randn(length, 2)*0.2

        self.policy = np.zeros((length, 2, 4)) + 0.25
        self.visits = np.zeros((length, 2))
        self.extra = 0.999
        self.extra = np.zeros((length, 2)) + 0.999

    def policy_fn(self, state):
        loc1 = state.location1
        player = state.current_player()
        return list(self.policy[loc1, player, :]), self.values[loc1, player]/(1.0-self.extra[loc1, player])+np.random.randn(1)*0.1

length = 8
n_games = 40000
num_distinct_actions = 4
pvtable = PVTable(length)
alpha = 0.01
for i_game in range(n_games):
    examples = play_game_self(pvtable.policy_fn, length, keep_search_tree=False, n_playouts=100, c_puct=2.5, dirichlet_ratio=0.25)
    for example in examples:
        player = example[1].current_player()
        loc1 = example[1].location1

        policy = np.array(example[2])
        value = float(example[3])
        pvtable.extra[loc1, player] = (1-alpha)*pvtable.extra[loc1, player]
        pvtable.visits[loc1, player] += 1
        alpha_p = 0.025
        pvtable.values[loc1, player] = (1-alpha)*pvtable.values[loc1, player] + alpha * value
        pvtable.policy[loc1, player, :] = (1-alpha_p)*pvtable.policy[loc1, player] + alpha_p * policy
    if i_game%2000 == 0:
        #print(str(pvtable.policy[1,0,0]) + str(pvtable.values[1,0,0]) + str(pvtable.policy[3,0,0]) + str(pvtable.values[6,0,0])
        print(str(pvtable.values[:,0]/(1.0-pvtable.extra[:,0])))

        print(str(pvtable.values[:,0]))
        print(str(pvtable.policy[:,0,:]))

        print(str(pvtable.visits[:,0]))
print(str(pvtable.visits[:,0]))
final_visits.append(pvtable.visits[:,0])

pol_on = pvtable.policy[:, 0,:]
values_on = pvtable.values[:,0]/(1.0-pvtable.extra[:,0])

num_distinct_actions = 4
pvtable = PVTable(length)
alpha = 0.01
for i_game in range(n_games):
    examples = play_game_self(pvtable.policy_fn, length, backup_type="soft-Z", keep_search_tree=False, n_playouts=100, c_puct=2.5, dirichlet_ratio=0.25)
    for example in examples:
        player = example[1].current_player()
        loc1 = example[1].location1

        policy = np.array(example[2])
        value = float(example[3])
        pvtable.extra[loc1, player] = (1-alpha)*pvtable.extra[loc1, player]
        pvtable.visits[loc1, player] += 1
        alpha_p = 0.025
        pvtable.values[loc1, player] = (1-alpha)*pvtable.values[loc1, player] + alpha * value
        pvtable.policy[loc1, player, :] = (1-alpha_p)*pvtable.policy[loc1, player] + alpha_p * policy
    if i_game%2000 == 0:
        #print(str(pvtable.policy[1,0,0]) + str(pvtable.values[1,0,0]) + str(pvtable.policy[3,0,0]) + str(pvtable.values[6,0,0])
        print(str(pvtable.values[:,0]/(1.0-pvtable.extra[:,0])))

        print(str(pvtable.values[:,0]))
        print(str(pvtable.policy[:,0,:]))

        print(str(pvtable.visits[:,0]))
print(str(pvtable.visits[:,0]))
final_visits.append(pvtable.visits[:,0])

pol_soft = pvtable.policy[:,0,:]
values_soft = pvtable.values[:,0]/(1.0-pvtable.extra[:,0])

num_distinct_actions = 4
pvtable = PVTable(length)
alpha = 0.01
for i_game in range(n_games):
    examples = play_game_self(pvtable.policy_fn, length, backup_type="A0C", keep_search_tree=False, n_playouts=100, c_puct=2.5, dirichlet_ratio=0.25)
    for example in examples:
        player = example[1].current_player()
        loc1 = example[1].location1

        policy = np.array(example[2])
        value = float(example[3])
        pvtable.extra[loc1, player] = (1-alpha)*pvtable.extra[loc1, player]
        pvtable.visits[loc1, player] += 1
        alpha_p = 0.025
        pvtable.values[loc1, player] = (1-alpha)*pvtable.values[loc1, player] + alpha * value
        pvtable.policy[loc1, player, :] = (1-alpha_p)*pvtable.policy[loc1, player] + alpha_p * policy
    if i_game%2000 == 0:
        #print(str(pvtable.policy[1,0,0]) + str(pvtable.values[1,0,0]) + str(pvtable.policy[3,0,0]) + str(pvtable.values[6,0,0])
        print(str(pvtable.values[:,0]/(1.0-pvtable.extra[:,0])))

        print(str(pvtable.values[:,0]))
        print(str(pvtable.policy[:,0,:]))

        print(str(pvtable.visits[:,0]))
print(str(pvtable.visits[:,0]))
final_visits.append(pvtable.visits[:,0])

pol_A0C = pvtable.policy[:,0,:]
values_A0C = pvtable.values[:,0]/(1.0-pvtable.extra[:,0])

num_distinct_actions = 4
pvtable = PVTable(length)
alpha = 0.01
for i_game in range(n_games):
    examples = play_game_self(pvtable.policy_fn, length, backup_type="off-policy", keep_search_tree=False, n_playouts=100, c_puct=2.5, dirichlet_ratio=0.25)
    for example in examples:
        player = example[1].current_player()
        loc1 = example[1].location1

        policy = np.array(example[2])
        value = float(example[3])
        pvtable.extra[loc1, player] = (1-alpha)*pvtable.extra[loc1, player]
        pvtable.visits[loc1, player] += 1
        alpha_p = 0.025
        pvtable.values[loc1, player] = (1-alpha)*pvtable.values[loc1, player] + alpha * value
        pvtable.policy[loc1, player, :] = (1-alpha_p)*pvtable.policy[loc1, player] + alpha_p * policy
    if i_game%2000 == 0:
        #print(str(pvtable.policy[1,0,0]) + str(pvtable.values[1,0,0]) + str(pvtable.policy[3,0,0]) + str(pvtable.values[6,0,0])
        print(str(pvtable.values[:,0]/(1.0-pvtable.extra[:,0])))

        print(str(pvtable.values[:,0]))
        print(str(pvtable.policy[:,0,:]))

        print(str(pvtable.visits[:,0]))
print(str(pvtable.visits[:,0]))
final_visits.append(pvtable.visits[:,0])

pol_off = pvtable.policy[:,0,:]

values_off = pvtable.values[:,0]/(1.0-pvtable.extra[:,0])

pol_total = [pol_on, pol_soft, pol_A0C, pol_off]
values_total = [values_on, values_soft, values_A0C, values_off]


# obj0, obj1, obj2 are created here...

# Saving the objects:
with open('toy_res.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    p.dump([length,final_visits, values_total, pol_total], f)

labels = ["On-policy", "Soft-Z", "A0C", "Off-policy"]
cmap = cm.get_cmap('RdYlGn', 30)

i = 0
fig, axes = plt.subplots(nrows=len(values_total), ncols=1, figsize=(8,5), tight_layout=False, constrained_layout=True)
fig.set_tight_layout(False)
for ax in axes.flat:
    #ax.set_axis_off()
    im = ax.imshow(np.expand_dims(values_total[i],0), label=labels[i], cmap=cmap, vmin=-0.1, vmax=0.1)
    ax.set_ylabel(labels[i])
    ax.get_yaxis().set_ticks([])
    ax.set_xticks(np.arange(length))
    ax.set_xticklabels(["State " + str(i) for i in range(length)])
    i+=1

#fig.subplots_adjust(bottom=0.1, top=0.9, left=0.0, right=0.7,
#                    wspace=0.02, hspace=0.02)

#cb_ax = fig.add_axes([0.87, 0.1, 0.02, 0.8])
cbar = fig.colorbar(im, ax=axes)
cbar.set_label('Value', rotation=270, labelpad=10.5)
plt.show()

i = 0
fig, axes = plt.subplots(nrows=len(values_total),ncols=1, figsize=(8,5),constrained_layout=True)
for ax in axes.flat:
    #ax.set_axis_off()
    for it, pol in enumerate(pol_total[i]):
        if int(np.argmax(pol)) == 0:
            pol_str = "U"
            ax.annotate("", xy=(it, 0.25), xytext=(it, -0.25), arrowprops=dict(arrowstyle="->"))
        elif int(np.argmax(pol)) == 1:
            pol_str = ""
            ax.annotate("", xy=(it+0.25, 0), xytext=(it-0.25, 0), arrowprops=dict(arrowstyle="->"))
        elif int(np.argmax(pol)) == 2:
            pol_str = ""
            ax.annotate("", xy=(it+0, 0.25), xytext=(it, -0.25), arrowprops=dict(arrowstyle="->"))
        else:
            pol_str = "L"
        ax.text(it, 0, pol_str, ha='center', va='center')
    im = ax.imshow(np.expand_dims(pol_total[i][:,1],0), label=labels[i], cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_ylabel(labels[i])
    ax.get_yaxis().set_ticks([])
    ax.set_xticks(np.arange(length))
    ax.set_xticklabels(["State " + str(i) for i in range(length)])
    i+=1

# fig.subplots_adjust(bottom=0.1, top=0.9, left=0.0, right=0.9,
#                     wspace=0.02, hspace=0.02)

#cb_ax = fig.add_axes([0.87, 0.1, 0.02, 0.8])
cbar = fig.colorbar(im, ax=axes)
cbar.set_label('Policy of moving right', rotation=270, labelpad=10.5)
plt.show()
# cax = ax1.imshow(values_total,  cmap=cmap)
# ax1.set_yticks(np.arange(len(labels)))
# ax1.set_yticklabels(labels)
# ax1.set_xticks(np.arange(length))
# ax1.set_xticklabels([str(i) for i in range(length)])

# plt.xlabel("State")
# fig.colorbar(cax)
# plt.show()
# length_cliff = 10
# start = [1,0]
#
# rewards = np.zeros((height, length_cliff))
# rewards[0,:] = -1
# rewards[0,-1] = 1
# rewards[-1,:] = 0.5
# print(rewards)

