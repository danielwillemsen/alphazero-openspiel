from alphazerobot import AlphaZeroBot
from game_utils import *
from mcts import MCTS
from matplotlib import pyplot as plt
import pyspiel
from open_spiel.python.algorithms import mcts

# game = pyspiel.load_game("connect_four")
# state = game.new_initial_state()
# n_playouts = 10
#
# state.apply_action(1)
# state.apply_action(2)
# state.apply_action(1)
# state.apply_action(2)
# state.apply_action(3)
# state.apply_action(4)
# state.apply_action(5)
# state.apply_action(6)
# state.apply_action(3)
# state.apply_action(4)
# state.apply_action(5)
# state.apply_action(3)
# state.apply_action(3)
# state.apply_action(1)
# state.apply_action(5)
# state.apply_action(2)
#
# my_mcts = MCTS(MCTS.random_rollout, n_playouts=n_playouts)
# my_probs = [item*(n_playouts-1) for item in my_mcts.search(state)]
# logger.info(my_probs)
# logger.info(mcts.mcts_search(state, 1, n_playouts, mcts.RandomRolloutEvaluator(25)))
# #
playout_list = [10, 20, 50, 100, 200]
scores_list = []
scores_list_keep_tree = []
scores_list_p = []
scores_list_p_keep_tree = []
n_tests = 100
score_tot = 0.
for n_playouts in playout_list:
    score_tot = 0.
    for i in range(n_tests):
        score1, score2 = test_zero_vs_mcts(MCTS.random_rollout, n_playouts, n_playouts=n_playouts,
                                           keep_search_tree=False, use_puct=False)
        score_tot += score1
        score_tot += score2
    avg = score_tot / (2 * n_tests)
    scores_list.append(avg)
    logger.info("Average score vs mcts" + str(n_playouts) + ":" + str(avg))

for n_playouts in playout_list:
    score_tot = 0.
    for i in range(n_tests):
        score1, score2 = test_zero_vs_mcts(MCTS.random_rollout, n_playouts, n_playouts=n_playouts,
                                           keep_search_tree=True, use_puct=False)
        score_tot += score1
        score_tot += score2
    avg = score_tot / (2 * n_tests)
    scores_list_keep_tree.append(avg)
    logger.info("Average score vs mcts" + str(n_playouts) + ":" + str(avg))

for n_playouts in playout_list:
    score_tot = 0.
    for i in range(n_tests):
        score1, score2 = test_zero_vs_mcts(MCTS.random_rollout, n_playouts, n_playouts=n_playouts,
                                           keep_search_tree=False, use_puct=True)
        score_tot += score1
        score_tot += score2
    avg = score_tot / (2 * n_tests)
    scores_list_p.append(avg)
    logger.info("Average score vs mcts" + str(n_playouts) + ":" + str(avg))

for n_playouts in playout_list:
    score_tot = 0.
    for i in range(n_tests):
        score1, score2 = test_zero_vs_mcts(MCTS.random_rollout, n_playouts, n_playouts=n_playouts,
                                           keep_search_tree=True, use_puct=True)
        score_tot += score1
        score_tot += score2
    avg = score_tot / (2 * n_tests)
    scores_list_p_keep_tree.append(avg)
    logger.info("Average score vs mcts" + str(n_playouts) + ":" + str(avg))

plt.plot(playout_list, scores_list, label="UCT")
plt.plot(playout_list, scores_list_keep_tree, label="UCT, tree retained")
plt.plot(playout_list, scores_list_p, label="PUCT")
plt.plot(playout_list, scores_list_p_keep_tree, label="PUCT, tree retained")
plt.grid()
plt.xlabel("Games Played")
plt.ylabel("Average Reward")
plt.legend()
plt.show()

