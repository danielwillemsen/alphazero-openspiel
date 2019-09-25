from torch import multiprocessing
import time

import numpy as np
import pyspiel
import torch
import os
from alphazerobot import AlphaZeroBot
from connect4net import Net
from game_utils import play_game_self, test_zero_vs_mcts, test_net_vs_mcts
from state_to_board import state_to_board
import copy


def generate_single_game(conn):
    niceness=os.nice(0)
    os.nice(5-niceness)
    time.sleep(float(np.random.rand())*10)
    evaluator = Evaluator(conn)
    example = play_game_self(evaluator.evaluate_nn)
    return example


def test_net_game_vs_mcts100(conn):
    niceness=os.nice(0)
    os.nice(5-niceness)
    evaluator = Evaluator(conn)
    score1, score2 = test_net_vs_mcts(evaluator.evaluate_nn, 100)
    return score1 + score2


def test_net_game_vs_mcts200(conn):
    niceness=os.nice(0)
    os.nice(5-niceness)
    evaluator = Evaluator(conn)
    score1, score2 = test_net_vs_mcts(evaluator.evaluate_nn, 200)
    return score1 + score2


def test_zero_game_vs_mcts200(conn):
    niceness=os.nice(0)
    os.nice(5-niceness)
    evaluator = Evaluator(conn)
    score1, score2 = test_zero_vs_mcts(evaluator.evaluate_nn, 200)
    return score1 + score2


class Evaluator():
    def __init__(self, conn):
        self.conn = conn

    def evaluate_nn(self, state):
        """

        @param state: The game which needs to be evaluated
        @return: pi: policy according to neural net, vi: value according to neural net.
        """
        board = state_to_board(state)
        self.conn.send(board)
        pi, vi = self.conn.recv()
        vi = float(vi[0])
        return pi, vi


def handle_gpu(net, parent_conns, is_done):
    """Thread which continuously pushes items from the gpu_queue to the gpu. This results in multiple games being
    evaluated in a single batch

    @return:
    """
    all_games_finished = False
    while not all_games_finished:
        with is_done.get_lock():
            if is_done.value == 1:
                all_games_finished = True

        reclist = []
        batch = []
        for conn in parent_conns:
            if conn.poll():
                reclist.append(conn)
                batch.append(conn.recv())
        if batch:
            batch = torch.from_numpy(np.array(batch)).float().to(net.device)
            p_t, v_t = net.forward(batch)
            p_t_list = p_t.tolist()
            v_t_list = v_t.tolist()
            for i in range(len(p_t_list)):
                reclist[i].send((p_t_list[i], v_t_list[i]))
    return


class ExampleGenerator:
    def __init__(self, net, **kwargs):
        self.kwargs = kwargs
        self.net = net
        self.examples = []
        self.gpu_queue = dict()
        self.ps = dict()
        self.v = dict()
        return


    def start_pool(self, n_games, game_fn, *args):
        is_done = multiprocessing.Value('i', 0)
        parent_conns = []
        child_conns = []
        for i in range(n_games):
            parent_conn, child_conn = multiprocessing.Pipe()
            parent_conns.append(parent_conn)
            child_conns.append(child_conn)
        time.sleep(2.0)
        pool = multiprocessing.Pool(processes=10, initializer=np.random.seed)
        gpu_handler = multiprocessing.Process(target=handle_gpu, args=(copy.deepcopy(self.net), parent_conns, is_done))
        gpu_handler.start()
        examples = pool.map_async(game_fn, child_conns)
        return [gpu_handler, pool, examples, is_done, child_conns, parent_conns]

    def run_games(self, n_games, game_fn):
        n_pools = 5
        pools = []
        examples = []
        for i in range(n_pools):
            pools.append(self.start_pool(int(n_games / n_pools), game_fn))
        for i in range(n_pools):
            examples.append(pools[i][2].get())
            pools[i][1].close()
            with pools[i][3].get_lock():
                pools[i][3].value = 1
            pools[i][0].join()
            pools[i][1].join()
        niceness = os.nice(0)
        os.nice(0 - niceness)
        return examples

    def generate_examples(self, n_games):
        """Creates threads with MCTSAgents who play one game each. They send neural network evaluation requests to the
        GPU_handler thread.

        @param n_games: amount of games to generate. Also is the amount of threads created
        @return:
        """

        examples_temp = self.run_games(n_games, generate_single_game)
        examples = [item for sublist in examples_temp for item in sublist]
        print("Generated " + str(len(examples)) + " games")
        return examples

    def generate_mcts_tests(self, n_games, game_fn):
        """Creates threads with MCTSAgents who play one game each. They send neural network evaluation requests to the
        GPU_handler thread.

        @param n_games: amount of games to generate. Also is the amount of threads created
        @return:
        """
        examples_temp = self.run_games(n_games, game_fn)
        examples = [item for sublist in examples_temp for item in sublist]
        avg_reward = sum(examples)/(2*n_games)
        return avg_reward


if __name__ == '__main__':
    use_gpu = True
    if use_gpu:
        if not torch.cuda.is_available():
            print("Tried to use GPU, but none is available")
            use_gpu = False
    device = torch.device("cuda:0" if use_gpu else "cpu")
    net = Net(device=device)
    net.to(device)

    net.eval()
    generator = ExampleGenerator(net)
    start = time.time()

    generator.generate_examples(500)
    print(time.time() - start)
