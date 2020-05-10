from torch import multiprocessing
import time

import numpy as np
import pyspiel
import torch
import os
from alphazerobot import AlphaZeroBot
from connect4net import Net
from game_utils import play_game_self, test_zero_vs_mcts, test_net_vs_mcts
from connect4net import state_to_board
import copy
import logging
logger = logging.getLogger('alphazero')


def generate_single_game(tup):
    conn, game_name, kwargs, _, _, _ = tup
    game = pyspiel.load_game(game_name)
    evaluator = Evaluator(conn, game)
    example = play_game_self(evaluator.evaluate_nn, game_name, **kwargs)
    return example


def test_single_game(tup):
    conn, game_name, kwargs, game_fn, conn2, _ = tup
    n_playouts_mcts = tup[5][0]
    game = pyspiel.load_game(game_name)
    evaluator = Evaluator(conn, game)
    if conn2:
        evaluator2 = Evaluator(conn2, game)
        score1, score2, statistics = game_fn(evaluator.evaluate_nn, n_playouts_mcts, game_name, policy_fn2=evaluator2.evaluate_nn, **kwargs)
    else:
        score1, score2, statistics = game_fn(evaluator.evaluate_nn, n_playouts_mcts, game_name, **kwargs)

    return score1 + score2, statistics


class Evaluator():
    def __init__(self, conn, game):
        self.conn = conn
        self.state_shape = game.information_state_normalized_vector_shape()

    def evaluate_nn(self, state):
        """

        @param state: The game which needs to be evaluated
        @return: pi: policy according to neural net, vi: value according to neural net.
        """
        board = state_to_board(state, self.state_shape)
        self.conn.send(board)
        pi, vi = self.conn.recv()
        vi = float(vi[0])
        return pi, vi


def handle_gpu(net, parent_conns, device):
    """Thread which continuously pushes items from the gpu_queue to the gpu. This results in multiple games being
    evaluated in a single batch

    @return:
    """
    net.to(device)
    while True:
        reclist = []
        batch = []
        for conn in parent_conns:
            if conn.poll():
                reclist.append(conn)
                batch.append(conn.recv())
        if batch:
            batch = torch.from_numpy(np.array(batch)).float().to(device)
            p_t, v_t = net.forward(batch)
            p_t_list = p_t.tolist()
            v_t_list = v_t.tolist()
            for i in range(len(p_t_list)):
                reclist[i].send((p_t_list[i], v_t_list[i]))


class ExampleGenerator:
    def __init__(self, net, game_name, device, **kwargs):
        self.is_test = bool(kwargs.get("is_test", False))
        self.device_count = torch.cuda.device_count()
        self.net = copy.deepcopy(net)
        self.net.to("cpu")
        self.net2 = copy.deepcopy(kwargs.get('net2', None))
        if self.net2:
            self.net2.to("cpu")
        self.generate_statistics = kwargs.get("generate_statistics", False)
        self.device = device
        self.game_name = game_name
        self.kwargs = kwargs
        self.examples = []
        self.gpu_queue = dict()
        self.ps = dict()
        self.v = dict()

        if self.is_test:
            self.single_game_fn = test_single_game
        else:
            self.single_game_fn = generate_single_game
        return

    def start_pool(self, n_games, game_fn, device, *args):
        parent_conns = []
        child_conns = []
        parent_conns2 = []
        child_conns2 = []
        for i in range(n_games):
            parent_conn, child_conn = multiprocessing.Pipe()
            parent_conns.append(parent_conn)
            child_conns.append(child_conn)
        if self.net2:
            for i in range(n_games):
                parent_conn2, child_conn2 = multiprocessing.Pipe()
                parent_conns2.append(parent_conn2)
                child_conns2.append(child_conn2)
        pool = multiprocessing.Pool(processes=1, initializer=np.random.seed)
        gpu_handler = multiprocessing.Process(target=handle_gpu, args=(copy.deepcopy(self.net), parent_conns, device))
        gpu_handler2 = None
        if self.net2:
            gpu_handler2 = multiprocessing.Process(target=handle_gpu,
                                                  args=(copy.deepcopy(self.net2), parent_conns2, device))
            gpu_handler2.start()

        gpu_handler.start()
        if self.net2:
            examples = pool.map_async(self.single_game_fn,
                                      [(conn, self.game_name, self.kwargs, game_fn, child_conns2[idx], args)
                                       for idx, conn in enumerate(child_conns)])
        else:
            examples = pool.map_async(self.single_game_fn,
                                      [(conn, self.game_name, self.kwargs, game_fn, None, args)
                                       for conn in child_conns])

        return [gpu_handler, pool, examples, child_conns, parent_conns, gpu_handler2, child_conns2, parent_conns2]

    def run_games(self, n_games, game_fn, *args):
        n_pools = 1
        pools = []
        examples = []
        device_no = 1
        for i in range(n_pools):
            if device_no>=self.device_count:
                device_no = 0
            device = torch.device("cuda:" + str(device_no) if self.device_count >= 1 else "cpu")
            pools.append(self.start_pool(int(n_games / n_pools), game_fn, device, *args))
            device_no += 1
        for i in range(n_pools):
            examples.append(pools[i][2].get())
            pools[i][1].close()
            pools[i][1].join()
            pools[i][0].terminate()
            pools[i][0].join()
            if self.net2:
                pools[i][5].terminate()
                pools[i][5].join()
        niceness = os.nice(0)
        os.nice(0 - niceness)
        return examples

    def generate_examples(self, n_games):
        """Creates threads with MCTSAgents who play one game each. They send neural network evaluation requests to the
        GPU_handler thread.

        @param n_games: amount of games to generate. Also is the amount of threads created
        @return:
        """

        examples_temp = self.run_games(n_games, play_game_self)
        examples = [item for sublist in examples_temp for item in sublist]
        logger.info("Generated " + str(len(examples)) + " games")
        return examples

    def generate_tests(self, n_games, game_fn, n_playouts_mcts):
        """Creates threads with alphaZero agents or NeuralNet agens who play one game against an MCTS bot each. 
        They send neural network evaluation requests to the
        GPU_handler thread.

        @param n_games: amount of games to generate. Also is the amount of threads created
        @param game_fn: which game function to use
        @param n_playouts_mcts: amount of playouts for the MCTS agent
        @return:
        """
        examples_temp = self.run_games(n_games, game_fn, n_playouts_mcts)
        examples = [item[0] for sublist in examples_temp for item in sublist]
        statistics = [item[1] for sublist in examples_temp for item in sublist]

        avg_reward = sum(examples)/(2*n_games)
        if self.generate_statistics:
            return avg_reward, statistics
        else:
            return avg_reward


if __name__ == '__main__':
    use_gpu = True
    if use_gpu:
        if not torch.cuda.is_available():
            logger.info("Tried to use GPU, but none is available")
            use_gpu = False
    device = torch.device("cuda:0" if use_gpu else "cpu")
    net = Net(device=device)
    net.to(device)

    net.eval()
    generator = ExampleGenerator(net)
    start = time.time()

    generator.generate_examples(500)
    logger.info(time.time() - start)
