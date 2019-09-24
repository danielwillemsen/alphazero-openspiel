import multiprocessing
import time

import numpy as np
import pyspiel
import torch

from alphazerobot import AlphaZeroBot
from connect4net import Net
from game_utils import play_game_self
from state_to_board import state_to_board


def generate_single_game(conn):
    evaluator = Evaluator(conn)
    example = play_game_self(evaluator.evaluate_nn)
    return example


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


class ExampleGenerator:
    def __init__(self, net, **kwargs):
        self.kwargs = kwargs
        self.net = net
        self.examples = []
        self.gpu_queue = dict()
        self.ps = dict()
        self.v = dict()
        return

    def handle_gpu(self, parent_conns, is_done):
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
                batch = torch.from_numpy(np.array(batch)).float().to(self.net.device)
                p_t, v_t = self.net.forward(batch)
                p_t_list = p_t.tolist()
                v_t_list = v_t.tolist()
                for i in range(len(p_t_list)):
                    reclist[i].send((p_t_list[i], v_t_list[i]))
        return

    def start_pool(self,n_games):
        spawn_context = multiprocessing.get_context('spawn')
        is_done = spawn_context.Value('i', 0)
        parent_conns = []
        child_conns = []
        for i in range(n_games):
            parent_conn, child_conn = spawn_context.Pipe()
            parent_conns.append(parent_conn)
            child_conns.append(child_conn)
        pool = spawn_context.Pool(processes=16, initializer=np.random.seed)
        gpu_handler = spawn_context.Process(target=self.handle_gpu, args=(parent_conns, is_done))
        gpu_handler.start()
        examples = pool.map_async(generate_single_game, child_conns)
        return [gpu_handler, pool, examples, is_done, child_conns]

    def generate_examples(self, n_games):
        """Creates threads with MCTSAgents who play one game each. They send neural network evaluation requests to the
        GPU_handler thread.

        @param n_games: amount of games to generate. Also is the amount of threads created
        @return:
        """
        self.examples = []


        n_pools = 4
        pools = []
        examples = []
        for i in range(n_pools):
            pools.append(self.start_pool(int(n_games/n_pools)))
        for i in range(n_pools):
            examples.append(pools[i][2].get())
            with pools[i][3].get_lock():
                pools[i][3].value = 1
            pools[i][0].join()
            pools[i][1].close()
        self.examples = [item for sublist in examples for item in sublist]
        print("Generated " + str(len(self.examples)) + " games")
        return self.examples


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

    generator.generate_examples(10)
    print(time.time() - start)
