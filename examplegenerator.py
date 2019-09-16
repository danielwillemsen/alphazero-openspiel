import multiprocessing
from mctsagent import MCTSAgent
from connect4net import Net
import torch
import numpy as np
import time
from alphazerobot import AlphaZeroBot
import pyspiel
from multiprocessing import Process, Pipe, Value, JoinableQueue, Queue
import multiprocessing

class Evaluator():
    def __init__(self, net, conn):
        self.conn = conn
        self.net = net

    def evaluate_nn(self, state):
        """

        @param state: The game which needs to be evaluated
        @return: pi: policy according to neural net, vi: value according to neural net.
        """
        board = self.net.state_to_board(state)
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

    def generate_game(self, conn, examples_queue, finished_indicator):
        evaluator = Evaluator(self.net, conn)
        example = self.play_game_self(evaluator.evaluate_nn)
        examples_queue.put(example)
        with finished_indicator.get_lock():
            finished_indicator.value = finished_indicator.value + 1
        return

    @staticmethod
    def play_game_self(policy_fn):
        examples = []
        game = pyspiel.load_game('connect_four')
        state = game.new_initial_state()
        alphazero_bot = AlphaZeroBot(game, 0, policy_fn, self_play=True)
        while not state.is_terminal():
            policy, action = alphazero_bot.step(state)
            policy_dict = dict(policy)
            policy_list = []
            for i in range(7):
                # Create a policy list. To be used in the net instead of a list of tuples.
                policy_list.append(policy_dict.get(i, 0.0))

            if sum(policy_list) > 0.0:
                # Normalize policy after taking out illegal moves
                policy_list = [item / sum(policy_list) for item in policy_list]
            else:
                policy_list = [1.0 / len(policy_list) for item in policy_list]

            examples.append([state.information_state(), Net.state_to_board(state), policy_list, None])
            state.apply_action(action)
        # Get return for starting player
        reward = state.returns()[0]
        for i in range(len(examples)):
            examples[i][3] = reward
            reward *= -1
        return examples

    def generate_examples(self, n_games):
        """Creates threads with MCTSAgents who play one game each. They send neural network evaluation requests to the
        GPU_handler thread.

        @param n_games: amount of games to generate. Also is the amount of threads created
        @return:
        """
        self.examples = []

        examples_queue = JoinableQueue(0)
        games = []
        parent_conns = []
        child_conns = []
        is_done = Value('i', 0)
        finished_indicator = Value('i', 0)
        for i in range(n_games):
            parent_conn, child_conn = Pipe()
            parent_conns.append(parent_conn)
            child_conns.append(child_conn)
            games.append(Process(target=self.generate_game, args=(child_conn, examples_queue, finished_indicator)))
        gpu_handler = Process(target=self.handle_gpu, args=(parent_conns, is_done))
        gpu_handler.start()
        for game in games:
            game.start()
        finished = False
        while not finished:
            with finished_indicator.get_lock():
                if finished_indicator.value == n_games:
                    finished = True
        while len(self.examples) < n_games:
            self.examples.append(examples_queue.get(timeout=0.1))
            examples_queue.task_done()
        examples_queue.join()
        for game in games:
            game.join()
        with is_done.get_lock():
            is_done.value = 1
        gpu_handler.join()
        print("Generated " + str(len(self.examples)) + " games")
        return self.examples


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

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

    generator.generate_examples(30)
    print(time.time() - start)
