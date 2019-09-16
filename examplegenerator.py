import multiprocessing
from mctsagent import MCTSAgent
from connect4net import Net
import threading
import torch
import numpy as np
import time
from alphazerobot import AlphaZeroBot
import pyspiel


class ExampleGenerator:
    def __init__(self, net, **kwargs):
        self.kwargs = kwargs
        self.net = net
        self.examples = []
        self.examples_lock = threading.Lock()
        self.gpu_queue = dict()
        self.gpu_lock = threading.Lock()
        self.gpu_done = threading.Event()
        self.all_games_finished = threading.Event()
        self.ps = dict()
        self.v = dict()
        return

    def evaluate_nn(self, state):
        """

        @param state: The game which needs to be evaluated
        @return: pi: policy according to neural net, vi: value according to neural net.
        """
        board = self.net.state_to_board(state)
        id = hash(board.tostring())
        with self.gpu_lock:
            self.gpu_queue[id] = board

        # Wait until nn is ready
        pi = None
        vi = None
        # Get correct item
        while pi is None or vi is None:
            self.gpu_done.wait()
            try:
                with self.gpu_lock:
                    pi = self.ps[id]
                    vi = float(self.v[id][0])
            except KeyError:
                pass
        return pi, vi

    def handle_gpu(self):
        """Thread which continuously pushes items from the gpu_queue to the gpu. This results in multiple games being
        evaluated in a single batch

        @return:
        """
        while not self.all_games_finished.isSet():
            with self.gpu_lock:
                if len(self.gpu_queue) > 0:
                    keys = list(self.gpu_queue.keys())
                    batch = list(self.gpu_queue.values())
                    batch = torch.from_numpy(np.array(batch)).float().to(self.net.device)
                    p_t, v_t = self.net.forward(batch)
                    p_t_list = p_t.tolist()
                    v_t_list = v_t.tolist()
                    for i in range(len(p_t_list)):
                        self.ps[keys[i]] = p_t_list[i]
                        self.v[keys[i]] = v_t_list[i]
                    self.gpu_done.set()
                    self.gpu_queue = dict()
        return

    def generate_game(self):
        example = self.play_game_self(self.evaluate_nn)
        with self.examples_lock:
            self.examples.append(example)
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
        self.gpu_queue = dict()
        self.ps = dict()
        self.v = dict()
        games = [
            threading.Thread(target=self.generate_game, )
            for i in range(n_games)
        ]
        self.all_games_finished.clear()
        gpu_handler = threading.Thread(target=self.handle_gpu)
        for game in games:
            game.start()
        gpu_handler.start()
        for game in games:
            game.join()
        self.all_games_finished.set()
        gpu_handler.join()
        return self.examples


if __name__ == '__main__':
    net = Net()
    net.eval()
    generator = ExampleGenerator(net)
    start = time.time()
    generator.generate_examples(1)
    print(time.time() - start)
