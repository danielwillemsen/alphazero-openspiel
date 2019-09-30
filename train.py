import pickle
import time
from datetime import datetime

import numpy as np
import pyspiel
import torch
import torch.nn as nn
from torch import multiprocessing
from open_spiel.python.algorithms import mcts

from alphazerobot import AlphaZeroBot, NeuralNetBot
from connect4net import Net
from examplegenerator import ExampleGenerator
from examplegenerator import test_net_game_vs_mcts100, test_net_game_vs_mcts200, test_zero_game_vs_mcts200
from game_utils import *
import logging


class Trainer:
    def __init__(self):
        # Experiment Parameters
        self.name_game = "connect_four"         # Name of game (should be from open_spiel library)
        self.name_run = "openspieltest"         # Name of run
        self.model_path = "models/"             # Path to save the models
        self.save = True                        # Save neural network
        self.save_n_gens = 5                    # How many iterations until network save
        self.test_n_gens = 5                    # How many iterations until testing
        self.n_tests = 4                        # How many tests to perform for testing
        self.use_gpu = True                     # Use GPU (if available)

        # Algorithm Parameters
        self.n_games_per_generation = 250         # How many games to generate per iteration
        self.n_batches_per_generation = 2000    # How batches of neural network training per iteration
        self.n_games_buffer = 20000             # How many games to store in FIFO buffer
        self.batch_size = 64                    # Batch size for neural network training
        self.lr = 0.0002                        # Learning rate for neural network

        # Initialization of the trainer
        self.game = pyspiel.load_game(self.name_game)
        self.buffer = []
        self.num_distinct_actions = self.game.num_distinct_actions()
        self.state_shape = self.game.information_state_normalized_vector_shape()
        self.game = pyspiel.load_game(self.name_game)
        self.games_played = 0
        self.start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.test_data = {'games_played': [], 'zero_vs_random': [], 'zero_vs_mcts100': [], 'zero_vs_mcts200': [],
                          'net_vs_random': [], 'net_vs_mcts100': [], 'net_vs_mcts200': []}

        # Setup CUDA if possible
        if self.use_gpu:
            if not torch.cuda.is_available():
                logger.info("Tried to use GPU, but none is available")
                self.use_gpu = False
        self.device = torch.device("cuda:0" if self.use_gpu else "cpu")

        # Initialize neural net
        self.current_net = Net(self.state_shape, self.num_distinct_actions, device=self.device)
        self.current_net.to(self.device)
        self.optimizer = torch.optim.Adam(self.current_net.parameters(), lr=self.lr, weight_decay=0.0001)
        self.criterion_policy = nn.BCELoss()
        self.criterion_value = nn.MSELoss()
        self.current_net.eval()

        # Initialize logger
        formatter = logging.Formatter('%(asctime)s %(message)s')
        logger.setLevel('DEBUG')
        fh = logging.FileHandler("logs/" + str(self.start_time) + str(self.name_run) + ".log")
        fh.setFormatter(formatter)
        fh.setLevel('INFO')
        ch = logging.StreamHandler()
        ch.setLevel('INFO')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.addHandler(fh)
        logger.info('Logger started')
        logger.info(self.__dict__)

    def net_step(self, flattened_buffer):
        """Samples a random batch and updates the NN parameters with this bat

        @return:
        """
        self.current_net.zero_grad()

        # Select samples and format them to use as batch
        sample_ids = np.random.randint(len(flattened_buffer), size=self.batch_size)
        x = [flattened_buffer[i][1] for i in sample_ids]
        p_r = [flattened_buffer[i][2] for i in sample_ids]
        v_r = [flattened_buffer[i][3] for i in sample_ids]

        x = torch.from_numpy(np.array(x)).float().to(self.device)
        p_r = torch.tensor(np.array(p_r)).float().to(self.device)
        v_r = torch.tensor(np.array(v_r)).float().to(self.device)

        # Pass through network
        p_t, v_t = self.current_net(x)

        # Backward pass
        loss_v = self.criterion_value(v_t, v_r.unsqueeze(1))
        loss_p = -torch.sum(p_r * torch.log(p_t)) / p_r.size()[0]

        # loss_p = self.criterion_policy(p_t, p_r)
        loss = loss_v + loss_p
        loss.backward()
        self.optimizer.step()
        return loss_p, loss_v

    def train_network(self, n_batches):
        """Trains the neural network for batches_per_generation batches

        @return:
        """
        logger.info("Training Network")
        self.current_net.train()
        flattened_buffer = [sample for game in self.buffer for sample in game]
        flattened_buffer = self.remove_duplicates(flattened_buffer)
        loss_tot_v = 0
        loss_tot_p = 0

        for i in range(n_batches):
            loss_p, loss_v = self.net_step(flattened_buffer)
            loss_tot_p += loss_p
            self.v = loss_v
            loss_tot_v += self.v
            if i % 200 == 0:
                logger.info("Batch: " + str(i) + "Loss policy: " + str(loss_tot_p / 200.) + "Loss value: " + str(
                    loss_tot_v / 200.))
                loss_tot_v = 0
                loss_tot_p = 0
        self.current_net.eval()

    @staticmethod
    def remove_duplicates(flattened_buffer):
        logger.info("Removing duplciates")
        logger.info("Initial amount of samples: " + str(len(flattened_buffer)))
        start = time.time()
        # Remove duplicates
        flattened_buffer_dict = dict()
        flattened_buffer_counts = dict()
        for item in flattened_buffer:
            if item[0] in flattened_buffer_dict:
                # Average policy
                flattened_buffer_dict[item[0]][2] = [sum(x) for x in zip(flattened_buffer_dict[item[0]][2], item[2])]
                # Average value
                flattened_buffer_dict[item[0]][3] += item[3]
                flattened_buffer_counts[item[0]] += 1

            else:
                flattened_buffer_dict[item[0]] = item
                flattened_buffer_counts[item[0]] = 1
        for key, value in flattened_buffer_dict.items():
            flattened_buffer_dict[key][2] = [x / flattened_buffer_counts[key] for x in flattened_buffer_dict[key][2]]
            flattened_buffer_dict[key][3] = flattened_buffer_dict[key][3] / flattened_buffer_counts[key]
        flattened_buffer = list(flattened_buffer_dict.values())
        logger.info("New amount of samples: " + str(len(flattened_buffer)))
        logger.info("Duplication removal took:" + str(time.time() - start) + "seconds")
        return flattened_buffer

    def generate_examples(self, n_games):
        """Generates new games in a multithreaded way.

        @param n_games: amount of games to generate
        """
        logger.info("Generating Data")
        # Generate new training samples
        # start = time.time()
        # for i in range(n_games):
        #     logger.info("Game " + str(i) + " / " + str(n_games))
        #     examples = play_game_self(self.current_net.predict)
        #     self.buffer.append(examples)
        # logger.info("Finished Generating Data (normal)")
        # logger.info(time.time()-start)

        start = time.time()

        # Generate the examples
        generator = ExampleGenerator(self.current_net, self.name_game,
                                     self.device)
        games = generator.generate_examples(n_games)
        self.games_played += self.n_games_per_generation

        # Add examples to buffer
        for examples in games:
            self.buffer.append(examples)
        logger.info("Finished Generating Data (threaded). Took: " + str(time.time() - start) + " seconds")

        # Remove oldest entries from buffer if too long
        if len(self.buffer) > self.n_games_buffer:
            logger.info("Buffer full. Deleting oldest samples.")
            while len(self.buffer) > self.n_games_buffer:
                del self.buffer[0]

    def test_agent(self):
        """Tests the current agent
        Tests against random opponents and pure MCTS agents

        @return: None
        """
        start = time.time()
        logger.info("Testing...")
        generator = ExampleGenerator(self.current_net, self.name_game,
                                     self.device)
        self.test_data['games_played'].append(self.games_played)
        # score_tot = 0.
        # for i in range(self.n_tests):
        #     score1, score2 = test_zero_vs_random(self.current_net.predict, self.game_name)
        #     score_tot += score1
        #     score_tot += score2
        # avg = score_tot / (2 * self.n_tests)
        # self.test_data['zero_vs_random'].append(avg)
        # logger.info("Average score vs random:" + str(avg))
        score_tot = 0.
        for i in range(self.n_tests):
            score1, score2 = test_net_vs_random(self.current_net.predict, self.name_game)
            score_tot += score1
            score_tot += score2
        avg = score_tot / (2 * self.n_tests)
        self.test_data['net_vs_random'].append(avg)
        logger.info("Average score vs random (net only):" + str(avg))
        score_tot = 0.
        for i in range(self.n_tests):
            score1, score2 = test_zero_vs_mcts(self.current_net.predict, 100, self.name_game)
            score_tot += score1
            score_tot += score2
        avg = score_tot / (2 * self.n_tests)
        self.test_data['zero_vs_mcts100'].append(avg)
        logger.info("Average score vs mcts100:" + str(avg))

        avg = generator.generate_mcts_tests(self.n_tests, test_net_game_vs_mcts100)
        self.test_data['net_vs_mcts100'].append(avg)
        logger.info("Average score vs mcts100 (net only):" + str(avg))

        avg = generator.generate_mcts_tests(self.n_tests, test_zero_game_vs_mcts200)
        self.test_data['zero_vs_mcts200'].append(avg)
        logger.info("Average score vs mcts200:" + str(avg))

        avg = generator.generate_mcts_tests(self.n_tests, test_net_game_vs_mcts200)
        self.test_data['net_vs_mcts200'].append(avg)
        logger.info("Average score vs mcts200 (net only):" + str(avg))
        with open("logs/" + self.start_time + str(self.name_run) + ".p", 'wb') as f:
            pickle.dump(self.test_data, f)
        logger.info("Testing took: " + str(time.time() - start) + "seconds")
        return

    def run(self):
        """Main alphaZero training loop
        @return:
        """
        self.test_agent()         # Start with testing the agent
        generation = 0
        while True:
            generation += 1
            logger.info("Generation:" + str(generation))
            self.generate_examples(self.n_games_per_generation)         # Generate new games through self-play
            self.train_network(self.n_batches_per_generation)           # Train network on games in the buffer

            # Perform testing periodically
            if generation % self.test_n_gens == 0:                      # Test the alphaZero bot against MCTS bots
                self.test_agent()

            # Periodically save network
            if self.save and generation % self.save_n_gens == 0:
                logger.info("Saving network")
                torch.save(self.current_net.state_dict(), self.model_path + self.name_run + str(generation) + ".pth")
                logger.info("Network saved")


if __name__ == '__main__':
    logger = logging.getLogger('alphazero')
    multiprocessing.set_start_method('spawn')
    trainer = Trainer()
    trainer.run()
