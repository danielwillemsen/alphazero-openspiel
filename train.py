import sys
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
from game_utils import *
import logging

logger = logging.getLogger('alphazero')


class Trainer:
    def __init__(self, name="openspieltest", backup="on-policy"):
        # Experiment Parameters
        self.name_game = "connect_four"#"breakthrough(rows=6,columns=6)"         # Name of game (should be from open_spiel library)
        self.name_run = name         # Name of run
        self.model_path = "models/"             # Path to save the models
        self.save = True                        # Save neural network
        self.save_n_gens = 10                   # How many iterations until network save
        self.test_n_gens = 10                   # How many iterations until testing
        self.n_tests = 200                      # How many tests to perform for testing
        self.use_gpu = True                     # Use GPU (if available)

        # Algorithm Parameters
        self.n_games_per_generation = 500        # How many games to generate per iteration
        self.n_batches_per_generation = 500      # How batches of neural network training per iteration
        self.n_games_buffer_max = 20000          # How many games to store in FIFO buffer, at most. Buffer is grown.
        self.batch_size = 256                   # Batch size for neural network training
        self.lr = 0.001                         # Learning rate for neural network
        self.n_games_buffer = 4 * self.n_games_per_generation
        self.temperature = 1.0
        self.dirichlet_ratio = 0.25
        self.uct_train = 2.5
        self.uct_test = 2.5
        self.n_playouts_train = 100
        self.backup = backup
        self.tree_strap = False
        self.it = 0

        # Initialization of the trainer
        self.generation = 0
        self.game = pyspiel.load_game(self.name_game)
        self.buffer = []
        self.num_distinct_actions = self.game.num_distinct_actions()
        self.state_shape = self.game.information_state_normalized_vector_shape()
        self.game = pyspiel.load_game(self.name_game)
        self.games_played = 0
        self.start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.test_data = {'games_played': [], 'zero_vs_random': [], 'zero_vs_mcts100': [], 'zero_vs_mcts200': [],
                          'net_vs_random': [], 'net_vs_mcts100': [], 'net_vs_mcts200': []}
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
        logger.info(str(torch.cuda.is_available()))
        logger.info(str(backup))
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


        logger.info(self.__dict__)
        logger.info("Using:" + str(self.device))

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

        # Pass through network
        p_t, v_t = self.current_net(x)
        p_r = [item if item else p_t[i,:].to("cpu").tolist() for i, item in enumerate(p_r)]

        p_r = torch.tensor(np.array(p_r)).float().to(self.device)
        v_r = torch.tensor(np.array(v_r)).float().to(self.device)


        # Backward pass
        loss_v = self.criterion_value(v_t, v_r.unsqueeze(1))
        loss_p = -torch.sum(p_r * torch.log(p_t)) / p_r.size()[0]
        loss = loss_v + loss_p
        loss.backward()
        self.optimizer.step()
        self.it += 1
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
            v = loss_v
            loss_tot_v += v
            if i % 100 == 99:
                logger.info("Batch: " + str(i) + "Loss policy: " + str(loss_tot_p / 100.) + "Loss value: " + str(
                    loss_tot_v / 100.))
                loss_tot_v = 0
                loss_tot_p = 0
        self.current_net.eval()

    def remove_duplicates(self, flattened_buffer):
        logger.info("Removing duplciates")
        logger.info("Initial amount of samples: " + str(len(flattened_buffer)))
        start = time.time()
        # Remove duplicates
        flattened_buffer_dict = dict()
        flattened_buffer_counts = dict()
        flattened_buffer_pol = dict()

        for item in flattened_buffer:
            if item[0] in flattened_buffer_dict:
                # Average policy
                if item[2] and flattened_buffer_dict[item[0]][2]:
                    flattened_buffer_dict[item[0]][2] = [sum(x) for x in zip(flattened_buffer_dict[item[0]][2], item[2])]
                    flattened_buffer_pol[item[0]] += 1
                elif item[2]:
                    flattened_buffer_dict[item[0]][2] = item[2]

                # Average value
                flattened_buffer_dict[item[0]][3] += item[3]
                flattened_buffer_counts[item[0]] += 1

            else:
                flattened_buffer_dict[item[0]] = item
                flattened_buffer_counts[item[0]] = 1
                flattened_buffer_pol[item[0]] = 1

        for key, value in flattened_buffer_dict.items():
            if flattened_buffer_dict[key][2]:
                flattened_buffer_dict[key][2] = [x / flattened_buffer_pol[key] for x in flattened_buffer_dict[key][2]]
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
        start = time.time()

        # Generate the examples
        generator = ExampleGenerator(self.current_net, self.name_game,
                                     self.device, n_playouts=self.n_playouts_train, temperature=self.temperature,
                                     dirichlet_ratio=self.dirichlet_ratio, c_puct=self.uct_train,
                                     backup=self.backup, tree_strap=self.tree_strap)
        games = generator.generate_examples(n_games)
        self.games_played += self.n_games_per_generation

        # Add examples to buffer
        for examples in games:
            self.buffer.append(examples)
        logger.info("Finished Generating Data (threaded). Took: " + str(time.time() - start) + " seconds")
        logger.info("Total amount of games played:" + str(self.generation*self.n_games_per_generation))
        self.update_buffer_size()

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
                                     self.device, is_test=True, temperature=self.temperature,
                                     dirichlet_ratio=self.dirichlet_ratio, c_puct=self.uct_test)
        self.test_data['games_played'].append(self.games_played)

        score_tot = 0.
        for i in range(self.n_tests):
            score1, score2 = test_net_vs_random(self.current_net.predict, self.name_game)
            score_tot += score1
            score_tot += score2
        avg = score_tot / (2 * self.n_tests)
        self.test_data['net_vs_random'].append(avg)
        logger.info("Average score vs random (net only):" + str(avg))

        avg = generator.generate_tests(self.n_tests, test_net_vs_mcts, 100)
        self.test_data['net_vs_mcts100'].append(avg)
        logger.info("Average score vs mcts100 (net only):" + str(avg))

        avg = generator.generate_tests(self.n_tests, test_zero_vs_mcts, 200)
        self.test_data['zero_vs_mcts200'].append(avg)
        logger.info("Average score vs mcts200:" + str(avg))

        avg = generator.generate_tests(self.n_tests, test_net_vs_mcts, 200)
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

        # Start with testing the agent
        self.test_agent()

        while self.generation < 201:
            self.generation += 1
            logger.info("Generation:" + str(self.generation))
            self.generate_examples(self.n_games_per_generation)         # Generate new games through self-play
            self.train_network(self.n_batches_per_generation)           # Train network on games in the buffer

            # Perform testing periodically
            if self.generation % self.test_n_gens == 0 or self.generation in [5, 10, 20, 30, 40, 50]:
                self.test_agent()

            # Periodically save network
            if self.save and self.generation % self.save_n_gens == 0 or self.generation in [5, 10, 20, 30, 40, 50]:
                logger.info("Saving network")
                torch.save(self.current_net.state_dict(), self.model_path + self.name_run + str(self.generation) + ".pth")
                logger.info("Network saved")

    def update_buffer_size(self):
        if self.generation % 1 == 0 and self.n_games_buffer < self.n_games_buffer_max:
            self.n_games_buffer += 200 #self.n_games_per_generation / 5
        logger.info("Buffer size:" + str(self.n_games_buffer))

if __name__ == '__main__':
    logger = logging.getLogger('alphazero')
    multiprocessing.set_start_method('spawn')

    backup_name = "on-policy"
    trainer = Trainer(name=backup_name,backup=backup_name)
    trainer.temperature = 1.0
    trainer.uct_train = 2.5
    trainer.dirichlet_ratio = 0.25
    trainer.run()

    backup_name = "soft-Z"
    trainer = Trainer(name=backup_name,backup=backup_name)
    trainer.temperature = 1.0
    trainer.uct_train = 2.5
    trainer.dirichlet_ratio = 0.25
    trainer.run()

    backup_name = "A0C"
    trainer = Trainer(name=backup_name,backup=backup_name)
    trainer.temperature = 1.0
    trainer.uct_train = 2.5
    trainer.dirichlet_ratio = 0.25
    trainer.run()

    backup_name = "off-policy"
    trainer = Trainer(name=backup_name,backup=backup_name)
    trainer.temperature = 1.0
    trainer.uct_train = 2.5
    trainer.dirichlet_ratio = 0.25
    trainer.run()