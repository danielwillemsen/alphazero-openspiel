import sys

from game_utils import *
import logging
from train import Trainer
from torch import multiprocessing
import torch
from examplegenerator import ExampleGenerator
import copy
from network import Net

if __name__ == '__main__':
    print(sys.path)
    logger = logging.getLogger('alphazero')
    multiprocessing.set_start_method('spawn')
    trainer = Trainer()
    trainer.current_net.load_state_dict(torch.load('../../meteor01/alphazero-connect4/models/soft-Z5.pth', map_location=trainer.device))

    test_zero_vs_human(trainer.current_net.predict)
