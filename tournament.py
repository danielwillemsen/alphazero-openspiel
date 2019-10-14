from game_utils import *
import logging
from train import Trainer
from torch import multiprocessing
import torch
from examplegenerator import ExampleGenerator
import copy
from connect4net import Net

if __name__ == '__main__':
    logger = logging.getLogger('alphazero')
    multiprocessing.set_start_method('spawn')
    trainer = Trainer()
    trainer.current_net.load_state_dict(torch.load('./breakthrough6x6_400.pth', map_location=trainer.device))
    trainer.current_net.load_state_dict(torch.load('./breakthrough6x6_400.pth', map_location=trainer.device))

    net50 = Net(trainer.state_shape, trainer.num_distinct_actions)
    net50.load_state_dict(torch.load('./breakthrough6x6_50.pth', map_location=trainer.device))

    net100 = Net(trainer.state_shape, trainer.num_distinct_actions)
    net100.load_state_dict(torch.load('../meteor01_09-okt_breakthrough/alphazero-connect4/models/openspieltest100.pth', map_location=trainer.device))

    net200 = Net(trainer.state_shape, trainer.num_distinct_actions)
    net200.load_state_dict(torch.load('../meteor01_09-okt_breakthrough/alphazero-connect4/models/openspieltest200.pth', map_location=trainer.device))

    n_tests = 500
    logger.info("n_tests: " + str(n_tests))

    agents = []
    agents.append(["AlphaZero net 50", {"n_playouts": 100, "use_probabilistic_actions": True}, net50])
    agents.append(["AlphaZero net 100", {"n_playouts": 100, "use_probabilistic_actions": True}, net100])
    agents.append(["AlphaZero net 200", {"n_playouts": 100, "use_probabilistic_actions": True}, net200])
    agents.append(["AlphaZero net 400", {"n_playouts": 100, "use_probabilistic_actions": True}, trainer.current_net])

    #agents.append(["AlphaZero 200 simulations", {"n_playouts": 200, "use_probabilistic_actions": True}])
    #agents.append(["AlphaZero 400 simulations", {"n_playouts": 400, "use_probabilistic_actions": True}])
    #agents.append(["AlphaZero 800 simulations", {"n_playouts": 800, "use_probabilistic_actions": True}])
    #agents.append(["AlphaZero 1600 simulations", {"n_playouts": 1600, "use_probabilistic_actions": True}])

    logger.info(str(agents))
    generator = ExampleGenerator(trainer.current_net, trainer.name_game,
                                    trainer.device, is_test=True)
    results = np.zeros((len(agents), len(agents)))
    for index1, agent1 in enumerate(agents):
        for index2, agent2 in enumerate(agents):
            if len(agent1) > 2:
                generator.net = copy.deepcopy(agent1[2])
                generator.net2 = copy.deepcopy(agent2[2])
                generator.net.to('cpu')
                generator.net2.to('cpu')
            generator.kwargs = {"settings1": agent1[1], "settings2": agent2[1]}
            avg_reward = generator.generate_tests(n_tests, test_zero_vs_zero, None)
            results[index1, index2] = avg_reward
            logger.info(agent1[0] + " vs " + agent2[0] + ": " + str(avg_reward*0.5+0.5))
    logger.info(results)
    logger.info(results*0.5 + 0.5)
