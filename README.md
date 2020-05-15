# alphazero-connect4
Implementation of an alphaZero like algorithm to play games from the openSpiel library such as breakthrough or connect-four.
To train a new network, run train.py

Overview of important files:
train.py: Main script to run to train an AlphaZero PV-network, also contains the trainer class that trains a network. Current configuration reproduces data for Figure 7.

network.py: ResNet implementation for board games.

examplegenerator.py: File deals with parallelization of games on multiple threads and GPU's.

alphazerobot.py: OpenSpiel bot implementation of an AlphaZero agent. Selects actions based on MCTS with some value and policy evaluator (a Neural Network)

game_utils.py: Contains the functions to play games. Note that here the difference between value targets is made.

mcts.py: Implementation of MCTS guided by a policy-value evaluator.

Here is a broad overview of the code structure.
![Code Structure](/documentation/code_structure.png)
Format: ![Code Structure](url)
