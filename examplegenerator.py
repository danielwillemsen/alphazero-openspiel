import multiprocessing
from mctsagent import MCTSAgent
from connect4net import Net
import threading




class ExampleGenerator:
	def __init__(self, net, **kwargs):
		self.kwargs = kwargs
		self.net = net
		self.examples = []
		self.examples_lock = threading.Lock()
		return

	def evaluate_nn(self):
		return self.net.predict

	def generate_game(self):
		mcts = MCTSAgent(self.net.predict, **self.kwargs)
		example = mcts.play_game_self()
		with self.examples_lock:
			self.examples.append(example)
		return

	def generate_examples(self, n_games):
		self.examples = []
		games = [
			threading.Thread(target=self.generate_game, )
			for i in range(n_games)
		]
		for game in games:
			game.start()
		for game in games:
			game.join()
		return self.examples


if __name__ == '__main__':
	net = Net()
	generator = ExampleGenerator(net)
	generator.generate_examples(10)