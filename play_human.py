import torch
from connect4net import Net
from mctsagent import MCTSAgent

if __name__ == '__main__':
	net = Net()
	net.load_state_dict(torch.load("models/initial105.pth", map_location='cpu'))
	net.eval()
	mctsagent = MCTSAgent(net.predict)
	mctsagent.play_game_vs_human_net_only()
