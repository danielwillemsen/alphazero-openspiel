with open('../meteor01/alphazero-connect4/logs/2020-01-13-14-56-41on-policytemp0.5.log', 'r') as f:
    for line in f:
        if "name_game" in line:
            temp = line[65:90]
        if "mcts5000" in line:
            temp = temp + "   ---   " + line[40:]
            print(temp)