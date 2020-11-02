import pyspiel


def get_move_dict():
    with open("tictactoe.txt", 'r') as f:
        lines = [line for line in f.readlines()]
    abcdict = {"a": 0, "b": 1, "c": 2}
    state_dict = {}
    for i, line in enumerate(lines):
        if i % 7 == 0:
            l1 = line.replace(" ", "").lower()
            l2 = lines[i + 1].replace(" ", "").lower()
            l3 = lines[i + 2].replace(" ", "").lower()[:-1]
            state_name = l1 + l2 + l3
            move_line = lines[i + 5]
            state_dict[state_name] = {}  # Dictionary with move-value combinations
            equal_locations = [j for j in range(len(move_line)) if move_line.startswith("=", j)]
            for loc in equal_locations:
                move_name = abcdict[move_line[loc - 5]] + 3 * (3 - int(move_line[loc - 4]))
                move_val = float(move_line[loc + 1:loc + 4])
                state_dict[state_name][move_name] = move_val*2-1.0
    return state_dict


#           move_lin
# game = pyspiel.load_game("tic_tac_toe")
# length = 50
# n_games = 500
# #num_distinct_actions = 4
# state_shape = game.observation_tensor_shape()
# #tables = [tab for tab in pvtables.values()]
# #use_table = pvtables["off-policy"]
# #use_table.tables = tables
# state = game.new_initial_state()
# state.apply_action(2)
# state.apply_action(5)
# state.apply_action(6)
# state.apply_action(8)
# state.apply_action(7)
#
# print(str(state))
# print(str(state) in state_dict.keys())
# print(state_dict[str(state)])