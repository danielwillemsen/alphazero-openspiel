import copy

class ToyGame:
    def __init__(self, length):
        self.length = length
        return

    def num_distinct_actions(self):
        return 4

    def new_initial_state(self):
        return State(0, self.length)

    def information_state_normalized_vector_shape(self):
        return (1,)

class State:
    def __init__(self, location1, length):
        self.location1 = location1
        self.length = length
        self.history_list = []
        self.terminal = False
        self.reward = 0

    def is_terminal(self):
        return self.terminal

    def current_player(self):
        if len(self.history_list) % 2 == 0:
            return 0
        else:
            return 1
    def history(self):
        return self.history_list

    def legal_actions(self, player):
        if self.current_player()==0:
            return [0, 1, 2]
        else:
            return [0]

    def information_state(self):
        return str(self.location1)

    def apply_action(self, action):
        if self.current_player() == 0:
            if len(self.history()) > self.length * 5:
                self.terminal = True
                self.reward = 0.
                return
            if action == 0:
                self.reward = -1.
                self.terminal = True
            if action == 1:
                self.location1 += 1
                if self.location1 == self.length:
                    self.reward = 0.1
                    self.terminal = True
            if action == 2:
                self.reward = 0.
                self.terminal = True
            if action == 3:
                if self.location1 > 0:
                    self.location1 -= 1
        self.history_list.append(action)


    def player_return(self, player):
        if player == 0:
            return self.reward
        else:
            return -self.reward

    def returns(self):
        return [self.reward, -self.reward]

    def clone(self):
        return copy.deepcopy(self)


def state_to_board(state, shape):
    return state.clone()
