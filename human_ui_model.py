from __future__ import print_function
from omok_env import OmokEnv
from mcts_puct import MCTS
import torch


COLUMN = {"a": 0, "b": 1, "c": 2,
          "d": 3, "e": 4, "f": 5,
          "g": 6, "h": 7, "i": 8,
          "j": 9, "k": 10, "l": 11,
          "m": 12, "n": 13, "o": 14,
          "A": 0, "B": 1, "C": 2,
          "D": 3, "E": 4, "F": 5,
          "G": 6, "H": 7, "I": 8,
          "J": 9, "K": 10, "L": 11,
          "M": 12, "N": 13, "O": 14}
N_BLOCK = 5
CHANNEL = 8
BOARD_SIZE = 9
HISTORY = 2
N_SIMUL = 16000
N_GAME = 5


class HumanAgent:
    def get_action(self):
        last_str = str(BOARD_SIZE)
        for k, v in COLUMN.items():
            if v == BOARD_SIZE - 1:
                last_str += k
                break
        move_target = str(input('1a ~ {}: '.format(last_str)))
        row = int(move_target[:1]) - 1
        col = COLUMN[move_target[1:2]]
        action = row * BOARD_SIZE + col
        return action


class HumanUI:
    def __init__(self):
        self.human = HumanAgent()
        self.ai = MCTS(N_BLOCK, CHANNEL, BOARD_SIZE, HISTORY, N_SIMUL, 'learn')

    def get_action(self, state, board, idx):
        if idx % 2 == 0:
            action, _ = self.ai.get_action(state, board, tau=0)
        else:
            action = self.human.get_action()
        return action


def play():
    USE_CUDA = torch.cuda.is_available()
    env = OmokEnv(BOARD_SIZE, HISTORY)
    manager = HumanUI()

    if USE_CUDA:
        manager.ai.model.cuda()

    model_path = None

    if model_path:
        print('load model: {}\n'.format(model_path))
        manager.ai.model.load_state_dict(torch.load(model_path))

    result = {'Black': 0, 'White': 0, 'Draw': 0}

    for g in range(N_GAME):
        print('#####  Game: {}  #####'.format(g + 1))
        state, board = env.reset()
        done = False
        idx = 0
        while not done:
            env.render()
            # start simulations
            action = manager.get_action(state, board, idx)
            state, board, z, done = env.step(action)
            idx += 1
        if done:
            if z == 1:
                result['Black'] += 1
            elif z == -1:
                result['White'] += 1
            else:
                result['Draw'] += 1
            # render & reset tree
            env.render()
            manager.ai.reset_tree()
        # result
        blw, whw, drw = result['Black'], result['White'], result['Draw']
        print('')
        print('=' * 20, " {}  Game End  ".format(blw + whw + drw), '=' * 20)
        stats = (
            'Black Win: {}  White Win: {}  Draw: {}  Winrate: {:.2f}%'.format(
                blw, whw, drw, (blw + 0.5 * drw) / (blw + whw + drw) * 100))
        print(stats, '\n')


if __name__ == '__main__':
    play()
