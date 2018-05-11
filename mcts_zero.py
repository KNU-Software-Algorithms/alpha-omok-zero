from __future__ import print_function
from omok_env import OmokEnv
import time
import sys
from collections import deque, defaultdict
import numpy as np
from numpy import random, sqrt, argwhere, zeros, ones, exp
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from neural_net import PVNet

N, Q, P = 0, 1, 2
CURRENT = 0
OPPONENT = 1
COLOR = 2
BLACK = 1
WHITE = 0
N_BLOCK = 5
CHANNEL = 128
BOARD_SIZE = 3
HISTORY = 2
N_SIMUL = 100
N_GAME = 1
N_EPOCH = 1
N_ITER = 100000
TAU_THRES = 2
BATCH_SIZE = 16
LR = 2e-5
L2 = 1e-4


class MCTS:
    def __init__(self, n_block, channel, board_size, n_history, n_simul):
        self.env_simul = OmokEnv(board_size, n_history, display=False)
        self.model = PVNet(n_block, n_history*2+1, channel, board_size)
        self.board_size = board_size
        self.n_simul = n_simul
        self.tree = None
        self.root = None
        self.root_key = None
        self.state = None
        self.board = None
        # used for backup
        self.key_memory = deque()
        self.action_memory = deque()
        self.reset_tree()

    def reset_tree(self):
        self.tree = defaultdict(lambda: zeros((self.board_size**2, 3)))

    def get_action(self, state, board, tau):
        self.root = state.copy()
        self.root_key = hash(self.root.tostring())
        self._simulation(state)
        # init root board after simulatons
        self.board = board
        # argmax visit count
        action, pi = self._selection(self.root_key, c_pucb=0)
        if tau == 1:
            action = random.choice(self.board_size**2, p=pi)

        return action, pi

    def _simulation(self, state):
        start = time.time()
        finish = 0
        for sim in range(self.n_simul):
            print('\rsimulation: {}'.format(sim + 1), end='')
            sys.stdout.flush()
            # reset state
            self.state, self.board = self.env_simul.reset(state)
            done = False

            while not done:
                key = hash(self.state.tostring())
                # search my tree
                if key in self.tree:
                    # selection
                    action = self._selection(key, c_pucb=5)
                    self.action_memory.appendleft(action)
                    self.key_memory.appendleft(key)
                else:
                    # expansion
                    reward, done = self._expansion(key, self.state)
                    break

                self.state, self.board, reward, done = self.env_simul.step(
                    action)

            if done:
                # backup & reset memory
                self._backup(reward)
                finish = time.time() - start
                # if finish >= self.think_time:
                #     break
        print('\r{} simulations end ({:0.0f}s)'.format(sim + 1, finish))

    def _selection(self, key, c_pucb):
        edges = self.tree[key]
        pucb = self._get_pucb(edges, key, c_pucb)

        if c_pucb == 0:
            visit = edges[:, N]
            print('\nvisit count')
            print(visit.reshape(self.board_size, self.board_size).round())
            action = argwhere(visit == visit.max()).flatten()
            action = action[random.choice(len(action))]
            pi = exp(visit) / exp(visit).sum()
            pi = visit / visit.sum()
            print('\npi')
            print(pi.reshape(
                self.board_size, self.board_size).round(decimals=2))
            return action, pi

        if self.board[COLOR][0] == WHITE:
            # black's choice
            action = argwhere(pucb == pucb.max()).flatten()
        else:
            # white's choice
            action = argwhere(pucb == pucb.min()).flatten()
        action = action[random.choice(len(action))]
        return action

    def _expansion(self, key, state):
        edges = self.tree[key]
        state_input = Variable(Tensor([state.reshape(
            HISTORY*2+1, self.board_size, self.board_size)]))
        prior, value = self.model(state_input)
        edges[:, P] = prior.data.cpu().numpy()[0]
        done = True
        return value.data.cpu().numpy()[0], done

    def _backup(self, reward):
        # update edges in my tree
        while self.key_memory:
            key = self.key_memory.popleft()
            edges = self.tree[key]
            if self.action_memory:
                action = self.action_memory.popleft()
                edges[action][N] += 1
                edges[action][Q] += (reward - edges[action][Q]) / \
                    edges[action][N]
        return 0

    def _get_no_legal_loc(self, board):
        board_fill = board[CURRENT] + board[OPPONENT]
        legal_action = argwhere(board_fill == 0).flatten()
        return board_fill, legal_action

    def _get_pucb(self, edges, key, c_pucb):
        no_legal_loc, legal_action = self._get_no_legal_loc(self.board)
        prob = edges[:, P]
        if key == self.root_key:
            noise = random.dirichlet(0.15 * ones(len(legal_action)))
            for i, action in enumerate(legal_action):
                prob[action] = 0.75 * prob[action] + 0.25 * noise[i]
        total_N = edges.sum(0)[N]
        # black's pucb
        if self.board[COLOR][0] == WHITE:
            no_legal_loc *= -999999
            pucb = edges[:, Q] + \
                c_pucb * prob * sqrt(total_N) / (edges[:, N] + 1) + \
                no_legal_loc
        # white's pucb
        else:
            no_legal_loc *= 999999
            pucb = edges[:, Q] - \
                c_pucb * prob * sqrt(total_N) / (edges[:, N] + 1) + \
                no_legal_loc
        return pucb


def self_play(n_game):
    for g in range(n_game):
        print('#' * (BOARD_SIZE - 4),
              ' GAME: {} '.format(g + 1),
              '#' * (BOARD_SIZE - 4))
        # reset state
        samples = []
        state, board = Env.reset()
        done = False
        move = 0
        while not done:
            Env.render()
            if move < TAU_THRES:
                tau = 1
            else:
                tau = 0
            # start simulations
            action, pi = Agent.get_action(state, board, tau)
            # print state evaluation
            state = state.reshape(HISTORY*2+1, BOARD_SIZE, BOARD_SIZE)
            state_input = Variable(Tensor([state]))
            prob, value = Agent.model(state_input)
            print('\nprob')
            print(
                prob.data.cpu().numpy()[0].reshape(
                    BOARD_SIZE, BOARD_SIZE).round(decimals=2))
            print('\nvalue')
            print(value.data.cpu().numpy()[0].round(decimals=4))
            # collect samples
            samples.append((state, pi))
            state, board, z, done = Env.step(action)
            move += 1
        if done:
            if z == 1:
                Result['Black'] += 1
            elif z == -1:
                Result['White'] += 1
            else:
                Result['Draw'] += 1

            for i in range(len(samples)):
                Memory.appendleft(
                    (samples[i][0], samples[i][1], z))
            # render & reset tree
            Env.render()
            Agent.reset_tree()
        # result
        blw, whw, drw = Result['Black'], Result['White'], Result['Draw']
        print('')
        print('=' * 20, " {}  Game End  ".format(blw+whw+drw), '=' * 20)
        stats = (
            'Black Win: {}  White Win: {}  Draw: {}  Winrate: {:.2f}%'.format(
                blw, whw, drw, blw/(blw+whw)*100 if blw+whw != 0 else 0))
        print('memory size:', len(Memory))
        print(stats, '\n')


STEPS = 0


def train(n_epoch):
    global STEPS
    print('=' * 20, ' Start Learning ', '=' * 20,)

    dataloader = DataLoader(Memory,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=use_cuda)

    optimizer = optim.SGD(Agent.model.parameters(),
                          lr=LR,
                          momentum=0.9,
                          weight_decay=L2)

    for epoch in range(n_epoch):
        running_loss = 0.

        for i, (s, pi, z) in enumerate(dataloader):
            if use_cuda:
                s_batch = Variable(s.float()).cuda()
                pi_batch = Variable(pi.float()).cuda()
                z_batch = Variable(z.float()).cuda()
            else:
                s_batch = Variable(s.float())
                pi_batch = Variable(pi.float())
                z_batch = Variable(z.float())

            p_batch, v_batch = Agent.model(s_batch)

            loss = F.mse_loss(v_batch, z_batch) + \
                torch.mean(torch.sum(-pi_batch * torch.log(p_batch), 1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            STEPS += 1

            print('{:3} step loss: {:.3f}'.format(
                STEPS, running_loss / (i + 1)))


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    use_cuda = torch.cuda.is_available()
    print('cuda:', use_cuda)
    Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    Memory = deque(maxlen=5120)
    Env = OmokEnv(BOARD_SIZE, HISTORY)
    Agent = MCTS(N_BLOCK, CHANNEL, BOARD_SIZE, HISTORY, N_SIMUL)
    Result = {'Black': 0, 'White': 0, 'Draw': 0}

    if use_cuda:
        Agent.model.cuda()

    for i in range(N_ITER):
        self_play(N_GAME)

        if len(Memory) == 5120:
            train(N_EPOCH)
            Memory.clear()
            Result = {'Black': 0, 'White': 0, 'Draw': 0}
            torch.save(
                Agent.model.state_dict(),
                '{}_step_model.pickle'.format(STEPS))
