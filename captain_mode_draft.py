from player import RandomPlayer, MCTSPlayer, AssocRulePlayer, HighestWinRatePlayer, MCTSPlayerSkill, \
    KNNPlayer2, MatrixFactorizationWinratePlayer, MatrixFactorizationThresholdPlayer, MCTSPlayerParallel, \
    MinMaxPlayer, MinMaxPlayerV2
from utils.parser import parse_mcts_maxiter_c, parse_rave_maxiter_c_k
import pickle
import logging
import numpy as np
import random

class Draft:
    """
    class handling state of the draft
    """

    def __init__(self, env_path=None, env_path2=None, p0_model_str=None, p1_model_str=None):
        if env_path and p0_model_str and p1_model_str:
            self.outcome_model, self.M = self.load(env_path)
            self.outcome_model_with_skill, self.M_with_skill = self.load(env_path2)
            self.state = [[], []]
            self.avail_moves = set(range(self.M_with_skill+1))
            self.move_cnt = [0, 0]
            self.player = None  # current player's turn
            self.next_player = 0  # next player turn
            self.controllers = [self.load_red_controllers(), self.load_blue_controllers()]
            self.sumdur0 = 0
            self.sumdur1 = 0
            # player 0 will pick first and be red team; player 1 will pick next and be blue team
            self.player_models = [self.construct_player_model(p0_model_str),
                                  self.construct_player_model(p1_model_str)]

    def get_state(self, player):
        return self.state[player]

    def get_player(self):
        return self.player_models[self.next_player]

    def construct_player_model(self, player_model_str):
        if player_model_str == 'random':
            return RandomPlayer(draft=self)
        elif player_model_str.startswith('mcts'):
            max_iters, c = parse_mcts_maxiter_c(player_model_str)
            return MCTSPlayer(name=player_model_str, draft=self, maxiters=max_iters, c=c)
        elif player_model_str.startswith('skillmcts'):
            max_iters, c = parse_mcts_maxiter_c(player_model_str)
            return MCTSPlayerSkill(name=player_model_str, draft=self, maxiters=max_iters, c=c)
        elif player_model_str == 'assocrule':
            return AssocRulePlayer(draft=self)
        elif player_model_str == 'hwr':
            return HighestWinRatePlayer(draft=self)
        elif player_model_str.split("_")[0] == 'knn':
            return KNNPlayer2(draft=self, k=int(player_model_str.split("_")[1]),
                              distance=player_model_str.split("_")[2])
        elif player_model_str.startswith('mfw'):
            return MatrixFactorizationWinratePlayer(draft=self)
        elif player_model_str.startswith('mfth'):
            return MatrixFactorizationThresholdPlayer(draft=self)
        elif player_model_str.startswith('parallelmcts'):
            max_iters, c = parse_mcts_maxiter_c(player_model_str)
            return MCTSPlayerParallel(name=player_model_str,draft=self,maxiters=max_iters, c=c)
        elif player_model_str.startswith('minmaxV2'):
            values = player_model_str.split("_")
            return MinMaxPlayerV2(depth=int(values[1]), maxPlayer=True, draft=self)
        elif player_model_str.startswith('minmax'):
            values = player_model_str.split("_")
            return MinMaxPlayer(actions=0, depth=int(values[1]), maxPlayer=True, draft=self)
        else:
            raise NotImplementedError

    def load(self, env_path):
        with open('models/{}'.format(env_path), 'rb') as f:
            # outcome model predicts the red team's  win rate
            # M is the number of champions
            outcome_model, M = pickle.load(f)
        return outcome_model, M

    def load_red_controllers(self):
        try:
            con1 = self.calculatewinrates(open("input/red/player1.txt", "r").readlines())
            con2 = self.calculatewinrates(open("input/red/player2.txt", "r").readlines())
            con3 = self.calculatewinrates(open("input/red/player3.txt", "r").readlines())
            con4 = self.calculatewinrates(open("input/red/player4.txt", "r").readlines())
            con5 = self.calculatewinrates(open("input/red/player5.txt", "r").readlines())
            return [con1, con2, con3, con4, con5]
        except:
            return []

    def load_blue_controllers(self):
        try:
            con1 = self.calculatewinrates(open("input/blue/player1.txt", "r").readlines())
            con2 = self.calculatewinrates(open("input/blue/player2.txt", "r").readlines())
            con3 = self.calculatewinrates(open("input/blue/player3.txt", "r").readlines())
            con4 = self.calculatewinrates(open("input/blue/player4.txt", "r").readlines())
            con5 = self.calculatewinrates(open("input/blue/player5.txt", "r").readlines())
            return [con1, con2, con3, con4, con5]
        except:
            return []

    def eval(self, withcontrolers=False):
        if withcontrolers:
            assert self.end()
            x = np.zeros((1, self.M_with_skill + 10))
            x[0, self.state[0]] = 1
            x[0, self.state[1]] = -1
            x[0, -10] = self.findwinrate(self.controllers[0][0], self.state[0][0])
            x[0, -9] = self.findwinrate(self.controllers[0][1], self.state[0][1])
            x[0, -8] = self.findwinrate(self.controllers[0][2], self.state[0][2])
            x[0, -7] = self.findwinrate(self.controllers[0][3], self.state[0][3])
            x[0, -6] = self.findwinrate(self.controllers[0][4], self.state[0][4])
            x[0, -5] = self.findwinrate(self.controllers[1][0], self.state[1][0])
            x[0, -4] = self.findwinrate(self.controllers[1][1], self.state[1][1])
            x[0, -3] = self.findwinrate(self.controllers[1][2], self.state[1][2])
            x[0, -2] = self.findwinrate(self.controllers[1][3], self.state[1][3])
            x[0, -1] = self.findwinrate(self.controllers[1][4], self.state[1][4])
            red_team_win_rate = self.outcome_model_with_skill.predict_proba(x)[0, 1]
            return red_team_win_rate
        assert self.end()
        x = np.zeros((1, self.M))
        x[0, self.state[0]] = 1
        x[0, self.state[1]] = -1
        red_team_win_rate = self.outcome_model.predict_proba(x)[0, 1]
        return red_team_win_rate

    def copy(self):
        """
        make copy of the board
        """
        copy = Draft()
        copy.outcome_model = self.outcome_model
        copy.M = self.M
        copy.outcome_model_with_skill = self.outcome_model_with_skill
        copy.M_with_skill = self.M_with_skill
        copy.state = [self.state[0][:], self.state[1][:]]
        copy.avail_moves = set(self.avail_moves)
        copy.move_cnt = self.move_cnt[:]
        copy.player = self.player
        copy.next_player = self.next_player
        copy.player_models = self.player_models
        copy.controllers = self.controllers
        return copy

    def move(self, move):
        """
        take move of form [x,y] and play
        the move for the current player
        """
        # player 0 -> place 1,  player 1 -> place -1
        # val = - self.player * 2 + 1
        self.player = self.next_player
        self.next_player = self.decide_next_player()
        move_type = self.decide_move_type()
        if move_type == 'pick':
            self.state[self.player].append(move)
        elif move_type == 'ban':
            pass
        else:
            raise NotImplementedError
        self.avail_moves.remove(move)
        self.move_cnt[self.player] += 1
        # logger.info('choose move: player {} ({}), move_cnt: {}, move: {}'.format(self.player, self.get_player().name, self.move_cnt[self.player], move))

    def decide_move_type(self):
        """ decide either the move to take is ban or pick """
        move_cnt = self.move_cnt[0] + self.move_cnt[1]
        if move_cnt in [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 18, 19]:
            return 'ban'
        else:
            return 'pick'

    def decide_next_player(self):
        """
        determine next player before a move is taken
        """
        move_cnt = self.move_cnt[0] + self.move_cnt[1]
        if move_cnt in [0, 2, 4, 6, 7, 9, 11, 13, 15, 17, 20]:
            return 1
        else:
            return 0

    def if_first_move(self):
        """ whether the next move is the first move """
        if self.move_cnt[0] == 0 and self.move_cnt[1] == 0:
            return True
        return False

    def get_moves(self):
        """
        return remaining possible draft moves
        (i.e., where there are no 1's or -1's)
        """
        if self.end():
            return set([])
        return set(self.avail_moves)
        # zero_indices = np.argwhere(self.state == 0).tolist()
        # zero_indices = []
        # for i in range(self.M):
        #     if i not in self.state[0] or i not in self.state[1]:
        #         zero_indices.append(i)
        # logger.info('get moves: player {} ({}), move_cnt: {}, moves: {}'.format(self.player, self.get_player().name, self.move_cnt[self.player], zero_indices))
        # return zero_indices

    def end(self):
        """
        return True if all players finish drafting
        """
        if self.move_cnt[0] == 11 and self.move_cnt[1] == 11:
            return True
        return False

    def print_move(self, match_id, move_duration, move_id, move_type):
        move_str = 'match {} player {} ({:15s}), {:4s}: {:3d}, move_cnt: {}, duration: {:.3f}' \
            .format(match_id, self.player, self.player_models[self.player].name, move_type, move_id,
                    self.move_cnt[self.player], move_duration)
        logger = logging.getLogger('mcts')
        logger.warning(move_str)
        return move_str

    def getcontroller(self):
        pickrounds = [3, 4, 7, 8, 10]
        player = self.next_player
        if player == 0:
            if self.move_cnt[0] in pickrounds:
                return self.controllers[0][pickrounds.index(self.move_cnt[0])]
            return random.sample(self.controllers[0], 1)[0]
        else:
            if self.move_cnt[1] in pickrounds:
                return self.controllers[1][pickrounds.index(self.move_cnt[1])]
            return random.sample(self.controllers[1], 1)[0]

    def findwinrate(self, controller, heroid):
        #for hero in controller:
        #    hero = hero.split(",")
        #    id = int(hero[0].split(":")[1].replace('"', ''))
        #    if id == heroid:
        #        gameplayed = int(hero[2].split(":")[1].replace('"', ''))
        #        if gameplayed == 0:
        #            return 0.0
        #        return int(hero[3].split(":")[1].replace('"', '')) / gameplayed
        return controller[heroid]

    def calculatewinrates(self, controller):
        ratings = [0] * (self.M_with_skill + 1)
        for hero in controller:
            hero = hero.replace('"', '').split(',')
            id = int(hero[0].split(':')[1])
            games = int(hero[2].split(':')[1])
            if games != 0 and id < 114:
                wins = int(hero[3].split(':')[1])
                ratings[id] = wins / games
        return ratings
