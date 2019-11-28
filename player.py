import random
from node import Node
import logging
import numpy 
import pickle
from os import listdir
from copy import deepcopy
import concurrent.futures as cf
import threading as thread
logger = logging.getLogger('mcts')

class MinMaxHelperClass:
    def __init__(self, Player=None, untried_actions=None, depth=None, maxP=None, move_type=None, action=None):
        if Player != None:
            if depth == 0:
                Player.eval(move_type, action)
            else:
                Player.minmax_tree(untried_actions, depth, maxP, move_type, action)

class Player:

    def get_first_move(self):
        with open('DraftArtistV2/models/hero_freqs.pickle', 'rb') as f:
            a, p = pickle.load(f)
            return numpy.random.choice(a, size=1, p=p)[0]

    def get_move(self, move_type):
        raise NotImplementedError

class RandomPlayer(Player):

    def __init__(self, draft):
        self.draft = draft
        self.name = 'random'

    def get_move(self, move_type):
        """
        decide the next move
        """
        if self.draft.if_first_move():
            return self.get_first_move()
        moves = self.draft.get_moves()
        return random.sample(moves, 1)[0]


class HighestWinRatePlayer(Player):

    def __init__(self, draft):
        self.draft = draft
        self.name = 'hwr'
        with open('models/hero_win_rates.pickle', 'rb') as f:
            self.win_rate_dist = pickle.load(f)

    def get_move(self, move_type):
        """
        decide the next move
        """
        if self.draft.if_first_move():
            return self.get_first_move()
        moves = self.draft.get_moves()
        move_win_rates = [(m, self.win_rate_dist[m]) for m in moves]
        best_move, best_win_rate = sorted(move_win_rates, key=lambda x: x[1])[-1]
        return best_move

class MinMaxPlayerV2(Player):
    def __init__(self, actions, depth, maxPlayer, draft):
        self.depth = depth
        self.maxPlayer = maxPlayer
        self.draft = draft
        self.name = 'minmaxV2'
        self.winrates = self.getwinrate()
        self.executor = cf.ThreadPoolExecutor(max_workers=20)
        self.curdepth = 3
        self.Helper = MinMaxHelperClass

    def get_move(self, move_type):
        if self.draft.if_first_move():
            return self.get_first_move()
        root = Node(player=self.draft.player, untried_actions=self.draft.get_moves())
        _, pick = self.minmax_tree(deepcopy(root.untried_actions), self.curdepth, True, move_type, 0)
        return pick

    def minmax_tree(self, untried_actions, depth, maxP, move_type, action):
        if maxP:
            value = -numpy.inf
        else:
            value = numpy.inf
        processes = []
        choice = 0
        if depth != 0:
            with cf.ProcessPoolExecutor(max_workers=6) as executor:
                for i in range(0, 113):
                    if i in untried_actions:
                        temp = deepcopy(untried_actions)
                        temp.discard(i)
                        if maxP:
                            if depth == 1:
                                processes.append(executor.submit(self.Helper, self, temp, depth-1, False, move_type, i))
                            else:
                                newval, choice = self.minmax_tree(temp, depth-1, False, move_type, i)
                                if value < newval:
                                    choice = i
                                    value = newval
                        else:
                            if depth == 1:
                                processes.append(executor.submit(self.Helper, self, temp, depth-1, True, move_type, i))
                            else:
                                newval, newchoice = self.minmax_tree(temp, depth-1, True, move_type, i)
                                if value > newval:
                                    choice = i
                                    value = newval
                if depth == 1:
                    for proc in cf.as_completed(processes):
                        processes.remove(proc)
                        val, newchoice = proc.result()
                        if maxP:
                            if value < val:
                                value = val
                                choice = newchoice
                        else:
                            if value > val:
                                value = val
                                choice = newchoice
                return value, choice
        else:
            return self.eval(move_type, action), action

    def eval(self, move_type, action):
        if move_type == 'ban':
            winrate = 0
            for val in range(5, 10):
                try:
                    person = self.winrates[val]
                    winrate += person[str(action)]
                except:
                    winrate += 0
        else:
            winrate = 0
            for val in range(0, 5):
                try:
                    person = self.winrates[val]
                    winrate += person[str(action)]
                except:
                    winrate += 0
        if winrate == 0:
            return -500
        else:
            return winrate/5

    def getwinrate(self):
        path = "DraftArtistV2\Data\People"
        filelist = listdir(path)
        dict = []
        lst = []
        dictionary = {}
        for files in filelist:
            with open(path+"\\"+files, "r") as file:
                for lines in file.readlines():
                    lines = lines.split(" ")
                    heroid = lines[1].replace(',', '').replace("'", "")
                    win = int(lines[7].replace(',', ''))
                    games = int(lines[5].replace(',', ''))
                    try:
                        dictionary[heroid] = (win/games)
                    except:
                        dictionary[heroid] = 0
                dict.append(dictionary)
                dictionary = {}
        return dict

class MinMaxPlayer(Player):

    def __init__(self, actions, depth, maxPlayer, draft):
        self.draft = draft
        self.actions = actions
        self.depth = depth
        self.maxPlayer = maxPlayer
        self.name = 'minmax'
        self.maxiters = 800
        self.tree = None

    def build_minmax_tree(self, root, node, depth, player):
        if (depth >= 20):
            return root
        for i in range(0, len(node.untried_actions)):
            if i in node.untried_actions:
                c = node.expand(i, player, self.draft.get_moves())
                c.untried_actions.discard(i)
        for child in node.children:
            self.build_minmax_tree(root, child, depth+1, player)

    def get_move(self, move_type):
        if self.draft.if_first_move():
            return self.get_first_move()

        root = Node(player=self.draft.player, untried_actions=self.draft.get_moves())
        self.tree = self.build_minmax_tree(root, root, 0, self.draft.player)
        for i in range(self.maxiters):
            node = root
            if(self.depth == 0 or node.untried_actions == []):
                return node.wins
            if(self.maxPlayer):
                value = -numpy.inf
                for action in node.untried_actions:
                    value = max(value, MinMaxPlayer(action, self.depth - 1, False, self.draft).get_move(move_type))
                return value
            else:
                value = numpy.inf
                for action in node.untried_actions:
                    value = min(value, MinMaxPlayer(action, self.depth - 1, True, self.draft).get_move(move_type))
                return value

class MCTSPlayer(Player):

    def __init__(self, name, draft, maxiters, c):
        self.draft = draft
        self.name =name
        self.maxiters = maxiters
        self.c = c

    def get_move(self, move_type):
        """
        decide the next move
        """
        if self.draft.if_first_move():
            return self.get_first_move()

        root = Node(player=self.draft.player, untried_actions=self.draft.get_moves(), c=self.c)

        for i in range(self.maxiters):
            node = root
            tmp_draft = self.draft.copy()

            # selection - select best child if parent fully expanded and not terminal
            while len(node.untried_actions) == 0 and node.children != []:
                # logger.info('selection')
                node = node.select()
                tmp_draft.move(node.action)
            # logger.info('')

            # expansion - expand parent to a random untried action
            if len(node.untried_actions) != 0:
                # logger.info('expansion')
                a = random.sample(node.untried_actions, 1)[0]
                tmp_draft.move(a)
                p = tmp_draft.player
                node = node.expand(a, p, tmp_draft.get_moves())
            # logger.info('')

            # simulation - rollout to terminal state from current
            # state using random actions
            while not tmp_draft.end():
                # logger.info('simulation')
                moves = tmp_draft.get_moves()
                a = random.sample(moves, 1)[0]
                tmp_draft.move(a)
            # logger.info('')

            # backpropagation - propagate result of rollout game up the tree
            # reverse the result if player at the node lost the rollout game
            while node != None:
                # logger.info('backpropagation')
                if node.player == 0:
                    result = tmp_draft.eval()
                else:
                    result = 1 - tmp_draft.eval()
                node.update(result)
                node = node.parent
            # logger.info('')

        return root.select_final()


class AssocRulePlayer(Player):

    def __init__(self, draft):
        self.draft = draft
        self.name = 'assocrule'
        self.load_rules(match_num=3056596,
                        oppo_team_spmf_path='apriori/dota_oppo_team_output.txt',
                        win_team_spmf_path='apriori/dota_win_team_output.txt',
                        lose_team_spmf_path='apriori/dota_lose_team_output.txt')

    def load_rules(self, match_num, oppo_team_spmf_path, win_team_spmf_path, lose_team_spmf_path):
        self.oppo_1_rules = dict()
        self.oppo_2_rules = dict()
        with open(oppo_team_spmf_path, 'r') as f:
            for line in f:
                items, support = line.split(' #SUP: ')
                items, support = list(map(int, items.strip().split(' '))), int(support.strip())
                # S(-e), because -e is losing champion encoded in 1xxx
                if len(items) == 1 and items[0] > 1000:
                    self.oppo_1_rules[frozenset(items)] = support / match_num
                elif len(items) == 2 and (items[0] < 1000 and items[1] > 1000):
                    self.oppo_2_rules[frozenset(items)] = support / match_num
                else:
                    continue

        self.win_rules = dict()
        with open(win_team_spmf_path, 'r') as f:
            for line in f:
                items, support = line.split(' #SUP: ')
                items, support = list(map(int, items.strip().split(' '))), int(support.strip())
                if len(items) == 1:
                    continue
                self.win_rules[frozenset(items)] = support / match_num

        self.lose_rules = dict()
        with open(lose_team_spmf_path, 'r') as f:
            for line in f:
                items, support = line.split(' #SUP: ')
                items, support = list(map(int, items.strip().split(' '))), int(support.strip())
                if len(items) == 1:
                    continue
                self.lose_rules[frozenset(items)] = support / match_num

    def get_move(self, move_type):
        if self.draft.if_first_move():
            return self.get_first_move()

        player = self.draft.next_player
        # if ban, we are selecting the best hero for opponent
        if move_type == 'ban':
            player = player ^ 1
        allies = frozenset(self.draft.get_state(player))
        oppo_player = player ^ 1
        # enemy id needs to add 1000
        enemies = frozenset([i+1000 for i in self.draft.get_state(oppo_player)])

        R = list()

        ally_candidates = list()
        for key in self.win_rules:
            intercept = allies & key
            assoc = key - intercept
            if len(intercept) > 0 and len(assoc) == 1:
                assoc = next(iter(assoc))  # extract the move from the set
                if assoc in self.draft.get_moves():
                    win_sup = self.win_rules[key]
                    lose_sup = self.lose_rules.get(key, 0.0)   # lose support may not exist
                    win_rate = win_sup / (win_sup + lose_sup)
                    ally_candidates.append((allies, key, assoc, win_rate))
        # select top 5 win rate association rules
        ally_candidates = sorted(ally_candidates, key=lambda x: x[-1])[-5:]
        R.extend([a[-2] for a in ally_candidates])

        enemy_candidates = list()
        for key in self.oppo_2_rules:
            intercept = enemies & key
            assoc = key - intercept
            if len(intercept) == 1 and len(assoc) == 1:
                assoc = next(iter(assoc))       # extract the move from the set
                if assoc in self.draft.get_moves():
                    confidence = self.oppo_2_rules[key] / self.oppo_1_rules[intercept]
                    enemy_candidates.append((enemies, key, assoc, confidence))
        # select top 5 confidence association rules
        enemy_candidates = sorted(enemy_candidates, key=lambda x: x[-1])[-5:]
        R.extend([e[-2] for e in enemy_candidates])

        if len(R) == 0:
            moves = self.draft.get_moves()
            return random.sample(moves, 1)[0]
        else:
            move = random.choice(R)
            return move
