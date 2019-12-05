import random
import math

from models.heroprofile import HeroProfile
from models.MatrixFactorization import startNormalWinrateMatrixFac
from models.MatrixFactorization import startTresholdMatrixFac
from node import Node
import logging
import numpy
import pickle

logger = logging.getLogger('mcts')


class Player:

    def get_first_move(self):
        with open('models/hero_freqs.pickle', 'rb') as f:
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


class MCTSPlayer(Player):

    def __init__(self, name, draft, maxiters, c):
        self.draft = draft
        self.name = name
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
        enemies = frozenset([i + 1000 for i in self.draft.get_state(oppo_player)])

        R = list()

        ally_candidates = list()
        for key in self.win_rules:
            intercept = allies & key
            assoc = key - intercept
            if len(intercept) > 0 and len(assoc) == 1:
                assoc = next(iter(assoc))  # extract the move from the set
                if assoc in self.draft.get_moves():
                    win_sup = self.win_rules[key]
                    lose_sup = self.lose_rules.get(key, 0.0)  # lose support may not exist
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
                assoc = next(iter(assoc))  # extract the move from the set
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


class KNNPlayer(Player):
    def __init__(self, draft, k):
        self.draft = draft
        self.name = 'knn'
        self.heroprofiles = self.loadheroes()
        self.allroles = ["Carry", "Escape", "nuker", "initiator", "durable", "disabler", "jungler", "support", "pusher"]
        self.k = k

    def loadheroes(self):
        file = open("input/heros.txt", "r").readlines()
        heroprofiles = []
        for line in file:
            line = line.split("]")

            line[0] = line[0].split("{")
            line[0][1] = line[0][1].split(",")
            id = line[0][1][0].split(":")[1]
            name = line[0][1][2].split(":")[1]

            count = 2
            for ability in line[0][2:]:
                line[0][count] = ability.replace("}", "")
                count += 1

            line[1] = line[1].split("{")

            count = 1
            for role in line[1][1:]:
                s = role.replace("}", "").split(",")
                switcher = {
                    0: "Carry",
                    1: "Escape",
                    2: "nuker",
                    3: "initiator",
                    4: "durable",
                    5: "disabler",
                    6: "jungler",
                    7: "support",
                    8: "pusher"
                }
                s = ",".join(s[0:3])
                line[1][count] = s + ',"rolename":' + switcher.get(int(role[9]))
                count += 1

            line[2] = line[2].split("{")

            count = 1
            for talent in line[2][1:]:
                line[2][count] = talent.replace("}", "")
                count += 1

            line[3] = line[3].split(',"language')[0].split("{")[1].split(",")

            count = 0
            for stat in line[3]:
                line[3][count] = stat.replace("}", "")
                count += 1

            hero = HeroProfile(id=id, name=name, abilities=line[0][2:], roles=line[1][1:], talents=line[2][1:],
                               stats=line[3])

            heroprofiles.append(hero)
        return heroprofiles

    def get_move(self, move_type):
        if self.draft.if_first_move():
            return self.get_first_move()

        player = self.draft.next_player
        if move_type == 'ban':
            player = player ^ 1
        allies = self.draft.get_state(player)

        moves = self.draft.get_moves()
        perfekt = self.perfekt(allies)

        rating = []
        for hero in self.heroprofiles:
            if int(hero.ID) in moves:
                roles = []
                for role in hero.Roles:
                    roles.append(role.split(",")[-1].split(":")[-1])
                rating.append(
                    (hero.ID, numpy.sqrt(abs(len(self.intersection(roles, perfekt.Roles)) - len(perfekt.Roles)))))
        rating = sorted(rating, key=lambda x: x[-1])[:5]

        return self.findbest(rating, self.draft.getcontroller())

    def findbest(self, rating, player):
        newrating = []
        for hero in player:
            hero = hero.replace('"', '').split(',')
            id = hero[0].split(':')
            for rated in rating:
                if rated[0] == id[1]:
                    winrate = 0
                    if int(hero[2].split(':')[1]) != 0:
                        winrate = float(hero[3].split(':')[1]) / float(hero[2].split(':')[1])
                    if winrate == 0:
                        winrate = 1
                    newrating.append((id[1], rated[1] / winrate))
        if len(newrating) == 0:
            return int(random.sample(rating, 1)[0][0])
        return int(self.minintuble(newrating)[0])

    def intersection(self, lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3

    def perfekt(self, allies):
        allieprofiles = []
        if allies == []:
            return HeroProfile(id=0, name="", abilities=[], roles=self.allroles, talents=[],
                               stats=[])
        for hero in self.heroprofiles:
            if int(hero.ID) in allies:
                allieprofiles.append(hero)
        roles = self.extractroles(allieprofiles)

        return HeroProfile(id=0, name="", abilities=[], roles=roles, talents=[],
                           stats=[])

    def extractroles(self, profiles):
        all = self.allroles.copy()
        for profile in profiles:
            for role in profile.Roles:
                role = role.split(",")[-1].split(":")[-1]
                if role in all:
                    all.remove(role)
        return all

    def minintuble(self, list):
        best = list[0]
        for tuble in list[1:]:
            if best[1] > tuble[1]:
                best = tuble
        return best


class MatrixFactorizationWinrate(Player):
    def __init__(self, draft):
        self.draft = draft
        self.name = 'mfw'

    def startMatrix(self):
        snmf = startNormalWinrateMatrixFac()
        return snmf.start()

    def getPlayers(self):
        snmf = startNormalWinrateMatrixFac()
        return snmf.getPlayers()

    def getListOfChosenCharacters(self):
        file = open("input/characters.txt", "r").readlines()
        return file

    def addCharactersToList(self, fileinput):
        file = open("input/characters.txt", "r")
        file.write(fileinput)
        return file

    def get_move(self, move_type):
        if move_type == 'ban':
            self.getBanMove()
        else:
            self.getBestMove()

    def getBestMove(self, player):
        nmf = self.startMatrix()
        nmfp = self.getPlayers()
        if player == 1:
            maxval = 0
            for x in range(0, len(nmf[0])):
                if nmf[0][x] > maxval:
                    maxval = nmf[0][x]
                    for i in [i for i, x in enumerate(nmfp[0][i][9][0]) if x == maxval]:
                        id = nmfp[0][i][0][1]
                        if id not in self.getListOfChosenCharacters():
                            self.addCharactersToList(id)
                            return id
        return []

    def getBanMove(self):

        return []


class MatrixFactorizationThreshold(Player):
    def __init__(self, draft):
        self.draft = draft
        self.name = 'mft'

    def get_move(self, move_type):
        if move_type == 'ban':
            startTresholdMatrixFac()
        else:
            self.getBestMove()
