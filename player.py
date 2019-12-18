import random
import math
import os
from models.heroprofile import HeroProfile
from models.MatrixFactorization2 import startNormalWinrateMatrixFac
from models.MatrixFactorization2 import startTresholdMatrixFac
from node import Node
import logging
import numpy 
import pickle
import copy
import multiprocessing as mp
#from multiprocessing import Array, Process, Value
from os import listdir
from copy import deepcopy
import concurrent.futures as cf
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
"""
class MinMaxPlayerV2(Player):
    def __init__(self, actions, depth, maxPlayer, draft):
        self.depth = depth
        self.maxPlayer = maxPlayer
        self.draft = draft
        self.name = 'minmaxV2'
        self.winrates = self.getwinrate()
        self.executor = cf.ThreadPoolExecutor(max_workers=20)
        self.curdepth = 3

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
        path = "Data\People"
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
"""
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


class MCTSPlayerSkill(Player):

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
                    result = tmp_draft.eval(True)
                else:
                    result = 1 - tmp_draft.eval(True)
                node.update(result)
                node = node.parent
            # logger.info('')

        return root.select_final()

# Monte Carlo Skill running in parallel. Slower than the others because of the overhead in creating a new process
# in python causes.
class MCTSPlayerParallel(Player):

    def __init__(self, name, draft, maxiters, c):
        self.draft = draft
        self.name = name
        self.maxiters = maxiters
        self.c = c

    def get_move(self, move_type):
        if self.draft.if_first_move():
            return self.get_first_move()
        """arr = Array('i', range(4))
        val = []
        val1 = []
        val.append(Value('d', 0.00))
        val1.append(Value('d', 0.00))
        q = Process(target=self.subfunction, args=val)
        p = Process(target=self.subfunction, args=val1)
        p.start()
        q.start()
        q.join()
        p.join()"""
        pool = mp.Pool(4)
        future_res = [pool.apply_async(self.subfunction) for _ in range(4)]
        res = [f.get() for f in future_res]
        res.sort(key=lambda t: t[1])
        pool.close()
        """highval = -numpy.inf
        pick = 0
        for values in val:
            value = str(values.value)
            value = value.split(".")
            if int(value[1]) > highval:
                pick = int(value[0])
                highval = value[1]
        return pick"""
        return res[-1][0]


    def subfunction(self):
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
        return [root.select_final(),result]

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


# K Nearest Neighbour Player
class KNNPlayer(Player):
    def __init__(self, draft, k, distance):
        self.draft = draft
        self.name = 'knn_' + distance
        self.heroprofiles = self.loadheroes()
        self.allroles = ["Carry", "Escape", "nuker", "initiator", "durable", "disabler", "jungler", "support", "pusher"]
        self.k = k
        self.distance = distance

    # Loads in hero profiles from disk
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

    # Overrides Players get_move function, adding functionality to
    # find a list of characters which fit the remaining roles
    # then from that list, get the character with the highest winrate
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
                dis = self.d(roles, perfekt.Roles)
                rating.append((hero.ID, dis))
        rating = sorted(rating, key=lambda x: x[-1])[:self.k]

        return self.findbest(rating, self.draft.getcontroller())

    def d(self, roles, missing):
        if self.distance == "euclid":
            return self.euclidiandis(roles, missing)
        elif self.distance == "manhatten":
            return self.manhattendis(roles, missing)
        elif self.distance == "cosd":
            return self.CosD(roles, missing)
        elif self.distance == "scd":
            return self.SCD(roles, missing)
        elif self.distance == "sed":
            return self.SED(roles, missing)
        elif self.distance == "kdd":
            return self.KDD(roles, missing)
        elif self.distance == "vwhd":
            return self.VWHD(roles, missing)
        else:
            raise NotImplementedError

    def euclidiandis(self, roles, missingroles):
        return numpy.sqrt(abs(len(self.intersection(roles, missingroles)) - len(missingroles)))

    def manhattendis(self, roles, missingroles):
        return abs(len(self.intersection(roles, missingroles)) - len(missingroles))

    def CosD(self, roles, missingroles):
        i = (numpy.sqrt(len(self.intersection(roles, missingroles)) ** 2) * numpy.sqrt(len(missingroles) ** 2))
        if i == 0:
            return numpy.inf
        return (len(self.intersection(roles, missingroles)) * len(missingroles)) / i

    def SCD(self, roles, missingroles):
        return (numpy.sqrt(len(self.intersection(roles, missingroles))) - numpy.sqrt(len(missingroles))) ** 2

    def SED(self, roles, missingroles):
        return (abs(len(self.intersection(roles, missingroles)) - len(missingroles))) ** 2

    def KDD(self, roles, missingroles):
        i = len(self.intersection(roles, missingroles)) + len(missingroles)
        if i == 0:
            return numpy.inf
        return len(self.intersection(roles, missingroles)) * numpy.log2(
            (2 * len(self.intersection(roles, missingroles))) / i)

    def VWHD(self, roles, missingroles):
        i = min(len(self.intersection(roles, missingroles)), len(missingroles))
        if i == 0:
            return numpy.inf
        return (abs(len(self.intersection(roles, missingroles)) - len(missingroles))) / i

    # From the K Nearest Neighbours find the best
    # hero in regards to the player in question
    def findbest(self, rating, player):
        newrating = []
        dictionary = {}
        if player == []:
            return int(random.sample(rating, 1)[0][0])
        for hero in player:
            hero = hero.replace('"', '').split(',')
            id = hero[0].split(':')
            dictionary[id[1]] = hero
        for rated in rating:
            hero = dictionary.get(rated[0])
            winrate = 0
            if int(hero[2].split(':')[1]) != 0:
                winrate = float(hero[3].split(':')[1]) / float(hero[2].split(':')[1])
            newrating.append((rated[0], winrate))
        if len(newrating) == 0:
            return int(random.sample(rating, 1)[0][0])
        val = newrating[0][1]
        id = newrating[0][0]
        for rated in newrating[1:]:
            if val < rated[1]:
                id = rated[0]
                val = rated[1]
        return int(id)

    def intersection(self, lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3

    # From the list of allies creates the perfect ally
    # so we can find the hero with most similarity to it
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

    # Given a list of profiles returns the list of roles
    # of every hero
    def extractroles(self, profiles):
        all = self.allroles.copy()
        for profile in profiles:
            for role in profile.Roles:
                role = role.split(",")[-1].split(":")[-1]
                if role in all:
                    all.remove(role)
        return all


class KNNPlayer2(Player):

    def __init__(self, draft, k, distance):
        self.draft = draft
        self.name = 'knn2_' + distance
        with open('models/hero_win_rates.pickle', 'rb') as f:
            self.win_rate_dist = pickle.load(f)
        self.k = k
        self.distance = distance
        self.controllergrid = self.makegrid()

    def makegrid(self):
        directory = os.fsencode("input/Players")
        grid = []
        for i, file in enumerate(os.listdir(directory)):
            if i < i+1:
                controller = open("input/Players/" + str(os.fsdecode(file)), "r").readlines()
                ratings = [0] * 114
                for hero in controller:
                    hero = hero.replace('"', '').replace('\'', "").split(',')
                    id = int(hero[0].split(':')[1])
                    games = int(hero[2].split(':')[1])
                    if games != 0 and id < 114:
                        wins = int(hero[3].split(':')[1])
                        ratings[id] = wins / games
                grid.append(ratings)
        return numpy.array(grid)

    def get_move(self, move_type):
        controller = self.draft.getcontroller()
        potential_controller = copy.deepcopy(controller)
        for i, rate in enumerate(controller):
            if rate == 0 and i not in [0, 24]:
                prediction = self.predict_classification(self.controllergrid, numpy.array(controller), self.k, i)
                potential_controller[i] = prediction
        moves = self.draft.get_moves()
        move_win_rates = [(m, potential_controller[m]) for m in moves]
        best_move, best_win_rate = sorted(move_win_rates, key=lambda x: x[1])[-1]
        return best_move

    # calculate the Euclidean distance between two vectors
    def euclidean_distance(self, row1, row2, predcol):
        distance = 0.0
        for i in range(len(row1)):
            if i != predcol:
                distance += (row1[i] - row2[i]) ** 2
        return math.sqrt(distance)

    # Locate the most similar neighbors
    def get_neighbors(self, train, test_row, num_neighbors, predcol):
        distances = list()
        for train_row in train:
            dist = self.d(test_row, train_row, predcol)
            distances.append((train_row, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        for i in range(num_neighbors):
            neighbors.append(distances[i])
        return neighbors

    # Make a classification prediction with neighbors
    def predict_classification(self, train, test_row, num_neighbors, predcol):
        neighbors = self.get_neighbors(train, test_row, num_neighbors, predcol)
        prediction = self.pred(test_row, neighbors, predcol)
        return prediction

    def pred(self, a, n, p):
        suma = 0
        for i in range(len(a)):
            if i != p:
                suma += a[i]
        suma = suma / (len(a) - 1)
        addsim = 0
        for naighbor in n:
            addsim += naighbor[1]
        for naighbor in n:
            sumnab = 0
            for i in range(len(naighbor[0])):
                if i != p:
                    sumnab += naighbor[0][i]
            sumnab = sumnab / (len(naighbor[0]) - 1)
            suma += (naighbor[1] / addsim) * (naighbor[0][p] - sumnab)

        return suma

    def d(self, test_row, train_row, predcol):
        if self.distance == "euclid":
            return self.euclidean_distance(test_row, train_row, predcol)
        elif self.distance == "manhatten":
            return self.manhatten_distance(test_row, train_row, predcol)
        elif self.distance == "cosd":
            return self.CosD(test_row, train_row, predcol)
        elif self.distance == "scd":
            return self.SCD(test_row, train_row, predcol)
        elif self.distance == "sed":
            return self.SED(test_row, train_row, predcol)
        elif self.distance == "kdd":
            return self.KDD(test_row, train_row, predcol)
        elif self.distance == "vwhd":
            return self.VWHD(test_row, train_row, predcol)
        else:
            raise NotImplementedError

    def manhatten_distance(self, row1, row2, predcol):
        distance = 0.0
        for i in range(len(row1)):
            if i != predcol:
                distance += abs(row1[i] - row2[i])
        return distance

    def CosD(self, row1, row2, predcol):
        distance = 0
        top = 0
        row1_squrd = 0
        row2_squrd = 0
        for i in range(len(row1)):
            if i != predcol:
                top += row1[i]*row2[i]
                row1_squrd += row1[i]**2
                row2_squrd += row2[i]**2
                if row1_squrd == 0 or row2_squrd == 0:
                    distance += 1
                else:
                    distance += top/(math.sqrt(row1_squrd)*math.sqrt(row2_squrd))
        return distance

    def SCD(self, row1, row2, predcol):
        distance = 0.0
        for i in range(len(row1)):
            if i != predcol:
                distance += (math.sqrt(row1[i]) - math.sqrt(row2[i])) ** 2
        return distance

    def SED(self, row1, row2, predcol):
        distance = 0.0
        for i in range(len(row1)):
            if i != predcol:
                distance += (row1[i] - row2[i]) ** 2
        return distance

    def KDD(self, row1, row2, predcol):
        distance = 0.0
        for i in range(len(row1)):
            divider = row1[i] + row2[i]
            top = 2 * row1[i]
            if i != predcol and divider != 0 and top != 0:
                distance += abs(row1[i] * math.log2(top / divider))
        return distance

    def VWHD(self, row1, row2, predcol):
        distance = 0.0
        for i in range(len(row1)):
            divider = min(row1[i], row2[i])
            if i != predcol:
                if divider != 0:
                    distance += abs(row1[i]-row2[i]/divider)
                else:
                    distance += 1
        return distance


class MatrixFactorizationWinratePlayer(Player):
    def __init__(self, draft):
        self.draft = draft
        self.name = 'mfw'
        """if os.path.exists('mf.pickle'):
            self.nmf = self.getMatrix()
        else:
            self.nmf = self.startMatrix()
            self.nmf = self.getMatrix()"""
        self.nmf = self.startMatrix()
        self.nmfp = self.getPlayers()

    def startMatrix(self):
        snmf = startNormalWinrateMatrixFac()
        return snmf.start()

    def getPlayers(self):
        snmf = startNormalWinrateMatrixFac()
        return snmf.getPlayers()

    def getMatrix(self):
        with open('mf.pickle', 'rb') as f:
            matrix = pickle.load(f)
        return matrix

    def getPlayerRed(self, movecount):
        players = [3, 4, 7, 8, 10]
        player = 0
        for i in range(0, 10):
            if movecount == players[i]:
                if players[i] == 3:
                    return 6
                elif players[i] == 4:
                    return 7
                elif players[i] == 7:
                    return 8
                elif players[i] == 8:
                    return 9
                elif players[i] == 10:
                    return 10
        return player

    def getPlayerBlue(self, movecount):
        players = [3, 4, 7, 8, 10]
        player = 0
        for i in range(0, 10):
            if movecount == players[i]:
                if players[i] == 3:
                    return 1
                elif players[i] == 4:
                    return 2
                elif players[i] == 7:
                    return 3
                elif players[i] == 8:
                    return 4
                elif players[i] == 10:
                    return 5
        return player

    def getBestId(self, nmf, nmfp, player, listOfMax):
        maxval = 0
        id = -1
        i = 0
        for x in range(0, len(nmf[player])):
            if nmf[player][x] > maxval and nmf[player][x] not in listOfMax:
                maxval = nmf[player][x]
                i = x
        #if nmfp[player][i][0][1] in self.draft.avail_moves:
        if nmfp[player][i][0] in self.draft.avail_moves:
            #id = nmfp[player][i][0][1]
            id = int(nmfp[player][i][0])
        else:
            listOfMax += [maxval]
            maxval = 0
            id = self.getBestId(nmf, nmfp, player, listOfMax)
        return id

    def get_move(self, move_type):
        if move_type == 'ban':
            return self.getBanMove()
        else:
            return self.getBestMove()

    def getBestMove(self):
        moveCount = self.draft.move_cnt
        player = 0
        team = self.draft.next_player

        if team == 0:
            """Red Team"""
            player = self.getPlayerRed(moveCount[0])
        elif team == 1:
            """Blue Team"""
            player = self.getPlayerBlue(moveCount[1])

        if player == 1:
            return self.getBestId(self.nmf, self.nmfp, 0, [])
        elif player == 2:
            return self.getBestId(self.nmf, self.nmfp, 1, [])
        elif player == 3:
            return self.getBestId(self.nmf, self.nmfp, 2, [])
        elif player == 4:
            return self.getBestId(self.nmf, self.nmfp, 3, [])
        elif player == 5:
            return self.getBestId(self.nmf, self.nmfp, 4, [])
        elif player == 6:
            return self.getBestId(self.nmf, self.nmfp, 5, [])
        elif player == 7:
            return self.getBestId(self.nmf, self.nmfp, 6, [])
        elif player == 8:
            return self.getBestId(self.nmf, self.nmfp, 7, [])
        elif player == 9:
            return self.getBestId(self.nmf, self.nmfp, 8, [])
        elif player == 10:
            return self.getBestId(self.nmf, self.nmfp, 9, [])

    def getBanMove(self):
        player = 0
        team = self.draft.next_player

        if team == 0:
            """Red Team"""
            player = random.randint(1, 5)
        elif team == 1:
            """Blue Team"""
            player = random.randint(6, 10)

        if player == 1:
            return self.getBestId(self.nmf, self.nmfp, 0, [])
        elif player == 2:
            return self.getBestId(self.nmf, self.nmfp, 1, [])
        elif player == 3:
            return self.getBestId(self.nmf, self.nmfp, 2, [])
        elif player == 4:
            return self.getBestId(self.nmf, self.nmfp, 3, [])
        elif player == 5:
            return self.getBestId(self.nmf, self.nmfp, 4, [])
        elif player == 6:
            return self.getBestId(self.nmf, self.nmfp, 5, [])
        elif player == 7:
            return self.getBestId(self.nmf, self.nmfp, 6, [])
        elif player == 8:
            return self.getBestId(self.nmf, self.nmfp, 7, [])
        elif player == 9:
            return self.getBestId(self.nmf, self.nmfp, 8, [])
        elif player == 10:
            return self.getBestId(self.nmf, self.nmfp, 9, [])


class MatrixFactorizationThresholdPlayer(Player):
    def __init__(self, draft):
        self.draft = draft
        self.name = 'mfth'
        """if os.path.exists('mfth.pickle'):
            self.tmf = self.getMatrix()
        else:
            self.tmf = self.startMatrix()
            self.tmf = self.getMatrix()"""
        #self.tmf = self.startMatrix()
        #self.tmf = self.getMatrix()
        self.tmfp = self.getPlayers()

    def get_move(self, move_type):
        if move_type == 'ban':
            return self.getBanMove()
        else:
            return self.getBestMove()

    def startMatrix(self):
        stmf = startTresholdMatrixFac()
        return stmf.start()

    def getMatrix(self):
        with open('mfth.pickle', 'rb') as f:
            matrix = pickle.load(f)
        return matrix

    def getPlayers(self):
        stmf = startTresholdMatrixFac()
        return stmf.getPlayers()

    def getRandomID(self, x, y):
        return random.randint(x, y)

    def getPlayerRed(self, movecount):
        players = [3, 4, 7, 8, 10]
        player = 0
        for i in range(0, 10):
            if movecount == players[i]:
                if players[i] == 3:
                    return 6
                elif players[i] == 4:
                    return 7
                elif players[i] == 7:
                    return 8
                elif players[i] == 8:
                    return 9
                elif players[i] == 10:
                    return 10
        return player

    def getPlayerBlue(self, movecount):
        players = [3, 4, 7, 8, 10]
        player = 0
        for i in range(0, 10):
            if movecount == players[i]:
                if players[i] == 3:
                    return 1
                elif players[i] == 4:
                    return 2
                elif players[i] == 7:
                    return 3
                elif players[i] == 8:
                    return 4
                elif players[i] == 10:
                    return 5
        return player

    def getBestId(self, nmf, nmfp, player, listOfID):
        i = []
        for x in range(0, len(nmf[player])):
            if round(nmf[player][x]) == 1 and nmf[player][x] not in listOfID:
                i.append(x)

        maxsize = len(i)
        id = self.getRandomID(0, maxsize)
        if nmfp[player][id][0] in self.draft.avail_moves:
            playerid = int(nmfp[player][id][0])
        else:
            listOfID += [id]
            playerid = self.getBestId(nmf, nmfp, player, listOfID)
        return playerid

    def getBestMove(self):
        moveCount = self.draft.move_cnt
        player = 0
        team = self.draft.next_player

        if team == 0:
            """Red Team"""
            player = self.getPlayerRed(moveCount[0])
        elif team == 1:
            """Blue Team"""
            player = self.getPlayerBlue(moveCount[1])

        if player == 1:
            return self.getBestId(self.tmf, self.tmfp, 0, [])
        elif player == 2:
            return self.getBestId(self.tmf, self.tmfp, 1, [])
        elif player == 3:
            return self.getBestId(self.tmf, self.tmfp, 2, [])
        elif player == 4:
            return self.getBestId(self.tmf, self.tmfp, 3, [])
        elif player == 5:
            return self.getBestId(self.tmf, self.tmfp, 4, [])
        elif player == 6:
            return self.getBestId(self.tmf, self.tmfp, 5, [])
        elif player == 7:
            return self.getBestId(self.tmf, self.tmfp, 6, [])
        elif player == 8:
            return self.getBestId(self.tmf, self.tmfp, 7, [])
        elif player == 9:
            return self.getBestId(self.tmf, self.tmfp, 8, [])
        elif player == 10:
            return self.getBestId(self.tmf, self.tmfp, 9, [])

    def getBanMove(self):
        player = 0
        team = self.draft.next_player

        if team == 0:
            """Red Team"""
            player = random.randint(1, 5)
        elif team == 1:
            """Blue Team"""
            player = random.randint(6, 10)

        if player == 1:
            return self.getBestId(self.tmf, self.tmfp, 0, [])
        elif player == 2:
            return self.getBestId(self.tmf, self.tmfp, 1, [])
        elif player == 3:
            return self.getBestId(self.tmf, self.tmfp, 2, [])
        elif player == 4:
            return self.getBestId(self.tmf, self.tmfp, 3, [])
        elif player == 5:
            return self.getBestId(self.tmf, self.tmfp, 4, [])
        elif player == 6:
            return self.getBestId(self.tmf, self.tmfp, 5, [])
        elif player == 7:
            return self.getBestId(self.tmf, self.tmfp, 6, [])
        elif player == 8:
            return self.getBestId(self.tmf, self.tmfp, 7, [])
        elif player == 9:
            return self.getBestId(self.tmf, self.tmfp, 8, [])
        elif player == 10:
            return self.getBestId(self.tmf, self.tmfp, 9, [])
