import numpy as np
import os
import pickle
from datetime import datetime
from numpy.random import seed
from numpy.random import randint


class MF():

    def __init__(self, R, K, alpha, beta, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter

        https://nbviewer.jupyter.org/github/albertauyeung/matrix-factorization-in-python/blob/master/mf.ipynb
        """

        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.starttime = datetime.now()
        self.starttime = self.starttime.strftime("%H:%M:%S")

    def train(self):
        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1. / self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1. / self.K, size=(self.num_items, self.K))

        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] >= 0
        ]

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            """if (i + 1) % 10 == 0:
                #currenttime = datetime.now()
                #current_time = currenttime.strftime("%H:%M:%S")
                #time = datetime.strptime(current_time, '%H:%M:%S') - datetime.strptime(self.starttime, '%H:%M:%S')
                print("Iteration: %d ; error = %.4f" % (i + 1, mse))
                #print("Time =", time)
            if (i + 1) % 200 == 0:
                currenttime = datetime.now()
                current_time = currenttime.strftime("%H:%M:%S")
                time = datetime.strptime(current_time, '%H:%M:%S') - datetime.strptime(self.starttime, '%H:%M:%S')
                print("Time after 200 =", time)"""
        return training_process

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            # Update biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            # Create copy of row of P since we need to update it but use older values for update on Q
            P_i = self.P[i, :][:]

            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i, :])
            self.Q[j, :] += self.alpha * (e * P_i - self.beta * self.Q[j, :])

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        return self.b + self.b_u[:, np.newaxis] + self.b_i[np.newaxis:, ] + self.P.dot(self.Q.T)


class importPlayers():
    def importAllPlayers(self):
        directory = os.fsencode("input/Players")
        grid = []
        for file in os.listdir(directory):
            with open("input/Players/" + str(os.fsdecode(file)), "r") as file:
                controller = file.readlines()
                ratings = [0] * 114
                for hero in controller:
                    hero = hero.replace('"', '').replace('\'', "").split(',')
                    id = int(hero[0].split(':')[1])
                    games = int(hero[2].split(':')[1])
                    if games != 0 and id < 114 and id != 24 and id != 0:
                        wins = int(hero[3].split(':')[1])
                        ratings[id] = [id, wins / games]
                    elif games == 0 and id < 114 and id != 24 and id != 0:
                        ratings[id] = [id, games]
                ratings.remove(0)
                ratings.remove(0)
                grid.append(ratings)
        return np.array(grid)

    def importPlayerBlue(self):
        playerlist = []
        with open("input/blue/Player1.txt", "r") as p1:
            inputstring = p1.read()
            player1red = inputstring.split("\n")
            playerlist.append(player1red)
        with open("input/blue/Player2.txt", "r") as p2:
            inputstring = p2.read()
            player2red = [inputstring.split("\n")]
            playerlist.append(player2red)
        with open("input/blue/Player3.txt", "r") as p3:
            inputstring = p3.read()
            player3red = [inputstring.split("\n")]
            playerlist.append(player3red)
        with open("input/blue/Player4.txt", "r") as p4:
            inputstring = p4.read()
            player4red = [inputstring.split("\n")]
            playerlist.append(player4red)
        with open("input/blue/Player5.txt", "r") as p5:
            inputstring = p5.read()
            player5red = [inputstring.split("\n")]
            playerlist.append(player5red)
        return playerlist

    def importPlayerRed(self):
        playerlist = []
        with open("input/red/Player1.txt", "r") as p1:
            inputstring = p1.read()
            player1blue = inputstring.split("\n")
            playerlist.append(player1blue)
        with open("input/red/Player2.txt", "r") as p2:
            inputstring = p2.read()
            player2blue = [inputstring.split("\n")]
            playerlist.append(player2blue)
        with open("input/red/Player3.txt", "r") as p3:
            inputstring = p3.read()
            player3blue = [inputstring.split("\n")]
            playerlist.append(player3blue)
        with open("input/red/Player4.txt", "r") as p4:
            inputstring = p4.read()
            player4blue = [inputstring.split("\n")]
            playerlist.append(player4blue)
        with open("input/red/Player5.txt", "r") as p5:
            inputstring = p5.read()
            player5blue = [inputstring.split("\n")]
            playerlist.append(player5blue)
        return playerlist

    def getPlayerlist(self):
        return [self.importPlayerBlue(), self.importPlayerRed()]

    def spiltPlayerData(self, list1):
        list2 = []
        if len(list1) == 117:
            for n in list1:
                list2.append(n.split(','))
        elif len(list1) == 1:
            for n in list1[0]:
                list2.append(n.split(','))
        return list2

    def spiltPlayerData2(self, list1):
        list3 = []
        for n in list1:
            list3.append(n.split(':'))
        return list3

    def splitPlayerData3(self):
        playerlist = self.getPlayerlist()
        playerlist2 = []
        playerlist3 = []
        for i in range(0, len(playerlist[0])):
            playerlist2.append(self.spiltPlayerData(playerlist[0][i]))
        for x in range(0, len(playerlist[1])):
            playerlist2.append(self.spiltPlayerData(playerlist[0][x]))
        for y in range(0, len(playerlist2)):
            for z in range(0, len(playerlist2[y])):
                playerlist3.append(self.spiltPlayerData2(playerlist2[y][z]))

        n = 117
        return [playerlist3[i * n:(i + 1) * n] for i in range((len(playerlist3) + n - 1) // n)]

    def playerlistComplete(self):
        return [self.splitPlayerData3()[0], self.splitPlayerData3()[1], self.splitPlayerData3()[2],
                self.splitPlayerData3()[3], self.splitPlayerData3()[4], self.splitPlayerData3()[5],
                self.splitPlayerData3()[6], self.splitPlayerData3()[7], self.splitPlayerData3()[8],
                self.splitPlayerData3()[9]]

    def calWinRate(self, player):
        player2 = []
        n = 117
        for p in range(0, len(player)):
            for x in range(0, 117):
                if int(player[p][x][3][1]) != 0 and int(player[p][x][2][1]) != 0:
                    winrate = [int(player[p][x][3][1]) / int(player[p][x][2][1])]
                    player2 += [player[p][x] + [winrate]]
                else:
                    player2 += [player[p][x] + [[0]] + [[0]]]
        for o in range(0, len(player2)):
            player2[o][0][1] = player2[o][0][1][1:-1]
            player2[o][0][1] = int(player2[o][0][1])
        player2 = [player2[i * n:(i + 1) * n] for i in range((len(player2) + n - 1) // n)]
        return player2

    def getRedAndBlue(self):
        return self.calWinRate(self.playerlistComplete())

    def getPlayerlistSorted(self):
        return self.importAllPlayers()


class normalMatrixFac():
    ip = importPlayers()
    playerListSorted = ip.getPlayerlistSorted()
    playersRedAndBlue = ip.getRedAndBlue()

    #maxsize = int(len(playerListSorted))
    maxsize = 200
    blueteam = []
    redteam = []
    team = []

    "Team - Normal Winrate"
    for i in range(0, maxsize):
        players = [0] * 114
        for x in range(0, len(playerListSorted[i])):
            players[x] = playerListSorted[i][x][1]
        team.append(players)

    "Blue Team - Normal Winrate"
    for i in range(0, 5):
        players = [0] * 114
        for x in range(0, 113):
            players[x] = playersRedAndBlue[i][x][9][0]
        blueteam.append(players)

    "Red Team - Normal Winrate"
    for i in range(5, 10):
        players = [0] * 114
        for x in range(0, 113):
            players[x] = playersRedAndBlue[i][x][9][0]
        redteam.append(players)

    "Winrate normal"

    def getListOfCharacters(self):
        players = []
        players += self.blueteam
        players += self.redteam
        players += self.team
        return np.array(players)


class thresholdMatrixFac():
    ip = importPlayers()
    playerListSorted = ip.getPlayerlistSorted()
    playersRedAndBlue = ip.getRedAndBlue()

    #maxsize = int(len(playerListSorted))
    maxsize = 0
    blueteam = []
    redteam = []
    team = []

    """Team - Winrate with threshold"""
    threshold = 0.5

    def setThreshold(self, value):
        self.threshold = value

    "Team - Winrate with threshold"
    for i in range(0, maxsize):
        players = [0] * 114
        for x in range(0, len(playerListSorted[i])):
            if playerListSorted[i][x][1] > threshold:
                players[x] = 1
            else:
                players[x] = 0
        team.append(players)

    "Blue Team - Winrate with threshold"
    for i in range(0, 5):
        players = [0] * 114
        for x in range(0, 113):
            if playersRedAndBlue[i][x][9][0] > threshold:
                players[x] = 1
            else:
                players[x] = 0
        blueteam.append(players)

    "Red Team - Winrate with threshold"
    for i in range(5, 10):
        players = [0] * 114
        for x in range(0, 113):
            if playersRedAndBlue[i][x][9][0] > threshold:
                players[x] = 1
            else:
                players[x] = 0
        redteam.append(players)

    "Winrate over threshold x with value 50%"

    def getListOfCharacters(self):
        players = []
        players += self.blueteam
        players += self.redteam
        players += self.team
        return np.array(players)


"Winrate normal"


class startNormalWinrateMatrixFac():
    nmf = normalMatrixFac()
    pl = importPlayers()

    def start(self):
        mf = MF(self.nmf.getListOfCharacters(), K=10, alpha=0.01, beta=0.01, iterations=200)
        training_process = mf.train()
        matrix = mf.full_matrix()
        with open('mf.pickle', 'wb') as f:
            pickle.dump(matrix, f)
        return mf.full_matrix()

    def getMatrix(self):
        with open('mf.pickle', 'rb') as f:
            matrix = pickle.load(f)
        return matrix

    def getPlayers(self):
        return self.pl.getPlayerlistSorted()


"Winrate with threshold x"


class startTresholdMatrixFac():
    thmf = thresholdMatrixFac()
    thmf.setThreshold(0.60)
    pl = importPlayers()

    def start(self):
        mfth = MF(self.thmf.getListOfCharacters(), K=10, alpha=0.01, beta=0.01, iterations=200)
        training_process_th = mfth.train()
        matrix = mfth.full_matrix()
        with open('mfth.pickle', 'wb') as f:
            pickle.dump(matrix, f)
        return mfth.full_matrix()

    def getPlayers(self):
        return self.pl.getPlayerlistSorted()

