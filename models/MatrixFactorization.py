import numpy as np
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
            if (i + 1) % 10 == 0:
                print("Iteration: %d ; error = %.4f" % (i + 1, mse))

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
    def importPlayerBlue(self):
        playerlist = []
        with open("../input/blue/Player1.txt", "r") as p1:
            inputstring = p1.read()
            player1red = inputstring.split("\n")
            playerlist.append(player1red)
        with open("../input/blue/Player2.txt", "r") as p2:
            inputstring = p2.read()
            player2red = [inputstring.split("\n")]
            playerlist.append(player2red)
        with open("../input/blue/Player3.txt", "r") as p3:
            inputstring = p3.read()
            player3red = [inputstring.split("\n")]
            playerlist.append(player3red)
        with open("../input/blue/Player4.txt", "r") as p4:
            inputstring = p4.read()
            player4red = [inputstring.split("\n")]
            playerlist.append(player4red)
        with open("../input/blue/Player5.txt", "r") as p5:
            inputstring = p5.read()
            player5red = [inputstring.split("\n")]
            playerlist.append(player5red)
        return playerlist

    def importPlayerRed(self):
        playerlist = []
        with open("../input/red/Player1.txt", "r") as p1:
            inputstring = p1.read()
            player1blue = inputstring.split("\n")
            playerlist.append(player1blue)
        with open("../input/red/Player2.txt", "r") as p2:
            inputstring = p2.read()
            player2blue = [inputstring.split("\n")]
            playerlist.append(player2blue)
        with open("../input/red/Player3.txt", "r") as p3:
            inputstring = p3.read()
            player3blue = [inputstring.split("\n")]
            playerlist.append(player3blue)
        with open("../input/red/Player4.txt", "r") as p4:
            inputstring = p4.read()
            player4blue = [inputstring.split("\n")]
            playerlist.append(player4blue)
        with open("../input/red/Player5.txt", "r") as p5:
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

    def getPlayerlistSorted(self):
        return self.calWinRate(self.playerlistComplete())




"List of characters that is not suitable for picks"
charNotAbleToPick = [0, 24, 114, 119, 120, 121, 129]


class normalMatrixFac():
    ip = importPlayers()
    playerListSorted = ip.getPlayerlistSorted()
    "Blue Team - Normal Winrate"
    values1 = []
    for v1 in range(0, 113):
        values1.append(playerListSorted[0][v1][9][0])

    values2 = []
    for v2 in range(0, 113):
        values2.append(playerListSorted[1][v2][9][0])

    values3 = []
    for v3 in range(0, 113):
        values3.append(playerListSorted[2][v3][9][0])

    values4 = []
    for v4 in range(0, 113):
        values4.append(playerListSorted[3][v4][9][0])

    values5 = []
    for v5 in range(0, 113):
        values5.append(playerListSorted[4][v5][9][0])

    "Red Team - Normal Winrate"
    values6 = []
    for v6 in range(0, 113):
        values6.append(playerListSorted[5][v6][9][0])

    values7 = []
    for v7 in range(0, 113):
        values7.append(playerListSorted[6][v7][9][0])

    values8 = []
    for v8 in range(0, 113):
        values8.append(playerListSorted[7][v8][9][0])

    values9 = []
    for v9 in range(0, 113):
        values9.append(playerListSorted[8][v9][9][0])

    values10 = []
    for v10 in range(0, 113):
        values10.append(playerListSorted[9][v10][9][0])

    "Set values for testing - Normal Winrate"
    values1[0] = -1
    values1[12] = -1
    values1[53] = -1
    values1[82] = -1

    values10[5] = -1
    values10[64] = -1
    values10[110] = -1

    "Winrate normal"

    def getListOfCharacters(self):
        return np.array([self.values1, self.values2, self.values3, self.values4, self.values5, self.values6,
                         self.values7, self.values8, self.values9, self.values10])

    def getValues(self, value):
        if value == 1:
            return self.values1
        elif value == 2:
            return self.values2
        elif value == 3:
            return self.values3
        elif value == 4:
            return self.values4
        elif value == 5:
            return self.values5
        elif value == 6:
            return self.values6
        elif value == 7:
            return self.values7
        elif value == 8:
            return self.values8
        elif value == 9:
            return self.values9
        elif value == 10:
            return self.values10


class thresholdMatrixFac():
    ip = importPlayers()
    playerListSorted = ip.getPlayerlistSorted()
    """Blue Team - Winrate with threshold"""
    threshold = 0.5

    def setThreshold(self, value):
        self.threshold = value

    values11 = []
    for v11 in range(0, 113):
        if playerListSorted[0][v11][9][0] > threshold:
            values11.append(1)
        else:
            values11.append(0)

    values22 = []
    for v22 in range(0, 113):
        if playerListSorted[1][v22][9][0] > threshold:
            values22.append(1)
        else:
            values22.append(0)

    values33 = []
    for v33 in range(0, 113):
        if playerListSorted[2][v33][9][0] > threshold:
            values33.append(1)
        else:
            values33.append(0)

    values44 = []
    for v44 in range(0, 113):
        if playerListSorted[3][v44][9][0] > threshold:
            values44.append(1)
        else:
            values44.append(0)

    values55 = []
    for v55 in range(0, 113):
        if playerListSorted[4][v55][9][0] > threshold:
            values55.append(1)
        else:
            values55.append(0)

    "Red team - Winrate with threshold"
    values66 = []
    for v66 in range(0, 113):
        if playerListSorted[5][v66][9][0] > threshold:
            values66.append(1)
        else:
            values66.append(0)

    values77 = []
    for v77 in range(0, 113):
        if playerListSorted[6][v77][9][0] > threshold:
            values77.append(1)
        else:
            values77.append(0)

    values88 = []
    for v88 in range(0, 113):
        if playerListSorted[7][v88][9][0] > threshold:
            values88.append(1)
        else:
            values88.append(0)

    values99 = []
    for v99 in range(0, 113):
        if playerListSorted[8][v99][9][0] > threshold:
            values99.append(1)
        else:
            values99.append(0)

    values110 = []
    for v110 in range(0, 113):
        if playerListSorted[9][v110][9][0] > threshold:
            values110.append(1)
        else:
            values110.append(0)

    values11[0] = -1
    values11[12] = -1
    values11[53] = -1
    values11[82] = -1
    valuetest1 = [0, 12, 53, 82]

    values110[5] = -1
    values110[64] = -1
    values110[110] = -1
    valuetest2 = [5, 64, 110]

    def getValueTest(self, value):
        if value == 1:
            return self.valuetest1
        elif value == 2:
            return self.valuetest2

    "Winrate over threshold x with value 50%"

    def getListOfCharacters(self):
        return np.array([self.values11, self.values22, self.values33, self.values44, self.values55, self.values66,
                         self.values77, self.values88, self.values99, self.values110])


"Winrate normal"


class startNormalWinrateMatrixFac():
    nmf = normalMatrixFac()

    def start(self):
        mf = MF(self.nmf.getListOfCharacters(), K=10, alpha=0.01, beta=0.01, iterations=1000)
        training_process = mf.train()
        print()
        print("Full Matrix")
        print(mf.full_matrix())
        return mf.full_matrix()

    def getValues(self, value):
        return self.nmf.getValues(value)


"Winrate with threshold x"


class startTresholdMatrixFac():
    def start(self):
        thmf = thresholdMatrixFac()
        thmf.setThreshold(0.60)
        mfth = MF(thmf.getListOfCharacters(), K=10, alpha=0.01, beta=0.01, iterations=1000)
        training_process_th = mfth.train()
        print()
        print("Full Matrix")
        print(mfth.full_matrix())
        return mfth.full_matrix()


def printTestResultnormal(test, val1, val2):
    ip = importPlayers()
    playerListSorted = ip.getPlayerlistSorted()
    test1 = test[0]
    test2 = test[9]

    error1 = 0
    count1 = 0
    for u in range(0, len(test1)):
        dif = test1[u] - val1[u]
        if dif > 0 and val1[u] != 0:
            count1 += 1
            error1 += dif
        elif dif < 0 and val1[u] != 0:
            count1 += 1
            error1 += -dif

    error1 = error1 / count1

    error2 = 0
    count2 = 0
    for t in range(0, len(test2)):
        dif = test2[t] - val2[t]
        if dif > 0 and val2[t] != 0:
            count2 += 1
            error2 += dif
        elif dif < 0 and val2[t] != 0:
            count2 += 1
            error2 += -dif
    error2 = error2 / count2

    test11 = [[test1[0], playerListSorted[0][0][9][0]], [test1[12], playerListSorted[0][12][9][0]],
              [test1[53], playerListSorted[0][53][9][0]], [test1[82], playerListSorted[0][82][9][0]]]
    test22 = [[test2[5], playerListSorted[9][5][9][0]], [test2[64], playerListSorted[9][64][9][0]],
              [test2[110], playerListSorted[9][110][9][0]]]
    print()
    print("Player Blue")
    print(test11)
    print(error1)
    print()
    print("Player Red")
    print(test22)
    print(error2)


def printTestResult(test):
    ip = importPlayers()
    playerListSorted = ip.getPlayerlistSorted()
    thmf = thresholdMatrixFac()
    test1 = test[0]
    test2 = test[9]
    for t in range(0, len(test1)):
        test1[t] = round(test1[t])
    test1 = test1.astype(int)
    for r in range(0, len(test2)):
        test2[r] = round(test2[r])
    test2 = test2.astype(int)

    win1 = 0
    for u in range(0, len(test1)):
        q = round(playerListSorted[0][u][9][0])
        if test1[u] == q and u in thmf.getValueTest(1):
            win1 += 1

    win2 = 0
    for e in range(0, len(test2)):
        w = round(playerListSorted[9][e][9][0])
        if test2[e] == w and e in thmf.getValueTest(2):
            win2 += 1

    test11 = [[test1[0], round(playerListSorted[0][0][9][0])], [test1[12], round(playerListSorted[0][12][9][0])],
              [test1[53], round(playerListSorted[0][53][9][0])], [test1[82], round(playerListSorted[0][82][9][0])]]
    test22 = [[test2[5], round(playerListSorted[9][5][9][0])], [test2[64], round(playerListSorted[9][64][9][0])],
              [test2[110], round(playerListSorted[9][110][9][0])]]
    print()
    print("Player Blue")
    print(test11)
    print(win1)
    print()
    print("Player Red")
    print(test22)
    print(win2)


print()
print("Normal matrix winrate")
snwmf = startNormalWinrateMatrixFac()
printTestResultnormal(snwmf.start(), snwmf.getValues(1), snwmf.getValues(10))
print()
print()
print("Threshold matrix")
stmf = startTresholdMatrixFac()
printTestResult(stmf.start())
