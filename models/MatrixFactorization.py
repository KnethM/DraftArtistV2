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
            if self.R[i, j] > 0
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
        return mf.b + mf.b_u[:, np.newaxis] + mf.b_i[np.newaxis:, ] + mf.P.dot(mf.Q.T)


def randomValue(x, y):
    return randint(x, y)


def importPlayerBlue():
    playerlist = []
    with open("/Users/frederik/Documents/GitHub/DraftArtistV2/input/blue/Player1.txt", "r") as p1:
        inputstring = p1.read()
        player1red = inputstring.split("\n")
        playerlist.append(player1red)
    with open("/Users/frederik/Documents/GitHub/DraftArtistV2/input/blue/Player2.txt", "r") as p2:
        inputstring = p2.read()
        player2red = [inputstring.split("\n")]
        playerlist.append(player2red)
    with open("/Users/frederik/Documents/GitHub/DraftArtistV2/input/blue/Player3.txt", "r") as p3:
        inputstring = p3.read()
        player3red = [inputstring.split("\n")]
        playerlist.append(player3red)
    with open("/Users/frederik/Documents/GitHub/DraftArtistV2/input/blue/Player4.txt", "r") as p4:
        inputstring = p4.read()
        player4red = [inputstring.split("\n")]
        playerlist.append(player4red)
    with open("/Users/frederik/Documents/GitHub/DraftArtistV2/input/blue/Player5.txt", "r") as p5:
        inputstring = p5.read()
        player5red = [inputstring.split("\n")]
        playerlist.append(player5red)
    return playerlist


def importPlayerRed():
    playerlist = []
    with open("/Users/frederik/Documents/GitHub/DraftArtistV2/input/red/Player1.txt", "r") as p1:
        inputstring = p1.read()
        player1blue = inputstring.split("\n")
        playerlist.append(player1blue)
    with open("/Users/frederik/Documents/GitHub/DraftArtistV2/input/red/Player2.txt", "r") as p2:
        inputstring = p2.read()
        player2blue = [inputstring.split("\n")]
        playerlist.append(player2blue)
    with open("/Users/frederik/Documents/GitHub/DraftArtistV2/input/red/Player3.txt", "r") as p3:
        inputstring = p3.read()
        player3blue = [inputstring.split("\n")]
        playerlist.append(player3blue)
    with open("/Users/frederik/Documents/GitHub/DraftArtistV2/input/red/Player4.txt", "r") as p4:
        inputstring = p4.read()
        player4blue = [inputstring.split("\n")]
        playerlist.append(player4blue)
    with open("/Users/frederik/Documents/GitHub/DraftArtistV2/input/red/Player5.txt", "r") as p5:
        inputstring = p5.read()
        player5blue = [inputstring.split("\n")]
        playerlist.append(player5blue)
    return playerlist


playerlistRed = importPlayerBlue()
playerlistBlue = importPlayerRed()


playerlist = [playerlistBlue, playerlistRed]


def spiltPlayerData(list1):
    list2 = []
    if len(list1) == 117:
        for n in list1:
            list2.append(n.split(','))
    elif len(list1) == 1:
        for n in list1[0]:
            list2.append(n.split(','))
    return list2


def spiltPlayerData2(list1):
    list3 = []
    for n in list1:
        list3.append(n.split(':'))
    return list3


playerlist2 = []
playerlist3 = []
for i in range(0, len(playerlist[0])):
    playerlist2.append(spiltPlayerData(playerlist[0][i]))
for x in range(0, len(playerlist[1])):
    playerlist2.append(spiltPlayerData(playerlist[0][x]))
for y in range(0, len(playerlist2)):
    for z in range(0, len(playerlist2[y])):
        playerlist3.append(spiltPlayerData2(playerlist2[y][z]))

n = 117
playerlist3 = [playerlist3[i * n:(i + 1) * n] for i in range((len(playerlist3) + n - 1) // n)]

player1blue = playerlist3[0]
player2blue = playerlist3[1]
player3blue = playerlist3[2]
player4blue = playerlist3[3]
player5blue = playerlist3[4]

player1red = playerlist3[5]
player2red = playerlist3[6]
player3red = playerlist3[7]
player4red = playerlist3[8]
player5red = playerlist3[9]

playerListComplete = [player1blue, player2blue, player3blue, player4blue, player5blue, player1red, player2red,
                      player3red, player4red, player5red]

x = 0
i = 0
y = 0
z = 0


def winrateSort(objt):
    objt.sort(key=lambda y: y[9])
    objt.reverse()
    return objt


def calWinRate(player):
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
    for z in range(0, len(player2)):
        winrateSort(player2[z])
    return player2


playerListSorted = calWinRate(playerListComplete)

charNotAbleToPick = [0, 24, 114, 119, 120, 121, 129]


def getRandom(num1, num2, num3):
    return randint(num1, num2, num3)


values1 = [playerListSorted[0][0][0][1], playerListSorted[0][1][0][1], playerListSorted[0][2][0][1],
           playerListSorted[0][3][0][1], playerListSorted[0][4][0][1], playerListSorted[0][5][0][1],
           playerListSorted[0][6][0][1], playerListSorted[0][7][0][1], playerListSorted[0][8][0][1],
           playerListSorted[0][9][0][1], playerListSorted[0][10][0][1], playerListSorted[0][11][0][1],
           playerListSorted[0][12][0][1], playerListSorted[0][13][0][1], playerListSorted[0][14][0][1],
           playerListSorted[0][15][0][1], playerListSorted[0][16][0][1], playerListSorted[0][17][0][1],
           playerListSorted[0][18][0][1], playerListSorted[0][19][0][1]]

values2 = [playerListSorted[1][0][0][1], playerListSorted[1][1][0][1], playerListSorted[1][2][0][1],
           playerListSorted[1][3][0][1], playerListSorted[1][4][0][1], playerListSorted[1][5][0][1],
           playerListSorted[1][6][0][1], playerListSorted[1][7][0][1], playerListSorted[1][8][0][1],
           playerListSorted[1][9][0][1], playerListSorted[1][10][0][1], playerListSorted[1][11][0][1],
           playerListSorted[1][12][0][1], playerListSorted[1][13][0][1], playerListSorted[1][14][0][1],
           playerListSorted[1][15][0][1], playerListSorted[1][16][0][1], playerListSorted[1][17][0][1],
           playerListSorted[1][18][0][1], playerListSorted[1][19][0][1]]

values3 = [playerListSorted[2][0][0][1], playerListSorted[2][1][0][1], playerListSorted[2][2][0][1],
           playerListSorted[2][3][0][1], playerListSorted[2][4][0][1], playerListSorted[2][5][0][1],
           playerListSorted[2][6][0][1], playerListSorted[2][7][0][1], playerListSorted[2][8][0][1],
           playerListSorted[2][9][0][1], playerListSorted[2][10][0][1], playerListSorted[2][11][0][1],
           playerListSorted[2][12][0][1], playerListSorted[2][13][0][1], playerListSorted[2][14][0][1],
           playerListSorted[2][15][0][1], playerListSorted[2][16][0][1], playerListSorted[2][17][0][1],
           playerListSorted[2][18][0][1], playerListSorted[2][19][0][1]]

values4 = [playerListSorted[3][0][0][1], playerListSorted[3][1][0][1], playerListSorted[3][2][0][1],
           playerListSorted[3][3][0][1], playerListSorted[3][4][0][1], playerListSorted[3][5][0][1],
           playerListSorted[3][6][0][1], playerListSorted[3][7][0][1], playerListSorted[3][8][0][1],
           playerListSorted[3][9][0][1], playerListSorted[3][10][0][1], playerListSorted[3][11][0][1],
           playerListSorted[3][12][0][1], playerListSorted[3][13][0][1], playerListSorted[3][14][0][1],
           playerListSorted[3][15][0][1], playerListSorted[3][16][0][1], playerListSorted[3][17][0][1],
           playerListSorted[3][18][0][1], playerListSorted[3][19][0][1]]

values5 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

R = np.array([
    [values1[0], values1[1], values1[2], values1[3], values1[4], values1[5], values1[6], values1[7], values1[8],
     values1[9], values1[10], values1[11], values1[12], values1[13], values1[14], values1[15], values1[16], values1[17],
     values1[18], values1[19]],
    [values2[0], values2[1], values2[2], values2[3], values2[4], values2[5], values2[6], values2[7], values2[8],
     values2[9], values2[10], values2[11], values2[12], values2[13], values2[14], values2[15], values2[16], values2[17],
     values2[18], values2[19]],
    [values3[0], values3[1], values3[2], values3[3], values3[4], values3[5], values3[6], values3[7], values3[8],
     values3[9], values3[10], values3[11], values3[12], values3[13], values3[14], values3[15], values3[16], values3[17],
     values3[18], values3[19]],
    [values4[0], values4[1], values4[2], values4[3], values4[4], values4[5], values4[6], values4[7], values4[8],
     values4[9], values4[10], values4[11], values4[12], values4[13], values4[14], values4[15], values4[16], values4[17],
     values4[18], values4[19]],
    [values5[0], values5[1], values5[2], values5[3], values5[4], values5[5], values5[6], values5[7], values5[8],
     values5[9], values5[10], values5[11], values5[12], values5[13], values5[14], values5[15], values5[16], values5[17],
     values5[18], values5[19]]
])

mf = MF(R, K=10, alpha=0.01, beta=0.01, iterations=100)
training_process = mf.train()
print()
print("P x Q:")
print(mf.full_matrix())
print()
print("Global bias:")
print(mf.b)
print()
print("User bias:")
print(mf.b_u)
print()
print("Item bias:")
print(mf.b_i)

print("Full Matrix")
print(mf.full_matrix())

test = mf.full_matrix()
test1 = test[4]
test11 = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]


def lowerValue(x, count):
    for y in range(1, len(test11)):
        if int(test1[x] - y) not in test11 and int(test1[x] - y) > 0 and count == 0:
            return int(test1[x]) - y


def higherValue(x, count):
    for y in range(1, len(test11)):
        if int(test1[x] + y) not in test11 and int(test1[x] + y) < 113 and count == 0:
            return int(test1[x]) + y


for x in range(0, len(test1)):
    count = 0
    if int(test1[x]) not in test11 and int(test1[x]) < 113:
        test11[x] = int(test1[x])
    elif int(test1[x]) == 112 and int(test1[x]) in test11:
        test11[x] = lowerValue(x, count)
        count = 1
    elif int(test1[x]) in test11:
        test11[x] = higherValue(x, count)
        count = 1

print()
print("Test")
print(test11)

"""
R = np.array([
    [values1[0], values1[1], values1[2], values1[3], values1[4], values1[5], values1[6], values1[7], values1[8],
     values1[9], values1[10], values1[11], values1[12], values1[13], values1[14], values1[15], values1[16], values1[17],
     values1[18], values1[19]],
    [values2[0], values2[1], values2[2], values2[3], values2[4], values2[5], values2[6], values2[7], values2[8],
     values2[9], values2[10], values2[11], values2[12], values2[13], values2[14], values2[15], values2[16], values2[17],
     values2[18], values2[19]],
    [values3[0], values3[1], values3[2], values3[3], values3[4], values3[5], values3[6], values3[7], values3[8],
     values3[9], values3[10], values3[11], values3[12], values3[13], values3[14], values3[15], values3[16], values3[17],
     values3[18], values3[19]],
    [values4[0], values4[1], values4[2], values4[3], values4[4], values4[5], values4[6], values4[7], values4[8],
     values4[9], values4[10], values4[11], values4[12], values4[13], values4[14], values4[15], values4[16], values4[17],
     values4[18], values4[19]],
    [values5[0], values5[1], values5[2], values5[3], values5[4], values5[5], values5[6], values5[7], values5[8],
     values5[9], values5[10], values5[11], values5[12], values5[13], values5[14], values5[15], values5[16], values5[17],
     values5[18], values5[19]]
])






[values1[0], values1[1], values1[2], values1[3], values1[4], values1[5], values1[6], values1[7], values1[8],
     values1[9], values1[10], values1[11], values1[12], values1[13], values1[14], values1[15], values1[16], values1[17],
     values1[18], values1[19], values1[20], values1[21], values1[22], values1[23], values1[24], values1[25],
     values1[26], values1[27], values1[28], values1[29], values1[30], values1[31], values1[32], values1[33],
     values1[34], values1[35], values1[36], values1[37], values1[38], values1[39], values1[40], values1[41],
     values1[42], values1[43], values1[44], values1[45], values1[46], values1[47], values1[48], values1[49],
     values1[50], values1[51], values1[52], values1[53], values1[54], values1[55], values1[56], values1[57],
     values1[58], values1[59], values1[60], values1[61], values1[62], values1[63], values1[64], values1[65],
     values1[66], values1[67], values1[68], values1[69], values1[70], values1[71], values1[72], values1[73],
     values1[74], values1[75], values1[76], values1[77], values1[78], values1[79], values1[80], values1[81],
     values1[82], values1[83], values1[84], values1[85], values1[86], values1[87], values1[88], values1[89],
     values1[90], values1[91], values1[92], values1[93], values1[94], values1[95], values1[96], values1[97],
     values1[98], values1[99], values1[100], values1[101], values1[102], values1[103], values1[104], values1[105],
     values1[106], values1[107], values1[108], values1[109], values1[110], values1[111], values1[112], values1[113]]
"""
