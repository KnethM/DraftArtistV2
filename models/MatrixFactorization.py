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
    """for z in range(0, len(player2)):
        winrateSort(player2[z])"""
    return player2


playerListSorted = calWinRate(playerListComplete)

charNotAbleToPick = [0, 24, 114, 119, 120, 121, 129]

"Blue Team"
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

"Red team"
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

"Set values for testing"
values1[0] = 0
values1[12] = 0
values1[53] = 0
values1[82] = 0

values10[5] = 0
values10[64] = 0
values10[110] = 0


"Winrate normal"
R1 = np.array([values1, values2, values3, values4, values5, values6, values7, values8, values9, values10])

"Winrate over threshold x with value 50%"
R2 = np.array([values1, values2, values3, values4, values5, values6, values7, values8, values9, values10])

"Winrate normalized"
R3 = np.array([values1, values2, values3, values4, values5, values6, values7, values8, values9, values10])

"Winrate normal"
mf = MF(R1, K=10, alpha=0.01, beta=0.01, iterations=20)
training_process = mf.train()
"""print()
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
print(mf.b_i)"""
print("Full Matrix")
print(mf.full_matrix())
"""
"Winrate over threshold x with value 50%"
mf = MF(R1, K=10, alpha=0.01, beta=0.01, iterations=20)
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

"Winrate normalized"
mf = MF(R1, K=10, alpha=0.01, beta=0.01, iterations=20)
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
"""

test = mf.full_matrix()
test1 = test[0]
test2 = test[1]
test11 = [[test1[0], playerListSorted[0][0][9][0]], [test1[12], playerListSorted[0][12][9][0]],
          [test1[53], playerListSorted[0][53][9][0]], [test1[82], playerListSorted[0][82][9][0]]]
test22 = [[test2[5], playerListSorted[9][5][9][0]], [test2[64], playerListSorted[9][64][9][0]],
          [test2[110], playerListSorted[9][110][9][0]]]
print()
print("Player Blue")
print(test11)
print()
print("Player Red")
print(test22)

