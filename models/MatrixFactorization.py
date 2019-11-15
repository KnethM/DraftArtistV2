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


values1 = randint(0, 113, 20)
print(values1)
values2 = randint(0, 113, 20)
print(values2)
values3 = randint(0, 113, 20)
print(values3)
values4 = randint(0, 113, 20)
print(values4)
values5 = randint(0, 113, 20)
print(values5)
values6 = randint(0, 113, 20)
print(values6)
values7 = randint(0, 113, 20)
print(values7)
values8 = randint(0, 113, 20)
print(values8)
values9 = randint(0, 113, 20)
print(values9)
values10 = randint(0, 113, 20)
print(values10)

R = np.array([
    [values1[0], values1[1], values1[2], values1[3], values1[4], values1[5], values1[6],
     values1[7], values1[8], values1[9], values1[10], values1[11], values1[12], values1[13],
     values1[14], values1[15], values1[16], values1[17], values1[18], values1[19]],
    [values2[0], values2[1], values2[2], values2[3], values2[4], values2[5], values2[6],
     values2[7], values2[8], values2[9], values2[10], values2[11], values2[12], values2[13],
     values2[14], values2[15], values2[16], values2[17], values2[18], values2[19]],
    [values3[0], values3[1], values3[2], values3[3], values3[4], values3[5], values3[6],
     values3[7], values3[8], values3[9], values3[10], values3[11], values3[12], values3[13],
     values3[14], values3[15], values3[16], values3[17], values3[18], values3[19]],
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
"""
    [111, 0, 0, 11, 0, 112, 0, 0, 91, 41, 31, 71, 0, 0, 17, 21, 0, 61, 0, 1],
    [111, 0, 0, 11, 0, 112, 0, 0, 91, 41, 31, 71, 0, 0, 17, 21, 0, 61, 0, 1],

    [1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1],
    
    [1, 0, 3, 4, 5, 6, 0, 8, 9, 10],
    [1, 2, 0, 4, 5, 6, 7, 0, 9, 10],
    [1, 2, 3, 0, 5, 6, 7, 8, 0, 10],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 0],
    [0, 2, 3, 4, 5, 0, 7, 8, 9, 10],
    [1, 0, 3, 4, 5, 6, 0, 8, 9, 10],
    [1, 2, 0, 4, 5, 6, 7, 0, 9, 10],
    [1, 2, 3, 0, 5, 6, 7, 8, 0, 10],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 0],
    [0, 2, 0, 4, 0, 6, 0, 8, 0, 10],
    
    [10, 50, 0, 71, 0, 12, 100, 0, 19, 104, 67, 0, 86, 0, 7, 45, 9, 0, 2, 0],
"""
