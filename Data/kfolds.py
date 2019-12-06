import numpy as np
from sklearn.model_selection import KFold

def kfolds(data):

    #f = open("DiscretizedData.txt", "r")
    #fl = f.readlines()
    #fl2 = np.array(fl)

    fl2 = data

    kf = KFold(n_splits=10)

    training = []
    testing = []

    for train_index, test_index in kf.split(fl2):
        print("Train:", train_index, "Test:", test_index)
        X_train, X_test = fl2[train_index], fl2[test_index]
        training.append(X_train), testing.append(X_test)

    return training, testing