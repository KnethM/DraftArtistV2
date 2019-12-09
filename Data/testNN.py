import numpy as np
import pickle as pp
import Data.kfolds as kfs
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import models.MLPclass as mlpc

def testNN():

    #Load in our pickled data.
    trainingdata = np.load('Trainingdata.npy')
    solutiondata = np.load('Solutiondata.npy')

    #Run kfolds function to get out 10-kfolds for our training and testing data.
    trainingdata, testingdata = kfs.kfolds(trainingdata)
    s_trainingdata, s_testingdata = kfs.kfolds(solutiondata)

    f = open("Score.txt", "w")
    f2 = open("Auc.txt", "w")

    #For each of our folds we load in the appropriate NN and calculate the score or mean accuracy
    for index in range(0,10):
        mlpload = pp.load(open('NN'+ str(index) +'_1layer.pickle', "rb"))
        mlpload2 = pp.load(open('NN'+ str(index) +'_2layer.pickle', "rb"))

        # Save the NN as our variable, position 1 holds an integer about the number of heroes in the game.
        classifier = mlpload[0]
        classifier2 = mlpload2[0]

        #Use sk-learn function to calculate the accuracy
        score = classifier.score(testingdata[index], s_testingdata[index])
        score2 = classifier2.score(testingdata[index], s_testingdata[index])

        print("NN"+str(index)+"_1layer: ",score)
        print("NN"+str(index)+"_2layer: ",score2)

        f.write("NN"+str(index)+"_1layer: "+str(score) + '\n')
        f.write("NN"+str(index)+"_2layer: "+str(score2) + '\n')


        #Use the NN to predict the answer as a probability
        probs1 = classifier.predict_proba(testingdata[index])
        probs2 = classifier2.predict_proba(testingdata[index])

        #Make sure we only get the second column as we only want the positive values.
        probs1 = probs1[:, 1]
        probs2 = probs2[:, 1]

        #Calculate the auc score using sk-learn function
        auc1 = roc_auc_score(s_testingdata[index], probs1)
        auc2 = roc_auc_score(s_testingdata[index], probs2)

        f2.write('NN'+str(index)+'_1layer AUC: %.5f' % auc1 + '\n')
        f2.write('NN'+str(index)+'_2layer AUC: %.5f' % auc2 + '\n')

        print('NN'+str(index)+'_1layer AUC: %.5f' % auc1)
        print('NN'+str(index)+'_2layer AUC: %.5f' % auc2)

        #Get the roc_curve and save the information in fpr(False positive rate) and tpr(True positive rate) using sk-learn function
        fpr1, tpr1, thresholds1 = roc_curve(s_testingdata[index], probs1)
        fpr2, tpr2, thresholds2 = roc_curve(s_testingdata[index], probs2)

        #We plot the roc curve in a diagram.
        plot_roc_curve(fpr1, tpr1, index)
        plot_roc_curve2(fpr2, tpr2, index)

#Function used to plot the roc_curve
def plot_roc_curve(fpr, tpr, index):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve:' + 'NN'+str(index)+'_1layer')
    plt.legend()
    plt.show()

def plot_roc_curve2(fpr, tpr, index):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve:' + 'NN' + str(index)+'_2layer')
    plt.legend()
    plt.show()

    #for n in range(0,50000):
    #    b = classifier.predict(testingdata[0])
    #    print("Predict:", b[n], "Actual:", s_testingdata[0][n])



def learningcurve():
    mlpload = pp.load(open('Fold6_1layer.pickle', "rb"))
    mlpload2 = mlpc.classifier.Dotaclf1layer
    trainsize = [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    trainingdata = np.load('Trainingdata.npy')
    solutiondata = np.load('Solutiondata.npy')

    train_sizes, train_scores, validation_scores = learning_curve(mlpload[0], trainingdata, solutiondata,
                                                                  train_sizes=trainsize, cv=5, scoring='accuracy')


    print(train_scores)
    print(validation_scores)

    train_scores_mean = train_scores.mean(axis=1)
    validation_scores_mean = validation_scores.mean(axis=1)

    print(train_scores_mean)
    print(validation_scores_mean)

    plt.style.use('seaborn')
    plt.plot(train_sizes, train_scores_mean, label='Training score')
    plt.plot(train_sizes, validation_scores_mean, label='Cross-validation score')
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    plt.title('Learning curves for trained MLPClassifier', fontsize=18, y=1.03)
    plt.legend()
    plt.ylim(0, 1)
    plt.show()

def getweights():

    mlpload = pp.load(open('Fold6_1layer.pickle', "rb"))
    classifier = mlpload[0]

    #the structure of coefs is number of layers -1, so coefs_[0] gives us the input layer
    #[0][0] gives
    weights = classifier.coefs_[1][0]

    print(weights)


if __name__ == '__main__':
    #Run this if you want to test the accuracy and score of the trained neural networks.
    #testNN()

    #Run this if you want the learningcurve for a Neural network classifier
    learningcurve()

    #Run this to get the weights between neurons in a neural network.
    #getweights()