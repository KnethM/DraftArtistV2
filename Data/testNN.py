import numpy as np
import pickle as pp
import Data.kfolds as kfs
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

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



if __name__ == '__main__':
    testNN()