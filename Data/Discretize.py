import numpy as np
import pickle as pp
from models import MLPclass as mlp

# Method for discretizing the data even further by cleaning it up and abstracting the numbers.
def discretize():

    f = open("DiscretizedData.txt", "r")

    fl = f.readlines()

    # Lists used to store the final information about matches
    # x contains hero picks and winrates, y contains radiant win or loss
    x = []
    y = []

    for match in fl:
        templist = []

        matchstring = removefromstring(match, "[],'")

        # Go through the list and replace any occurence of "Null" with "0" then split the string into a list
        matchstring = matchstring.replace("Null", "0")
        matchlist = matchstring.split()

        #append the outcome for the single match to the array out outcomes
        y.append(matchlist[1])

        # list of indexes to delete
        # go through the list on index numbers and delete it in reverse order.
        indexes = [1, 2, 3, 4, 5, 6, 17, 18, 19, 20, 21]
        for index in sorted(indexes, reverse=True):
            del matchlist[index]

        # go through the list and convert each string to an integer
        for str in matchlist:
            templist.append(int(str))

        #numpy array of 0's with size 123 (113 heroes + 10 player winrates)
        finallist = np.zeros((123))

        # go through the list on index numbers and set their value to 1 for radiant team hero picks
        # and -1 for dire team hero picks.
        findex1 = [templist[1], templist[2], templist[3], templist[4], templist[5]]
        findex2 = [templist[11], templist[12], templist[13], templist[14], templist[15]]
        for index in findex1:
            finallist[index] = 1
        for index in findex2:
            finallist[index] = -1

        # go through the list on index numbers and set their value to be equal the playerwinrate divided by 100.
        # indexcounter set to 113 as we only wanna change index 113 to 123
        tempindex = [templist[6], templist[7], templist[8], templist[9], templist[10], templist[16], templist[17],
                      templist[18], templist[19], templist[20]]
        indexcounter = 113
        for index in tempindex:
            finallist[indexcounter] = (index/100)
            indexcounter += 1

        # append the cleaned up information for a match to list of matches
        x.append(finallist)

    y = indextoint(y)

    # trains the classifier on the data and places the trained classifier in mlpc
    mlpc = mlp.classifier.Dotaclf1layer.fit(x,y)
    mlpclist = []
    mlpclist.append(mlpc)
    mlpclist.append(113)

    pp.dump(mlpclist, open('mlp.pickle', "wb"))

    mlpload = pp.load(open('mlp.pickle', "rb"))
    print(mlpload)

    # Tries to predict the outcome of an input
    #a = mlpc.predict_proba(x[255].reshape(1,-1))[0,1]
    #print(a)
 
# Går gennem listen og ændre alle "true" til 1 og "false" til 0
def indextoint(list):
    newlist = []
    for n in list:
        if n == "true":
            newlist.append(1)
        else:
            newlist.append(0)
    return newlist

def removefromstring(match, characters):

    # Go through the list and remove any occurencess of [] , '
    trantab = match.maketrans("", "", characters)
    cleanedstring = match.translate(trantab)
    return cleanedstring

if __name__ == '__main__':
    discretize()