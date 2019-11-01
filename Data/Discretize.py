import numpy as np

def discretize():

    f = open("DiscretizedData.txt", "r")

    fl = f.readlines()

    # Lists used to store the final information about matches
    # x contains hero picks and winrates, y contains if radiant won or lost.
    x = []
    y = []

    for match in fl:
        templist = []

        # Go through the list and remove any occurencess of [] , '
        remove = "[],'"
        trantab = match.maketrans("", "", remove)
        mstring = match.translate(trantab)

        # Go through the list and replace any occurence of "Null" with "0" then split the string into a list
        cleanstring = mstring.replace("Null", "0")
        matchlist = cleanstring.split()

        y.append(matchlist[1])

        # list of indexes to delete
        # go through the list on index numbers and delete it in reverse order.
        indexes = [1, 2, 3, 4, 5, 6, 17, 18, 19, 20, 21]
        for index in sorted(indexes, reverse=True):
            del matchlist[index]

        # go through the list and convert each string to an integer
        for str in matchlist:
            templist.append(int(str))

        # make a numpy array of 0's with size 121
        finallist = np.zeros((121))

        # go through the list on index numbers and set their value to 1
        findex = [templist[1], templist[2], templist[3], templist[4], templist[5], templist[11], templist[12],
                  templist[13], templist[14], templist[15]]
        for index in findex:
            finallist[index] = 1

        # finallist[110] til finallist[120] skal sættes lig med  templist[6] til templist[10] og templist[16] til templist[20]
        fuckdether = [templist[6], templist[7], templist[8], templist[9], templist[10], templist[16], templist[17],
                      templist[18], templist[19], templist[20]]
        countminroev = 111
        for sutden in fuckdether:
            finallist[countminroev] = sutden
            countminroev += 1

        x.append(templist)

    yy = indextoint(y)
    print(x)
    print(yy)


def indextoint(list):
    newlist = []
    for n in list:
        if n == "true":
            newlist.append(1)
        else:
            newlist.append(0)
    return newlist