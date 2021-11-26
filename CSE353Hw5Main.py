import math
import random

import numpy as  np
import matplotlib.pyplot as plt

##Assume that X as a list of lists where x = [[1, x, y], [1, x, y], ...] unless otherwise stated
##simple dot product between the w vector and another vector x
def dotProduct(w, x):
    total = 0
    for i in range(len(w)):
        total += w[i] * x[i]
    return total
##calculates the norm squared of a gradient vector
def normSquared(N, vec):
    total = 0
    for i in range(len(vec)):
        total += vec[i]/N
    return total
##The linear Regression method
def linearReg(x, y):
    transposeX = np.transpose(x);
    return np.linalg.inv(transposeX * x) * transposeX * y
##The logistic regression method
def logisticReg(w, x, y, iterations ,N ,a, e):
    returnedW = [0,0,0]
    for i in range(iterations):
        for j in range(len(w)):
            sigma = logisticRegSigma(w, x, y, N)
            if(normSquared(N, sigma) <= e):
                break
            w[j] += (a * 1/N * sigma[j])
    return w
##helper method to find sigma
def logisticRegSigma(w, x, y, N):
    returned = [0,0,0]
    if(N == len(x)):
        for i in range(N):
            point = x[i]
            sigmaValue = (1/(1 + np.exp(-y[i] * dotProduct(point, w))))
            for j in range(len(point)):
                returned[j] = sigmaValue * (point[j] * y[i])
        return returned
    for i in range(N):
        randomIndex = random.randint(0, len(x)-1)
        point = x[randomIndex]
        sigmaValue = (1 / (1 + np.exp(-y[randomIndex] * dotProduct(point, w))))
        for j in range(len(point)):
            returned[j] = sigmaValue * (point[j] * y[randomIndex])
    return returned
##gets the list of y boolean truth values for PLA
def getListOfY(input):
    tempY = []
    for i in range(len(input)):
        tempY.extend(input[i].split(','))
        tempY = [x.replace('\n', '') for x in tempY]
    for i in range(len(tempY)):
        tempY[i] = float(tempY[i])
    return tempY
##creates the the list of lists x from the input list of strings for PLA
def getListOfX(input, stride, pointSize):
    tempX = []
    for i in range(len(input)):
        tempX.extend(input[i].split(','))
        tempX = [x.replace('\n', '') for x in tempX]
    for i in range(len(tempX)):
        tempX[i] = float(tempX[i])
    return groupX(tempX, stride, pointSize)

##helper method for getListOfX which groups the sub lists together forming each individual [1, x, y, ...]
def groupX(list, stride, pointSize):
    returnedX = []
    for i in range(stride):
        tempXPoint = []
        for j in range(pointSize):
            tempXPoint.append(list[i + j*stride])
        returnedX.append(tempXPoint)
    return returnedX
##Sets a number to 1 while all others are set to -1
def setYNumberAs1(y, number):
    tempY = []
    for i in range(len(y)):
        if y[i] == number:
            tempY.append(1)
        else:
            tempY.append(-1)
    return tempY
##gets the x values for plotting
def getXPoints(x, y, truth):
    tempX = []
    for i in range(len(x)):
        if(y[i] == truth):
            tempX.append(x[i][1])
    return tempX
##Gets the y values for plotting
def getYPoints(x, y, truth):
    tempY = []
    for i in range(len(x)):
        if (y[i] == truth):
            tempY.append(x[i][2])
    return tempY
##gets wither the largest x value or smallest value
def getExtremeX(x, largest):
    largestX = 0
    smallestX = 0
    for i in range(len(x)):
        point = x[i]
        if point[1] > largestX:
            largestX = point[1]
        if point[1] < smallestX:
            smallestX = point[1]
    if largest:
        return largestX
    return smallestX
##gets the y values from the w linear line from pla
def getYFromW(w, x):
    return -1 * ((w[0] + (w[1] * x))/w[2])
##checks the error of one vs all algorithm
def checkwError(x, y, wlist, incorrectList):
    total = 0
    for i in range(len(x)):
        point = x[i]
        if dotProduct(wlist[int(y[i]-1)], point) < 0:
            total += 1
            incorrectList.append((point[1], point[2]))
        else:
            for j in range(len(wlist)):
                if not j == y[i]-1 and dotProduct(wlist[j], point) > 0:
                    total += 1
                    incorrectList.append((point[1], point[2]))
    return total
##removes the unwanted points from the list
def removeExtras(x, y, keep1, keep2):
    tempX = []
    tempY = []
    for i in range(len(x)):
        if y[i] == keep1 or y[i] == keep2:
            tempX.append(x[i])
            tempY.append(y[i])
    return (tempX, tempY)
##checks the error of one vs one
def checkErrorOvo(x, y, wLists, incorrectList):
    total = 0
    for i in range(len(x)):
        votesList = [0, 0, 0, 0]
        point = x[i]
        maxIndex = 0
        for j in range(len(wLists)):
            if dotProduct(wLists[j][0], point) > 0:
                votesList[wLists[j][1] - 1] += 1
            else:
                votesList[wLists[j][2] - 1] += 1
        for k in range(len(votesList)):
            if votesList[k] > votesList[maxIndex]:
                maxIndex = k
        if not maxIndex+1 == y[i]:
            total += 1
            incorrectList.append((point[1], point[2]))
    return total

#load the data from the directories
xDir = 'X.txt'
yDir = 'Y.txt'
with open(xDir) as f:
    hw5x = f.readlines()
    f.close()
with open(yDir) as f:
    hw5y = f.readlines()
    f.close()
#get the original list with numbers 1,2,3,4
listY = getListOfY(hw5y)
#one vs all lists with the number being 1 and all the rest are -1
listY1vall = setYNumberAs1(listY, 1)
listY2vall = setYNumberAs1(listY, 2)
listY3vall = setYNumberAs1(listY, 3)
listY4vall = setYNumberAs1(listY, 4)
matY1vall = np.transpose(np.matrix(np.array(listY1vall)))
matY2vall = np.transpose(np.matrix(np.array(listY2vall)))
matY3vall = np.transpose(np.matrix(np.array(listY3vall)))
matY4vall = np.transpose(np.matrix(np.array(listY4vall)))
#get the list of x values
listX = getListOfX(hw5x, 80, 3)
matX = np.matrix(np.array(listX))
#get all the linear regression for the
linw1vall = np.transpose(linearReg(matX, matY1vall)).tolist()[0]
linw2vall = np.transpose(linearReg(matX, matY2vall)).tolist()[0]
linw3vall = np.transpose(linearReg(matX, matY3vall)).tolist()[0]
linw4vall = np.transpose(linearReg(matX, matY4vall)).tolist()[0]
#using logistic regression for the 4 w values
logw1vall = logisticReg(linw1vall, listX, listY1vall, 500, len(listX), 0.005, .1)
logw2vall = logisticReg(linw2vall, listX, listY2vall, 500, len(listX), 0.005, .1)
logw3vall = logisticReg(linw3vall, listX, listY3vall, 500, len(listX), 0.005, .1)
logw4vall = logisticReg(linw4vall, listX, listY4vall, 500, len(listX), 0.005, .1)
incorrectList = []
incorrectListX = []
incorrectListY = []
totalIncorrectova  = checkwError(listX, listY, [logw1vall, logw2vall, logw3vall, logw4vall], incorrectList)
print("Error for one vs all is " + str(totalIncorrectova/ 80))
for i in range(len(incorrectList)):
    incorrectListX.append(incorrectList[i][0])
    incorrectListY.append(incorrectList[i][1])
#get the lists of points for plotting
listX1 = getXPoints(listX, listY, 1)
listY1 = getYPoints(listX, listY, 1)
listX2 = getXPoints(listX, listY, 2)
listY2 = getYPoints(listX, listY, 2)
listX3 = getXPoints(listX, listY, 3)
listY3 = getYPoints(listX, listY, 3)
listX4 = getXPoints(listX, listY, 4)
listY4 = getYPoints(listX, listY, 4)

largestX = getExtremeX(listX, True)
smallestX = getExtremeX(listX, False)

plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('One Vs All Regression')
plt.plot(listX1, listY1, 'yD')
plt.plot(listX2, listY2, 'gs')
plt.plot(listX3, listY3, 'b*')
plt.plot(listX4, listY4, 'c^')
plt.plot(incorrectListX, incorrectListY, 'rx')
plt.plot([smallestX, largestX], [getYFromW(logw1vall, smallestX), getYFromW(logw1vall, largestX)], "y-")
plt.plot([smallestX, largestX], [getYFromW(logw2vall, smallestX), getYFromW(logw2vall, largestX)], "g-")
plt.plot([smallestX, largestX], [getYFromW(logw3vall, smallestX), getYFromW(logw3vall, largestX)], "b-")
plt.plot([smallestX, largestX], [getYFromW(logw4vall, smallestX), getYFromW(logw4vall, largestX)], "c-")
##plt.show()


##Start of the second question
#creates tuples (X, Y) of the lists where X and Y are lists of the points and truth values
tuple1vs2 = removeExtras(listX, listY, 1, 2)
tuple1vs3 = removeExtras(listX, listY, 1, 3)
tuple1vs4 = removeExtras(listX, listY, 1, 4)
tuple2vs3 = removeExtras(listX, listY, 2, 3)
tuple2vs4 = removeExtras(listX, listY, 2, 4)
tuple3vs4 = removeExtras(listX, listY, 3, 4)
#Get the linear regression values
linw1vs2 = np.transpose(linearReg(np.matrix(np.array(tuple1vs2[0])), np.transpose(np.matrix(np.array(setYNumberAs1(tuple1vs2[1], 1)))))).tolist()[0]
linw1vs3 = np.transpose(linearReg(np.matrix(np.array(tuple1vs3[0])), np.transpose(np.matrix(np.array(setYNumberAs1(tuple1vs3[1], 1)))))).tolist()[0]
linw1vs4 = np.transpose(linearReg(np.matrix(np.array(tuple1vs4[0])), np.transpose(np.matrix(np.array(setYNumberAs1(tuple1vs4[1], 1)))))).tolist()[0]
linw2vs3 = np.transpose(linearReg(np.matrix(np.array(tuple2vs3[0])), np.transpose(np.matrix(np.array(setYNumberAs1(tuple2vs3[1], 2)))))).tolist()[0]
linw2vs4 = np.transpose(linearReg(np.matrix(np.array(tuple2vs4[0])), np.transpose(np.matrix(np.array(setYNumberAs1(tuple2vs4[1], 2)))))).tolist()[0]
linw3vs4 = np.transpose(linearReg(np.matrix(np.array(tuple3vs4[0])), np.transpose(np.matrix(np.array(setYNumberAs1(tuple3vs4[1], 3)))))).tolist()[0]
#Get the w values
w1vs2 = logisticReg(linw1vs2, tuple1vs2[0], tuple1vs2[1], 500, len(tuple1vs2[0]), .005, .1)
w1vs3 = logisticReg(linw1vs3, tuple1vs3[0], tuple1vs3[1], 500, len(tuple1vs3[0]), .005, .1)
w1vs4 = logisticReg(linw1vs4, tuple1vs4[0], tuple1vs4[1], 500, len(tuple1vs4[0]), .005, .1)
w2vs3 = logisticReg(linw2vs3, tuple2vs3[0], tuple2vs3[1], 500, len(tuple2vs3[0]), .005, .1)
w2vs4 = logisticReg(linw2vs4, tuple2vs4[0], tuple2vs4[1], 500, len(tuple2vs4[0]), .005, .1)
w3vs4 = logisticReg(linw3vs4, tuple3vs4[0], tuple3vs4[1], 500, len(tuple3vs4[0]), .005, .1)
#Loads the w values into a single list in the form of (w, a, b) where if a point dot w = 1 then
#it will classify as a and it will be classified as b otherwise
wList = [(w1vs2, 1, 2), (w1vs3, 1, 3), (w1vs4, 1, 4), (w2vs3, 2, 3), (w2vs4, 2, 4), (w3vs4, 3, 4)]
incorrectList = []
incorrectListX = []
incorrectListY = []
incorrectTotal = checkErrorOvo(listX, listY, wList, incorrectList)
print("One vs one error is " + str(incorrectTotal/80))
for i in range(len(incorrectList)):
    incorrectListX.append(incorrectList[i][0])
    incorrectListY.append(incorrectList[i][1])
plt.subplot(1, 2, 2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('One Vs One Regression')
plt.plot(listX1, listY1, 'yD')
plt.plot(listX2, listY2, 'gs')
plt.plot(listX3, listY3, 'b*')
plt.plot(listX4, listY4, 'c^')
plt.plot(incorrectListX, incorrectListY, 'rx')
plt.show()
