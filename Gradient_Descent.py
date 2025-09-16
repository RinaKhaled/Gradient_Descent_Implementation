import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random


def LossMSE(predicted_y , y):
    count = 0
    for i  in range(len(y)):
        n = predicted_y[i] - y[i]
        count += n*n
    return (count/(2*len(y)))

def GradientDecent(x_train, y_train, w = [1, 1, 1, 1], learningRate = 0.0001, iterations = 10000):
    x_train.to_numpy()
    cost = []

    for i in range(iterations):
        weight_derivative = np.matmul(x_train.transpose(), (np.matmul(x_train, w) - y_train))
        step = learningRate * weight_derivative

        w = w - step

        y_pred = x_train.dot(w)
        y_pred = np.array(y_pred)
        cost.append(LossMSE(y_pred, y_train))

        if(len(cost) > 3 and cost[-1] > cost[-2]):
            learningRate/=10

    return w

def accuracy_1(predicted_y , y):
    count = 0
    for i  in range(0,len(y)):
        n = predicted_y[i] - y[i]
        count += n*n
    return (count/(len(y)))

def accuracy_2(predicted_y , y):
    return 1 - LossMSE(predicted_y, y)

upperBound = 3000

x1 = []
x2 = []
x3 = []

y = []

for i in range(1000):
    n1 = random.randint(1,upperBound) /upperBound
    x1.append(n1)

    n2 = random.randint(1,upperBound) /upperBound
    x2.append(n2)

    n3 = random.randint(1,upperBound) /upperBound
    x3.append(n3)

    y.append(5*n1 + 3*n2 + 1.5*n3 + 6)

Xfeatures = pd.DataFrame(data = [x1, x2, x3]).transpose()

x_train, x_test, y_train, y_test = train_test_split(Xfeatures, y, test_size=0.33)

#---LEAST_SQURAED---
#X is the matrix of features
X = np.array(x_train)

#Y is the Vector of Labels
Y = np.array(y_train)

#W(a, b) = Xt.Y(Xt.X)-1
Xt = X.transpose()

W = np.matmul((np.linalg.inv(np.matmul(Xt, X))), np.matmul(Xt, Y))

print("Weights:" , W)

#-------------------------

Xtest = np.array(x_test)

Y_Pred = np.matmul(Xtest, W)

print("loss: ",LossMSE(Y_Pred, y_test))
print("accuracy: ",accuracy_1(Y_Pred,y_test))

#---------------------------------------------

x_train['B'] = 1
#print (GradientDecent(x_train, y_train))

#---------------------------------------------
df = pd.read_csv("USA_Housing.csv")

#X is the matrix of features
X = df.iloc[:, 0:4]
#X.iloc[:,0]/= 107701.7484

#Y is the Vector of Labels
Y = df["Price"].to_numpy()
normalizedX=(X-X.min())/(X.max()-X.min())
normalizedY=(Y-Y.min())/(Y.max()-Y.min())
W = GradientDecent(normalizedX, normalizedY)

Y_Pred = np.matmul(normalizedX, W)


print("Weight is: ", W)
print("Loss is: ", LossMSE(Y_Pred,normalizedY))