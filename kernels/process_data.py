import numpy as np
import sklearn
from sklearn.preprocessing import Imputer
import pandas as pd
import matplotlib as plt

X = np.genfromtxt("Data/train.csv",delimiter=",")[1:,1:-1] # remove first row (labels), first and last columns 
Y = np.reshape(np.genfromtxt("Data/train.csv",delimiter=",")[1:,-1], (-1,1))
test  = np.genfromtxt("Data/test.csv",delimiter=",")[1:,1:]

# print("First 5 columns of X")
# print(X[:5,:])

# print("First 5 columns of Y")
# print(Y[:5,:])

# print("First 5 columns of test")
# print(test[:5,:])


imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X)
X=imp.transform(X)
test=imp.transform(test)

# print("After imputation \n\n")

# print("First 5 columns of X")
# print(X[:5,:])

# print("First 5 columns of Y")
# print(Y[:5,:])

# print("First 5 columns of test")
# print(test[:5,:])

np.savetxt("Data/train_imp.csv", X, delimiter=",")
np.savetxt("Data/train_Y.csv", Y, delimiter=",")
np.savetxt("Data/test_imp.csv", test, delimiter=",")


print(X.shape)
print(Y.shape)
print(test.shape)
