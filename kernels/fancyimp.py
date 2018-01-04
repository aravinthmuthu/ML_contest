import numpy as np
import sklearn
import pandas as pd
import matplotlib as plt
from fancyimpute import (
    BiScaler,
    KNN,
    NuclearNormMinimization,
    SoftImpute,
    SimpleFill
)

print("reading data...")

X = pd.read_csv("Data/train.csv").iloc[:,1:-1].as_matrix() # remove first row (labels), first and last columns 
test = pd.read_csv("Data/test.csv").iloc[:,1:].as_matrix()


test_incomplete = test.copy()


X1 = X[:1565,:]
X2 = X[1565:2809,:]
X3 = X[2809:4434,:]
X4 = X[4434:6106,:]
X5 = X[6106:7655,:]
X6 = X[7655:,:]





print("seting knn object...")
knnImpute = KNN(k=3)


print("imputing X1 ...")
X1_knn = knnImpute.complete(X1)

print("imputing X2 ...")
X2_knn = knnImpute.complete(X2)

print("imputing X3...")
X3_knn = knnImpute.complete(X3)

print("imputing X4 ...")
X4_knn = knnImpute.complete(X4)

print("imputing X5 ...")
X5_knn = knnImpute.complete(X5)

print("imputing X6 ...")
X6_knn = knnImpute.complete(X6)

X_knn = np.concatenate((X1_knn, X2_knn, X3_knn, X4_knn, X5_knn, X6_knn), axis=0)
print (X_knn.shape)

print("saving data...")
np.savetxt("Data/train_knnimp_full.csv", X_knn, delimiter=",")

print("imputing test...")
test_filled_knn = knnImpute.complete(test_incomplete)

np.savetxt("Data/test_knnimp.csv", test_filled_knn, delimiter=",")
print '\a'

