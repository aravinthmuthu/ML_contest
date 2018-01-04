import numpy as np
import sklearn
import pandas as pd
import matplotlib as plt
from fancyimpute import IterativeSVD

print("reading data...")

X = pd.read_csv("Data/train.csv").iloc[:,1:-1].as_matrix() # remove first row (labels), first and last columns
test = pd.read_csv("Data/test.csv").iloc[:,1:].as_matrix()
# ind = np.genfromtxt('Class_change_ind.csv', delimiter = ',', dtype = 'int32')

# test_incomplete = test.copy()

print X.shape

print("setting svd object...")
svd = IterativeSVD(rank = 1000, convergence_threshold=0.0001)

X_svd = svd.complete(X);
# print X_svd[:,0]


print("saving data...")
np.savetxt("Data/train_isvdimp.csv", X_svd, delimiter=",")

print("imputing test...")
test_svd = svd.complete(test)

np.savetxt("Data/test_isvdimp.csv", test_svd, delimiter=",")
print '\a'
