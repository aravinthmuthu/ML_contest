import numpy as np
import sklearn
import pandas as pd
import matplotlib as plt
from fancyimpute import IterativeSVD, MICE


print("reading data...")

X = pd.read_csv("../Data/train.csv").iloc[:,1:-1].as_matrix() # remove first row (labels), first and last columns
test = pd.read_csv("../Data/test.csv").iloc[:,1:].as_matrix()
# ind = np.genfromtxt('Class_change_ind.csv', delimiter = ',', dtype = 'int32')

# test_incomplete = test.copy()
print X[0]
fulldata = np.concatenate((X, test), axis=0)
fulldata[fulldata == 1.] = np.nan
print fulldata.shape

print("setting MICE object...")
mc = MICE(n_nearest_columns=10, n_imputations=15)

# X_mc = mc.complete(X)
# print X_mc[:,0]
print("imputing data...")
fulldata_mc = mc.complete(fulldata)

print("saving data...")
np.savetxt("../Data/train_mice15bin.csv", fulldata_mc[:9501,:], delimiter=",")

# print("imputing test...")
# test_mc = mc.complete(test)

np.savetxt("../Data/test_mice15bin.csv", fulldata_mc[9501:,:], delimiter=",")
print '\a'
