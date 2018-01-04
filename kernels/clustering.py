from sklearn.cluster import FeatureAgglomeration
import numpy as np
import pandas as pd

print("reading data...")
X = np.genfromtxt('../Data/train_mice_clubbed.csv',delimiter=",")
Y = np.genfromtxt('../Data/train.csv', delimiter = ',')[1:, -1]
test = np.genfromtxt('../Data/test_mice_clubbed.csv', delimiter = ',')
print "data imported..."

fulldata = np.concatenate((X, test), axis=0)

# agg = FeatureAgglomeration(n_clusters = 2000)
# print ("fitting")
# agg.fit(fulldata)
# print "transform"
# fulldata_agg = agg.transform(fulldata)

first500 = fulldata[:,:500]
second = fulldata[:,500:]

agg = FeatureAgglomeration(n_clusters = 200)
print ("fitting")
agg.fit(first500)
first500_agg = agg.transform(first500)

agg = FeatureAgglomeration(n_clusters=1900)
print ("fitting")
agg.fit(second)
second_agg = agg.transform(second)

# new_X = fulldata[:9501,:]
# new_test = fulldata[9501:,:]

new_X = np.concatenate((first500_agg[:9501,:], second_agg[:9501,:]), axis=1)
new_test = np.concatenate((first500_agg[9501:,:], second_agg[9501:,:]), axis=1)

print("saving data...")
np.savetxt("../Data/train_agg_2100.csv", new_X, delimiter=",")
print("saving data...")
np.savetxt("../Data/test_agg_2100.csv", new_test, delimiter=",")
print '\a'
