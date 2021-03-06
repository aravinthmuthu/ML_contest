from __future__ import division

import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
# label need to be 0 to num_class -1
# data = np.loadtxt('./dermatology.data', delimiter=',',
#         converters={33: lambda x:int(x == '?'), 34: lambda x:int(x)-1})
# sz = data.shape

# train = data[:int(sz[0] * 0.7), :]
# test = data[int(sz[0] * 0.7):, :]

# train_X = train[:, :33]
# train_Y = train[:, 34]

# test_X = test[:, :33]
# test_Y = test[:, 34]

X = np.genfromtxt("Data/train_imp.csv",delimiter=",")
Y = np.genfromtxt("Data/train_Y.csv",delimiter=",", dtype='int32')
test = np.genfromtxt("Data/test_imp.csv",delimiter=",")
test_Y = np.zeros((test.shape[0],1))

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
test = scaler.transform(test)

print(" Read input ")
xg_train = xgb.DMatrix(X, label=Y)
xg_test = xgb.DMatrix(test, label=test_Y)
# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 2
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 29

watchlist = [(xg_train, 'train')]#, (xg_test, 'test')]
num_round = 300
print(" training ")
bst = xgb.train(param, xg_train, num_round, watchlist)
# get prediction
print(" testing ")
pred = bst.predict(xg_test)
print ("pred ")
print (pred)
pred = pred.reshape((-1,1))
pred = pred.astype(np.int64)

idx = np.arange(test.shape[0]).reshape((-1,1))
idx= idx.astype(np.int64)

output = np.concatenate((idx,pred), axis=1)
np.savetxt("results/xgboost_depth2_iter300_0.88train_acc_meanimp.csv", output.astype(int), fmt='%i', delimiter=",")


# error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
# print('Test error using softmax = {}'.format(error_rate))

# # do the same thing again, but output probabilities
# param['objective'] = 'multi:softprob'
# bst = xgb.train(param, xg_train, num_round, watchlist)
# # Note: this convention has been changed since xgboost-unity
# # get prediction, this is in 1D array, need reshape to (ndata, nclass)
# pred_prob = bst.predict(xg_test).reshape(test_Y.shape[0], 6)
# pred_label = np.argmax(pred_prob, axis=1)
# error_rate = np.sum(pred_label != test_Y) / test_Y.shape[0]
# print('Test error using softprob = {}'.format(error_rate))