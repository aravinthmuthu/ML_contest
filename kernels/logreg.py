from sklearn import linear_model
import numpy as np
from sklearn.metrics import f1_score


X = np.genfromtxt("Data/train_knnimp_full.csv",delimiter=",")[:,:]
Y = np.reshape(np.genfromtxt("Data/train_Y.csv",delimiter=",", dtype='int32'),(-1,1))
test = np.genfromtxt("Data/test_knnimp.csv",delimiter=",")[:,:]

# print (X.shape)
# print(Y.shape)

# train = np.concatenate((X,Y), axis=1)
# print (train.shape)
# X = train[:,:-1]
# Y = train[:,-1]


print("read")
logreg = linear_model.LogisticRegression(C=0.01) #Increase C to reduce execution time
logreg.fit(X, Y)

# get prediction
print(" testing ")

pred = logreg.predict(test)
ids = range(0,test.shape[0])
stre = np.append(ids, pred).reshape(2, test.shape[0]).T
np.savetxt('Data/samplesub6_knn.csv', stre, delimiter = ',', fmt = '%d')
# 
# print ("pred ")
# print (pred)
# pred = pred.reshape((-1,1))
# pred = pred.astype(np.int64)


# idx = np.arange(test.shape[0]).reshape((-1,1))
# idx= idx.astype(np.int64)

# output = np.concatenate((idx,pred), axis=1)
# print("saving")
# print (pred)
# np.savetxt("results/logreg_1.csv", output.astype(int), fmt='%i', delimiter=",")



