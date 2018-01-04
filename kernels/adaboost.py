import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle

X = np.genfromtxt("Data/train_mice2000PCA.csv",delimiter=",")
print(" read X ")
Y = np.genfromtxt("Data/train_Y.csv",delimiter=",", dtype='int32')
test = np.genfromtxt("Data/test_mice2000PCA.csv",delimiter=",")
test_Y = np.zeros((test.shape[0],1))

bdt_discrete = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=7),
    n_estimators=20,
    learning_rate=1.5,
    algorithm="SAMME")
print("fitting ")
bdt_discrete.fit(X, Y)

print("score :", bdt_discrete.score(X,Y))

filename = 'adaboost_model.sav'
pickle.dump(bdt_discrete, open(filename, 'wb'))
pred= bdt_discrete.staged_predict(test)
pred= pred.reshape(-1,1)

pred = pred.reshape((-1,1))
pred = pred.astype(np.int64)

idx = np.arange(test.shape[0]).reshape((-1,1))
idx= idx.astype(np.int64)

output = np.concatenate((idx,pred), axis=1)
np.savetxt("results/adaboost_2000ftrs.csv", output.astype(int), fmt='%i', delimiter=",")