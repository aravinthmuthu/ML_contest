from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np

X = np.genfromtxt("Data/train_imp.csv",delimiter=",")[:,500:]
Y = np.genfromtxt("Data/train_Y.csv",delimiter=",", dtype='int32')
test = np.genfromtxt("Data/test_imp.csv",delimiter=",")[:,500:]
test_Y = np.zeros((test.shape[0],1))

# b

clf = SVC(C = 10, gamma = 10**-3, kernel = 'rbf')

print("read")
model = BaggingClassifier(base_estimator=clf,  max_samples=0.8, n_estimators = 20, n_jobs = -1)
model.fit(X, Y)

# get prediction
print(" testing ")

pred = model.predict(test)
ids = range(0,test.shape[0])
stre = np.append(ids, pred).reshape(2, test.shape[0]).T
np.savetxt('Results/svm_2100ftrs_c10_0.001_rbf_bag20.csv', stre, delimiter = ',', fmt = '%d')
# 