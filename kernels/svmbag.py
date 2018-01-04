import numpy as np
from sklearn.svm import SVC
def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(Y))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [X[ i] for i in idx]
    labels_shuffle = [Y[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

X = np.genfromtxt('Data/train_mice2000PCA.csv', delimiter = ',')
Y = np.genfromtxt('Data/train_Y.csv', delimiter = ',')
test_features = np.genfromtxt('Data/test_mice2000PCA.csv', delimiter = ',')

for i in np.arange(20):
	print i
	batch_X, batch_Y = next_batch(8000, X, Y)
	clf = SVC(C = 10, gamma = 10**-2, kernel = 'rbf') #default rbf will overfit
	clf.fit(batch_X, batch_Y)
	print "testing..."
	y_pred = clf.predict(test_features)

	print "saving..."
	ids = range(0,test_features.shape[0])
	stre = np.append(ids, y_pred).reshape(2, test_features.shape[0]).T
	np.savetxt("results/SVM_bag80/svm_mice2000PCA_c10_g10e-2_rbf"+str(i)+".csv", stre, delimiter = ',', fmt = '%d')
