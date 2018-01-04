import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split

print "importing data..."
X = np.genfromtxt('../Data/train_mice15bin.csv', delimiter = ',')
Y = np.genfromtxt('../Data/train_Y.csv', delimiter = ',')
test_features = np.genfromtxt('../Data/test_mice15bin.csv', delimiter = ',')
print "data imported..."


X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)


print "training..."
#C = 10 with rbf gives best results so far
clf = SVC(C = 10, gamma = 10**-2, kernel = 'rbf') #default rbf will overfit
clf.fit(X_train, Y_train)
print "testing..."

y_preds = clf.predict(X_val)
print f1_score(Y_val, y_preds, average='macro')



# y_pred = clf.predict(test_features)

# print "saving..."
# ids = range(0,test_features.shape[0])
# stre = np.append(ids, y_pred).reshape(2, test_features.shape[0]).T
# np.savetxt('../results/svm_mice1700PCA_c10_g10e-3_rbf.csv', stre, delimiter = ',', fmt = '%d')

print '\a'
