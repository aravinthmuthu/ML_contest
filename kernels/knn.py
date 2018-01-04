from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split
import numpy as np

X = np.genfromtxt("Data/train_mice1700PCA.csv",delimiter=",")
Y = np.genfromtxt("Data/train_Y.csv",delimiter=",", dtype='int32')
# X_test = np.genfromtxt("Data/test_mice1700PCA.csv",delimiter=",")[:,500:]
print("read data")

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
k=20
# for k in range(2,9,2):
neigh = KNeighborsClassifier(n_neighbors=k, weights='distance')
neigh.fit(X_train, Y_train) 

y_preds = neigh.predict(X_val)
score = f1_score(Y_val,y_preds,average='macro')
print score

