from sklearn import svm, grid_search
from sklearn.model_selection import GridSearchCV
import numpy as np
def svc_param_selection(X, y, nfolds):
    Cs = [ 0.1, 1, 10, 100]
    gammas = [ 0.001, 0.01, 0.1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='sigmoid'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_, grid_search.cv_results_


print "importing data..."
train_features = np.genfromtxt('../Data/train_mice1700PCA.csv', delimiter = ',')
train_labels = np.genfromtxt('../Data/train_Y.csv', delimiter = ',')
test_features = np.genfromtxt('../Data/test_mice1700PCA.csv', delimiter = ',')

best, results = svc_param_selection(train_features, train_labels, 3)

print(best)
print(results)