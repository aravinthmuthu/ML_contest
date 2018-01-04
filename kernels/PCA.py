import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib
import seaborn as sns

import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities,
                                    _kl_divergence)
from sklearn.utils.extmath import _ravel
# Random state.
RS = 20150101

# We'll use matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

X = np.genfromtxt("Data/train_mice10.csv",delimiter=",")
Y = np.genfromtxt("Data/train_Y.csv",delimiter=",", dtype='int32')
test = np.genfromtxt("Data/test_mice10.csv",delimiter=",")

# lda = LinearDiscriminantAnalysis(n_components =2)
# lda.fit(X,Y)
# X_new = lda.transform(X)
fulldata = np.concatenate((X, test), axis=0)

first500 = fulldata[:,:500]
second = fulldata[:,500:]
pca = PCA(n_components=200)
print ("fitting")
pca.fit(first500)
first500_pca = pca.transform(first500)
print(pca.explained_variance_ratio_) 

pca = PCA(n_components=1800)
print ("fitting")
pca.fit(second)
second_pca = pca.transform(second)

new_X = np.concatenate((first500_pca[:9501,:], second[:9501,:]), axis=1)
new_test = np.concatenate((first500_pca[9501:,:], second[9501:,:]), axis=1)

print("saving data...")
np.savetxt("Data/train_mice2000PCA.csv", new_X, delimiter=",")
print("saving data...")
np.savetxt("Data/test_mice2000PCA.csv", new_test, delimiter=",")
# X_new = pca.transform(X)

def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 29))

    # We create a scatter plot.
    f = plt.figure(figsize=(15, 15))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=30,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    plt.title("PCA")
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(29):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

scatter(X_new, Y)
plt.savefig('classes_PCA-generated.png', dpi=120)



np.savetxt("Data/train_PCA1000.csv", X_new, delimiter=",")
# np.savetxt("Data/train_Y.csv", Y, delimiter=",")
np.savetxt("Data/test_PCA1000.csv", test_new, delimiter=",")