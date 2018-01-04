
import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist

import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities,
                                    _kl_divergence)
from sklearn.utils.extmath import _ravel
# Random state.
RS = 20150101

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 29))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    plt.title("t-SNE")
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



import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


# import tensorflowvisu
# from tensorflow.examples.tutorials.mnist import input_data as mnist_data
print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)


# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
# mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)
X_input = np.genfromtxt("Data/train_mice10.csv",delimiter=",")[:,500:]
Y_input = np.genfromtxt("Data/train_Y.csv",delimiter=",", dtype='int32')
test = np.genfromtxt("Data/test_mice10.csv",delimiter=",")[:,500:]
print("read data...")

summaries_dir = 'summary/dropout_nn_0.9'

def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

X_train, X_val, Y_train, Y_val = train_test_split(X_input, indices_to_one_hot(Y_input,29), test_size=0.2, random_state=42)


# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None,  2100])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 29])
keep_prob = tf.placeholder(tf.float32)
phase_train = tf.placeholder(tf.bool, name='phase_train')

# five layers and their number of neurons (tha last layer has 10 softmax neurons)
L = 2000
M = 1000
N = 500
# O = 200
# Weights initialised with small random values between -0.2 and +0.2
# When using RELUs, make sure biases are initialised with small *positive* values for example 0.1 = tf.ones([K])/10
W1 = tf.Variable(tf.truncated_normal([2100, L], stddev=0.1))  # 784 = 28 * 28
B1 = tf.Variable(tf.zeros([L]))
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
B2 = tf.Variable(tf.zeros([M]))
W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
B3 = tf.Variable(tf.zeros([N]))
# W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
# B4 = tf.Variable(tf.zeros([O]))
W5 = tf.Variable(tf.truncated_normal([N, 29], stddev=0.1))
B5 = tf.Variable(tf.zeros([29]))

# The model
XX = tf.reshape(X, [-1, 2100])
Y1 = tf.nn.sigmoid(tf.matmul(XX, W1) + B1)
print Y1.shape
Y1_d = tf.nn.dropout(Y1, keep_prob)
Y2 = tf.nn.sigmoid(tf.matmul(Y1_d, W2) + B2)
Y2_d = tf.nn.dropout(Y2, keep_prob)
Y3 = tf.nn.sigmoid(tf.matmul(Y2_d, W3) + B3)
Y3_d = tf.nn.dropout(Y3, keep_prob)
# Y4 = tf.nn.sigmoid(tf.matmul(Y3_d, W4) + B4)
# Y4_d = tf.nn.dropout(Y4, keep_prob)
Ylogits = tf.matmul(Y3_d, W5) + B5
Y = tf.nn.softmax(Ylogits)
prediction=tf.argmax(Y,1)

# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(prediction, tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_acc_summary = tf.summary.scalar('train_accuracy', accuracy)
validation_acc_summary = tf.summary.scalar('validation_accuracy', accuracy)  # intended to run on validation set

sess = tf.Session()
train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
val_writer = tf.summary.FileWriter(summaries_dir + '/val')

# matplotlib visualisation
# allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1])], 0)
# allbiases  = tf.concat([tf.reshape(B1, [-1]), tf.reshape(B2, [-1]), tf.reshape(B3, [-1]), tf.reshape(B4, [-1]), tf.reshape(B5, [-1])], 0)

# allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W5, [-1])], 0)
# allbiases  = tf.concat([tf.reshape(B1, [-1]), tf.reshape(B2, [-1]), tf.reshape(B3, [-1]), tf.reshape(B5, [-1])], 0)

# training step, learning rate = 0.003
learning_rate = 0.001
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

saver = tf.train.Saver()
# init
init = tf.global_variables_initializer()

sess.run(init)

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(Y_train))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [X_train[ i] for i in idx]
    labels_shuffle = [Y_train[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)





# You can call this function in a loop to train the model, 100 images at a time
def training_step(i, update_test_data, update_train_data):

    # training on batches of 100 images with 100 labels
    global learning_rate
    batch_X, batch_Y = next_batch(1000, X_train, Y_train)

    # compute training values for visualisation
    if update_train_data:
        if(i==10000):
            learning_rate=learning_rate/10
        elif(i==50000):
            learning_rate=learning_rate/10

        if(i==3500):
			RS = 20150101
			y1, y3, y, train_summ, a, c = sess.run([Y1, Y3, Y, train_acc_summary, accuracy, cross_entropy], {X: X_input, Y_: Y_input, keep_prob: 1, phase_train: True})
			print y1.shape
			print Y_train.shape
			IND_Y = np.argmax(Y_train,axis=1)
			digits_proj = TSNE(random_state=RS).fit_transform(y1)

			scatter(digits_proj, IND_Y)
			plt.savefig('3classes_nn_tsne_Y1-generated.png', dpi=120)
			digits_proj = TSNE(random_state=RS).fit_transform(y3)
			scatter(digits_proj, IND_Y)
			plt.savefig('3classes_nn_tsne_Y3-generated.png', dpi=120)
			digits_proj = TSNE(random_state=RS).fit_transform(y)
			scatter(digits_proj, IND_Y)
			plt.savefig('3classes_nn_tsne_Y-generated.png', dpi=120)

        
        train_summ, a, c = sess.run([train_acc_summary, accuracy, cross_entropy], {X: batch_X, Y_: batch_Y, keep_prob: 1, phase_train: True})
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")
        train_writer.add_summary(train_summ, i)
        if(i%50 ==0):
        	train_loss[i/50]=c

    # compute test values for visualisation
    if update_test_data:
        val_summ, a, c = sess.run([ validation_acc_summary, accuracy, cross_entropy], {X:X_val, Y_:Y_val, keep_prob: 1, phase_train: False})
        print(str(i) + ": ********* epoch " + str(i*100//8400+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))
        pred = prediction.eval(session=sess,feed_dict={X: test, keep_prob:1})
        val_writer.add_summary(val_summ, i)
        # if(i>8000):
        if(i%50 ==0):
        	test_loss[i/50]=c

                # pred = pred.reshape((-1,1))
                # pred = pred.astype(np.int64)

                # idx = np.arange(test.shape[0]).reshape((-1,1))
                # idx= idx.astype(np.int64)

                # output = np.concatenate((idx,pred), axis=1)
                # np.savetxt("results/nn_dropout/dropoutnn_2100_0.5_"+str(i)+".csv", output.astype(int), fmt='%i', delimiter=",")
                # if(i%5000):
                #     saver.save(sess, 'models/dropout_nn_0.9/my-model'+str(i), global_step=i)

    # the backpropagation training step
    sess.run(train_step, {X: batch_X, Y_: batch_Y, keep_prob:1, phase_train: True})

# datavis.animate(training_step, iterations=10000+1, train_data_update_freq=20, test_data_update_freq=100, more_tests_at_start=True)
train_loss = np.zeros(80)
test_loss = np.zeros(80)
iterations = np.arange(0,4000,50)

for i in np.arange(4000):
    if(i%50 == 0):
        training_step(i,True,True)
    else:
        training_step(i,False, False)

plt.plot(iterations,train_loss,'r', label='train loss')
plt.plot(iterations,test_loss,'g', label='test loss')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.title('model1 overfit')
plt.show()



