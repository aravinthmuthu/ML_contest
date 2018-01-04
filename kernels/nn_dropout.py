# encoding: UTF-8
# Copyright 2016 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


# import tensorflowvisu
# from tensorflow.examples.tutorials.mnist import input_data as mnist_data
print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)

X_input = np.genfromtxt("Data/train_mice2000PCA.csv",delimiter=",")
Y = np.genfromtxt("Data/train_Y.csv",delimiter=",", dtype='int32')
test = np.genfromtxt("Data/test_mice2000PCA.csv",delimiter=",")
print("read data...")

summaries_dir = 'summary/dropout_nn_0.9'

def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

X_train, X_val, Y_train, Y_val = train_test_split(X_input, indices_to_one_hot(Y,29), test_size=0.2, random_state=42)


# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
INP =2000
X = tf.placeholder(tf.float32, [None,  INP])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 29])
keep_prob = tf.placeholder(tf.float32)
phase_train = tf.placeholder(tf.bool, name='phase_train')

# five layers and their number of neurons (tha last layer has 10 softmax neurons)

L = 300
# M = 500
N = 100
# O = 200
# Weights initialised with small random values between -0.2 and +0.2
# When using RELUs, make sure biases are initialised with small *positive* values for example 0.1 = tf.ones([K])/10
W1 = tf.Variable(tf.truncated_normal([INP, L], stddev=0.1))  # 784 = 28 * 28
B1 = tf.Variable(tf.zeros([L]))
# W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
# B2 = tf.Variable(tf.zeros([M]))
W3 = tf.Variable(tf.truncated_normal([L, N], stddev=0.1))
B3 = tf.Variable(tf.zeros([N]))
# W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
# B4 = tf.Variable(tf.zeros([O]))
W5 = tf.Variable(tf.truncated_normal([N, 29], stddev=0.1))
B5 = tf.Variable(tf.zeros([29]))

# The model
XX = tf.reshape(X, [-1, INP])
Y1 = tf.nn.sigmoid(tf.matmul(XX, W1) + B1)
Y1_d = tf.nn.dropout(Y1, keep_prob)
# Y2 = tf.nn.sigmoid(tf.matmul(Y1_d, W2) + B2)
# Y2_d = tf.nn.dropout(Y2, keep_prob)
Y3 = tf.nn.sigmoid(tf.matmul(Y1_d, W3) + B3)
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
        
        train_summ, a, c = sess.run([train_acc_summary, accuracy, cross_entropy], {X: batch_X, Y_: batch_Y, keep_prob: 0.7, phase_train: True})
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")
        train_writer.add_summary(train_summ, i)

    # compute test values for visualisation
    if update_test_data:
        val_summ, a, c = sess.run([ validation_acc_summary, accuracy, cross_entropy], {X:X_val, Y_:Y_val, keep_prob: 1, phase_train: False})
        print(str(i) + ": ********* epoch " + str(i*100//8400+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))
        pred = prediction.eval(session=sess,feed_dict={X: test, keep_prob:1})
        val_writer.add_summary(val_summ, i)
        if(i>=8000):
            if(i%500 == 0):
                pred = pred.reshape((-1,1))
                pred = pred.astype(np.int64)

                idx = np.arange(test.shape[0]).reshape((-1,1))
                idx= idx.astype(np.int64)

                output = np.concatenate((idx,pred), axis=1)
                np.savetxt("results/nn_dropout_small_0.7/dropoutnn_2100_0.5_"+str(i)+".csv", output.astype(int), fmt='%i', delimiter=",")
                # if(i%5000):
                #     saver.save(sess, 'models/dropout_nn_0.9/my-model'+str(i), global_step=i)

    # the backpropagation training step
    sess.run(train_step, {X: batch_X, Y_: batch_Y, keep_prob:0.7, phase_train: True})

# datavis.animate(training_step, iterations=10000+1, train_data_update_freq=20, test_data_update_freq=100, more_tests_at_start=True)
for i in np.arange(100000):
    if(i%50 == 0):
        training_step(i,True,True)
    else:
        training_step(i,False, False)
# to save the animation as a movie, add save_movie=True as an argument to datavis.animate
# to disable the visualisation use the following line instead of the datavis.animate line
# for i in range(10000+1): training_step(i, i % 100 == 0, i % 20 == 0)

# print("max test accuracy: " + str(datavis.get_max_test_accuracy()))

# Some results to expect:
# (In all runs, if sigmoids are used, all biases are initialised at 0, if RELUs are used,
# all biases are initialised at 0.1 apart from the last one which is initialised at 0.)

## learning rate = 0.003, 10K iterations
# final test accuracy = 0.9788 (sigmoid - slow start, training cross-entropy not stabilised in the end)
# final test accuracy = 0.9825 (relu - above 0.97 in the first 1500 iterations but noisy curves)

## now with learning rate = 0.0001, 10K iterations
# final test accuracy = 0.9722 (relu - slow but smooth curve, would have gone higher in 20K iterations)

## decaying learning rate from 0.003 to 0.0001 decay_speed 2000, 10K iterations
# final test accuracy = 0.9746 (sigmoid - training cross-entropy not stabilised)
# final test accuracy = 0.9824 (relu - training set fully learned, test accuracy stable)