import tensorflow as tf
from scipy import io as sio
import numpy as np

def weight_varible(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


sess = tf.InteractiveSession()

# paras
W_conv1 = weight_varible([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# conv layer-1
x = tf.placeholder(tf.float32, [None, 1600])
x_image = tf.reshape(x, [-1, 40, 40, 1])

h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# conv layer-2
W_conv2 = weight_varible([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# full connection
W_fc1 = weight_varible([10 * 10 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 10 * 10 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# output layer: softmax
W_fc2 = weight_varible([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
y_ = tf.placeholder(tf.float32, [None, 10])

# model training
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

#loading data set
for j in range(1,33):
    trainDataStruct = sio.loadmat('F:\\计算机视觉\\hw2\\transformed\\training_batches\\'+str(j)+'.mat')

    trainLabel = trainDataStruct['affNISTdata']['label_one_of_n'][0][0].T
    trainData  = trainDataStruct['affNISTdata']['image'][0][0].T

    trainLabel = trainLabel.astype(np.float32)
    trainData = trainData.astype(np.float32)/[255]


    for i in range(500):
        trainBatch = trainData[i*100:(i+1)*100,:]
        trainLableBatch = trainLabel[i*100:(i+1)*100,:]
        if i % 100 == 0:
            train_accuacy = accuracy.eval(feed_dict={x: trainBatch, y_: trainLableBatch, keep_prob: 1.0})
            print("step %d, training accuracy %.6lf"%(j, train_accuacy))
        train_step.run(feed_dict = {x: trainBatch, y_: trainLableBatch, keep_prob: 0.5})


# accuacy on test
for i in range(1,33):
    testDataStruct  = sio.loadmat('F:\\计算机视觉\\hw2\\transformed\\test_batches\\'+str(i)+'.mat')
    testLabel  = testDataStruct['affNISTdata']['label_one_of_n'][0][0].T
    testData   = testDataStruct['affNISTdata']['image'][0][0].T

    testLabel = testLabel.astype(np.float32)
    testData = testData.astype(np.float32)/[255]

    print("test accuracy %.6lf"%(accuracy.eval(feed_dict={x: testData, y_: testLabel, keep_prob: 1.0})))