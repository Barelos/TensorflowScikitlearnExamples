"""
Work with breast cancer wisconsin using scikit-learn and tensorfow
"""
import numpy as np
from sklearn.metrics import accuracy_score
# read data and make sample and test
def read_data(filename):
    data = {}
    with open(filename, "r") as f:
        for line in f.readlines():
            line = line.replace("\n", "").split(",")
            line[2:] = [float(i) for i in line[2:]]
            if line[1] not in data.keys():
                data[line[1]] = []
            data[line[1]].append(line[1:])
    return data

def split_data(data, ratio=0.7):
    training = []
    test = []
    for key in data.keys():
        training += data[key][:int(ratio * len(data[key]))]
        test += data[key][int(ratio * len(data[key])):]
    return np.stack(training), np.stack(test)
# get data and sets
data = read_data("wdbc_data.txt")
tr, te = split_data(data)
# Tree classifier
from sklearn import tree
clf_tree = tree.DecisionTreeClassifier()
clf_tree.fit(tr[:,1:], tr[:,0])
predictions_tree = clf_tree.predict(te[:,1:])
# KNN
from sklearn.neighbors import KNeighborsClassifier
clf_knn = KNeighborsClassifier()
clf_knn.fit(tr[:,1:], tr[:,0])
predictions_knn = clf_knn.predict(te[:,1:])
# neural net
import tensorflow as tf

name_to_vec= {"M" : np.array([1,0]),
              "B" : np.array([0,1])}

learning_rate = 0.01

n_in = 30
n_class = 2
n_h1 = 50
n_h2 = 20
n_h3 = 10

X = tf.placeholder("float", [None, n_in])
Y = tf.placeholder("float")

weights = {
        'h1' : tf.Variable(tf.random_normal([n_in, n_h1])),
        'h2' : tf.Variable(tf.random_normal([n_h1, n_h2])),
        'h3' : tf.Variable(tf.random_normal([n_h2, n_h3])),
        'out' : tf.Variable(tf.random_normal([n_h3, n_class]))
}
biases = {
        'b1' : tf.Variable(tf.random_normal([n_h1])),
        'b2' : tf.Variable(tf.random_normal([n_h2])),
        'b3' : tf.Variable(tf.random_normal([n_h3])),
        'out' : tf.Variable(tf.random_normal([n_class]))
}

def neural_net(x):
    # first layer
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # second layer
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # third layer
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    # out layer
    layer_out = tf.matmul(layer_3, weights['out']) + biases['out']

    return layer_out

logits = neural_net(X) # activation
prediction = tf.nn.softmax(logits) # softmax

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    sess.run(init)
    batch_x, batch_y = tr[:,1:], [name_to_vec[i] for i in tr[:,0]]
    for i in range(100):
        sess.run(train_op, feed_dict={X:batch_x, Y:batch_y})
    # Calculate accuracy
    accuracy_nn = sess.run(accuracy, feed_dict={X: te[:,1:], Y: [name_to_vec[i] for i in te[:,0]]})

# show results
print("Tree:", accuracy_score(te[:,0], predictions_tree))
print("KNN:", accuracy_score(te[:,0], predictions_knn))
print("NN:", accuracy_nn)
