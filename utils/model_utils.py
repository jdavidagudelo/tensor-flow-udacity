from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import numpy as np


def train_and_predict(sample_size, train_data_set, train_labels, test_data_set, test_labels, image_size=0):
    regression = LogisticRegression()
    x_train = train_data_set[:sample_size].reshape(sample_size, image_size * image_size)
    y_train = train_labels[:sample_size]
    regression.fit(x_train, y_train)
    x_test = test_data_set.reshape(test_data_set.shape[0], image_size * image_size)
    y_test = test_labels
    predicted_labels = regression.predict(x_test)
    print('Accuracy:', regression.score(x_test, y_test), 'when sample_size=', sample_size)
    # plot_utils.display_sample_data_set(test_data_set, predicted_labels, 'sample_size=' + str(sample_size))


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def reformat(data_set, labels, image_size=28, num_labels=10):
    data_set = data_set.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return data_set, labels


def train_tensor_flow_basic(train_data_set, train_labels,
                            test_data_set, test_labels,
                            valid_data_set, valid_labels,
                            image_size, num_labels,
                            train_subset=10000, num_steps=801):
    train_data_set, train_labels = reformat(train_data_set, train_labels, image_size, num_labels)
    test_data_set, test_labels = reformat(test_data_set, test_labels, image_size, num_labels)
    valid_data_set, valid_labels = reformat(valid_data_set, valid_labels, image_size, num_labels)
    graph = tf.Graph()
    with graph.as_default():
        # Input data.
        # Load the training, validation and test data into constants that are
        # attached to the graph.
        tf_train_data_set = tf.constant(train_data_set[:train_subset, :])
        tf_train_labels = tf.constant(train_labels[:train_subset])
        tf_valid_data_set = tf.constant(valid_data_set)
        tf_test_data_set = tf.constant(test_data_set)

        # Variables.
        # These are the parameters that we are going to be training. The weight
        # matrix will be initialized using random values following a (truncated)
        # normal distribution. The biases get initialized to zero.
        weights = tf.Variable(
            tf.truncated_normal([image_size * image_size, num_labels]))
        biases = tf.Variable(tf.zeros([num_labels]))

        # Training computation.
        # We multiply the inputs with the weight matrix, and add biases. We compute
        # the softmax and cross-entropy (it's one operation in TensorFlow, because
        # it's very common, and it can be optimized). We take the average of this
        # cross-entropy across all training examples: that's our loss.
        logits = tf.matmul(tf_train_data_set, weights) + biases
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

        # Optimizer.
        # We are going to find the minimum of this loss using gradient descent.
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        # Predictions for the training, validation, and test data.
        # These are not part of training, but merely here so that we can report
        # accuracy figures as we train.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(
            tf.matmul(tf_valid_data_set, weights) + biases)
        test_prediction = tf.nn.softmax(tf.matmul(tf_test_data_set, weights) + biases)
    with tf.Session(graph=graph) as session:
        # This is a one-time operation which ensures the parameters get initialized as
        # we described in the graph: random weights for the matrix, zeros for the
        # biases.
        tf.global_variables_initializer().run()
        for step in range(num_steps):
            # Run the computations. We tell .run() that we want to run the optimizer,
            # and get the loss value and the training predictions returned as numpy
            # arrays.
            _, l, predictions = session.run([optimizer, loss, train_prediction])
            if step % 100 == 0 and False:
                print('Loss at step %d: %f' % (step, l))
                print('Training accuracy: %.1f%%' % accuracy(
                    predictions, train_labels[:train_subset, :]))
                # Calling .eval() on valid_prediction is basically like calling run(), but
                # just to get that one numpy array. Note that it recomputes all its graph
                # dependencies.
                print('Validation accuracy: %.1f%%' % accuracy(
                    valid_prediction.eval(), valid_labels))
        print('Test accuracy of basic model: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
        print('Validation accuracy of basic model: %.1f%%' % accuracy(
            valid_prediction.eval(), valid_labels))


def train_tensor_flow_batch(train_data_set, train_labels,
                            test_data_set, test_labels,
                            valid_data_set, valid_labels,
                            image_size, num_labels,
                            batch_size=128, num_steps=300001,
                            learning_rate=0.5):
    train_data_set, train_labels = reformat(train_data_set, train_labels, image_size, num_labels)
    test_data_set, test_labels = reformat(test_data_set, test_labels, image_size, num_labels)
    valid_data_set, valid_labels = reformat(valid_data_set, valid_labels, image_size, num_labels)
    graph = tf.Graph()
    with graph.as_default():
        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_train_data_set = tf.placeholder(tf.float32,
                                           shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_data_set = tf.constant(valid_data_set)
        tf_test_data_set = tf.constant(test_data_set)

        # Variables.
        weights = tf.Variable(
            tf.truncated_normal([image_size * image_size, num_labels]))
        biases = tf.Variable(tf.zeros([num_labels]))

        # Training computation.
        logits = tf.matmul(tf_train_data_set, weights) + biases
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(
            tf.matmul(tf_valid_data_set, weights) + biases)
        test_prediction = tf.nn.softmax(tf.matmul(tf_test_data_set, weights) + biases)
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = train_data_set[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_data_set: batch_data, tf_train_labels: batch_labels}
            extra_staff, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if step % 500 == 0 and False:
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % accuracy(
                    valid_prediction.eval(), valid_labels))
        print("Test accuracy of batched model: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
        print("Validation accuracy of batched model: %.1f%%" % accuracy(
            valid_prediction.eval(), valid_labels))


def reformat_convolution(data_set, labels, image_size=28, num_channels=1, num_labels=10):
    data_set = data_set.reshape(
        (-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return data_set, labels


def train_tensor_flow_convolution(train_data_set, train_labels,
                                  test_data_set, test_labels,
                                  valid_data_set, valid_labels,
                                  image_size=28, num_labels=10, num_channels=1,
                                  num_steps=1001, learning_rate=0.05, batch_size=16, patch_size=5,
                                  depth=16, num_hidden=64, dropout_probability=0.1):
    train_data_set, train_labels = reformat_convolution(train_data_set, train_labels, image_size,
                                                        num_channels, num_labels)
    test_data_set, test_labels = reformat_convolution(test_data_set, test_labels, image_size,
                                                      num_channels, num_labels)
    valid_data_set, valid_labels = reformat_convolution(valid_data_set, valid_labels, image_size,
                                                        num_channels, num_labels)
    graph = tf.Graph()
    with graph.as_default():
        # Input data.
        tf_train_dataset = tf.placeholder(
            tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_data_set)
        tf_test_dataset = tf.constant(test_data_set)

        # Variables.
        layer1_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, num_channels, depth], stddev=0.1))
        layer1_biases = tf.Variable(tf.zeros([depth]))
        layer2_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, depth, depth], stddev=0.1))
        layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
        layer3_weights = tf.Variable(tf.truncated_normal(
            [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
        layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
        layer4_weights = tf.Variable(tf.truncated_normal(
            [num_hidden, num_labels], stddev=0.1))
        layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

        # Model.
        def model(data):
            # Changes
            conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
            conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer1_biases)
            hidden = tf.nn.dropout(hidden, 1 - dropout_probability)
            conv = tf.nn.conv2d(hidden, layer2_weights, [1, 1, 1, 1], padding='SAME')
            conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer2_biases)
            hidden = tf.nn.dropout(hidden, 1 - dropout_probability)
            shape = hidden.get_shape().as_list()
            reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
            hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
            hidden = tf.nn.dropout(hidden, 1 - dropout_probability)
            return tf.matmul(hidden, layer4_weights) + layer4_biases

        # Training computation.
        logits = model(tf_train_dataset)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
        test_prediction = tf.nn.softmax(model(tf_test_dataset))
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_data_set[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if step % 1000 == 0:
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                print('Validation accuracy: %.1f%%' % accuracy(
                    valid_prediction.eval(), valid_labels))
        print('Test accuracy convolution: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
        print('Validation accuracy convolution: %.1f%%' % accuracy(
            valid_prediction.eval(), valid_labels))
