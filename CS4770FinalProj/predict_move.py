import numpy as np
import tensorflow as tf
import random

def predict_cnn(board, output, params_dir):

    n = 1
    num_channels = 1
    patch_size = 2
    depth = 32
    num_nodes_layer3 = 400
    num_nodes_output = 128
    board_height = 8
    board_width = 4
    label_height = 32
    label_width = 4

    # Model input
    tf_x = tf.placeholder(tf.float32, shape=[n, board_height, board_width, num_channels])

    # Start interactive tf session
    session = tf.InteractiveSession()

    # Variables
    w1 = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1), name='w1')
    b1 = tf.Variable(tf.zeros([depth]), name='b1')
    w2 = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1), name='w2')
    b2 = tf.Variable(tf.zeros([depth]), name='b2')
    w3 = tf.Variable(tf.truncated_normal([board_height * board_width * depth, num_nodes_layer3], stddev=0.1), name='w3')
    b3 = tf.Variable(tf.zeros([num_nodes_layer3]), name='b3')
    w4 = tf.Variable(tf.truncated_normal([num_nodes_layer3, num_nodes_output], stddev=0.1), name='w4')
    b4 = tf.Variable(tf.zeros([num_nodes_output]), name='b4')

    # Compute
    c1 = tf.nn.conv2d(tf_x, w1, strides=[1, 1, 1, 1], padding='SAME')
    h1 = tf.nn.relu(c1 + b1)
    c2 = tf.nn.conv2d(h1, w2, strides=[1, 1, 1, 1], padding='SAME')
    h2 = tf.nn.relu(c2 + b2)
    h2_shape = tf_x.get_shape().as_list()
    h2_out_vec = tf.reshape(h2, shape=[h2_shape[0], board_height * board_width * depth])
    y3 = tf.matmul(h2_out_vec, w3) + b3
    h3 = tf.nn.relu(y3)
    y4 = tf.matmul(h3, w4) + b4
    y_pred = tf.nn.softmax(y4)

    # Restore saved model params
    var_dict = {'w1': w1,
                'b1': b1,
                'w2': w2,
                'b2': b2,
                'w3': w3,
                'b3': b3,
                'w4': w4,
                'b4': b4,
                }
    saver = tf.train.Saver(var_dict)
    init = tf.global_variables_initializer()
    session.run(init)
    saver.restore(session, params_dir)

    # Predict
    board = np.reshape(board.as_matrix(), (n, board_height, board_width, num_channels))
    y = y_pred.eval(feed_dict={tf_x: board})
    norm = np.sum(y)

    # Return max or top-5 prob(s)
    if output == 'one-vs-all':
        for i in range(n):
            ind = np.argmax(y[i, :])
            y[i, :] = 0
            y[i, ind] = 1
    elif output == 'top-5':
        for i in range(n):
            ind = np.argsort(y[i, :])[::-1][:5]
            probs = y[i, ind] / norm
            y[i, :] = 0
            for j in range(1, 6):
                y[i, ind[j - 1]] = j
    elif output == 'top-10':
        for i in range(n):
            ind = np.argsort(y[i, :])[::-1][:10]
            probs = y[i, ind] / norm
            y[i, :] = 0
            for j in range(1, 11):
                y[i, ind[j - 1]] = j
    elif output == 'top-50':
        for i in range(n):
            ind = np.argsort(y[i, :])[::-1][:50]
            probs = y[i, ind] / norm
            y[i, :] = 0
            for j in range(1, 51):
                y[i, ind[j - 1]] = j

    session.close()
    return np.reshape(y, (label_height, label_width)).astype(int), probs


class Player(object):
    def __init__(self, number):
        self.number = number
        self.score = 0

    
def evolutionary_player(player_dir, player_count):

    n = 1
    num_channels = 1
    patch_size = 2
    depth = 32
    num_nodes_layer3 = 400
    num_nodes_output = 128
    board_height = 8
    board_width = 4
    label_height = 32
    label_width = 4

    # Model input
    tf_x = tf.placeholder(tf.float32, shape=[n, board_height, board_width, num_channels])

    # Start interactive tf session
    session = tf.InteractiveSession()

    # Variables
    w1 = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1), name='w1')
    b1 = tf.Variable(tf.zeros([depth]), name='b1')
    w2 = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1), name='w2')
    b2 = tf.Variable(tf.zeros([depth]), name='b2')
    w3 = tf.Variable(tf.truncated_normal([board_height * board_width * depth, num_nodes_layer3], stddev=0.1), name='w3')
    b3 = tf.Variable(tf.zeros([num_nodes_layer3]), name='b3')
    w4 = tf.Variable(tf.truncated_normal([num_nodes_layer3, num_nodes_output], stddev=0.1), name='w4')
    b4 = tf.Variable(tf.zeros([num_nodes_output]), name='b4')

    # Restore saved model params
    var_dict = {'w1': w1,
                'b1': b1,
                'w2': w2,
                'b2': b2,
                'w3': w3,
                'b3': b3,
                'w4': w4,
                'b4': b4,
                }
    saver = tf.train.Saver(var_dict)
    init = tf.global_variables_initializer()
    session.run(init)
    
    saver.save(session, player_dir + str(player_count) + 'model.ckpt')

    #saver.restore(session, player_dir + str(player_count) + 'model.ckpt')

    session.close()
    return Player(player_count)


def fogel_create_offspring(player_dir, parent_num, offspring_num):

    n = 1
    num_channels = 1
    patch_size = 2
    depth = 32
    num_nodes_layer3 = 400
    num_nodes_output = 128
    board_height = 8
    board_width = 4
    label_height = 32
    label_width = 4

    # Model input
    tf_x = tf.placeholder(tf.float32, shape=[n, board_height, board_width, num_channels])

    # Start interactive tf session
    session = tf.InteractiveSession()

    # Variables
    w1 = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1), name='w1')
    b1 = tf.Variable(tf.zeros([depth]), name='b1')
    w2 = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1), name='w2')
    b2 = tf.Variable(tf.zeros([depth]), name='b2')
    w3 = tf.Variable(tf.truncated_normal([board_height * board_width * depth, num_nodes_layer3], stddev=0.1), name='w3')
    b3 = tf.Variable(tf.zeros([num_nodes_layer3]), name='b3')
    w4 = tf.Variable(tf.truncated_normal([num_nodes_layer3, num_nodes_output], stddev=0.1), name='w4')
    b4 = tf.Variable(tf.zeros([num_nodes_output]), name='b4')

    # Restore saved model params
    var_dict = {'w1': w1,
                'b1': b1,
                'w2': w2,
                'b2': b2,
                'w3': w3,
                'b3': b3,
                'w4': w4,
                'b4': b4,
                }
    saver = tf.train.Saver(var_dict)
    init = tf.global_variables_initializer()
    session.run(init)
    saver.restore(session, player_dir + str(parent_num) + 'model.ckpt')

    #create offspring
    hold = w1.eval() + (.25 * np.random.normal(0,1))
    w1.assign(hold).eval()
    hold = b1.eval() + (.25 * np.random.normal(0,1))
    b1.assign(hold).eval()
    hold = w2.eval() + (.25 * np.random.normal(0,1))
    w2.assign(hold).eval()
    hold = b2.eval() + (.25 * np.random.normal(0,1))
    b2.assign(hold).eval()
    hold = w3.eval() + (.25 * np.random.normal(0,1))
    w3.assign(hold).eval()
    hold = b3.eval() + (.25 * np.random.normal(0,1))
    b3.assign(hold).eval()
    hold = w4.eval() + (.25 * np.random.normal(0,1))
    w4.assign(hold).eval()
    hold = b4.eval() + (.25 * np.random.normal(0,1))
    b4.assign(hold).eval()

    
    saver.save(session, player_dir + str(offspring_num) + 'model.ckpt')
    session.close()
    return Player(offspring_num)


