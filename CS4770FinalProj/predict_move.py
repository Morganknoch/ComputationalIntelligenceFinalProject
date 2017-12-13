import numpy as np
import random
import tensorflow as tf
import math

# Weight and bias dimensions for the NN
first_layer_hidden_weights = (32,90) #32,90
first_layer_hidden_bias = (1,90) #90
second_layer_hidden_weights = (90,40) #90,40 
second_layer_hidden_bias = (1,40) #40
third_layer_hidden_weights = (40,10) #40,10
third_layer_hidden_bias = (1,10) #10

class Evol_Player(object):
    def __init__(self, number, first_layer_weights, first_layer_bias, second_layer_weights, second_layer_bias, third_layer_weights, third_layer_bias):
        self.number = number
        self.score = 0
        self.first_layer_weights = first_layer_weights
        self.first_layer_bias = first_layer_bias
        self.second_layer_weights = second_layer_weights
        self.second_layer_bias = second_layer_bias
        self.third_layer_weights = third_layer_weights
        self.third_layer_bias = third_layer_bias

# CNN for move prediction
def predict_cnn(board):

    params_dir = 'parameters/convnet_150k_full/model.ckpt-150001'
    n = 1
    num_channels = 1
    patch_size = 2
    depth = 32
    num_nodes_layer3 = 1024
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
    board = np.reshape(board, (n, board_height, board_width, num_channels))
    y = y_pred.eval(feed_dict={tf_x: board})
    norm = np.sum(y)


    
    for i in range(n):
        ind = np.argsort(y[i, :])[::-1][:50]
        probs = y[i, ind] / norm
        y[i, :] = 0
        for j in range(1, 51):
            y[i, ind[j - 1]] = j

    session.close()
    return np.reshape(y, (label_height, label_width)).astype(int), probs

# initialize a player
def evolutionary_player(count):
    
    first_layer_weights = np.random.normal(0, scale=1.0, size=first_layer_hidden_weights)
    first_layer_bias = np.random.normal(0, scale=1.0, size=first_layer_hidden_bias)
    second_layer_weights = np.random.normal(0, scale=1.0, size=second_layer_hidden_weights)
    second_layer_bias = np.random.normal(0, scale=1.0, size=second_layer_hidden_bias)
    third_layer_weights = np.random.normal(0, scale=1.0, size=third_layer_hidden_weights)
    third_layer_bias = np.random.normal(0, scale=1.0, size=third_layer_hidden_bias)

    return Evol_Player(count, first_layer_weights, first_layer_bias, second_layer_weights, second_layer_bias, third_layer_weights, third_layer_bias)


# Use this Neural Network as the heuristic function for the minimax tree
def predict_nn(board, player):
    #board should be given as a 1x32 np array
    first_hidden_output = sigmoid( np.dot( board, player.first_layer_weights) + player.first_layer_bias )
    second_hidden_output = sigmoid( np.dot(first_hidden_output, player.second_layer_weights) + player.second_layer_bias )
    third_layer_output = sigmoid( np.dot(second_hidden_output, player.third_layer_weights) + player.third_layer_bias )

    output = np.sum(third_layer_output)
    
    return output

# Perform the fogel variation method
def fogel_create_offspring(parent, child):
    child.first_layer_weights = np.copy(parent.first_layer_weights) + (.5 * np.random.normal(0,1))
    child.first_layer_bias = np.copy(parent.first_layer_bias) + (.5 * np.random.normal(0,1))
    child.second_layer_weights =  np.copy(parent.second_layer_weights) + (.5 * np.random.normal(0,1))
    child.second_layer_bias =  np.copy(parent.second_layer_bias) + (.5 * np.random.normal(0,1))
    child.third_layer_weights = np.copy(parent.third_layer_weights) + (.5 * np.random.normal(0,1))
    child.third_layer_bias = np.copy(parent.third_layer_bias) + (.5 * np.random.normal(0,1))

# Perform the one point crossover variation, with mutation
def one_point_w_mutation_create_offspring(parent1, parent2, child):
    parent1_first_layer_weights = np.copy(parent1.first_layer_weights)
    parent1_first_layer_bias = np.copy(parent1.first_layer_bias)
    parent1_second_layer_weights = np.copy(parent1.second_layer_weights)
    parent1_second_layer_bias = np.copy(parent1.second_layer_bias)
    parent1_third_layer_weights = np.copy(parent1.third_layer_weights)
    parent1_third_layer_bias = np.copy(parent1.third_layer_bias)

    parent2_first_layer_weights = np.copy(parent2.first_layer_weights)
    parent2_first_layer_bias = np.copy(parent2.first_layer_bias)
    parent2_second_layer_weights = np.copy(parent2.second_layer_weights)
    parent2_second_layer_bias = np.copy(parent2.second_layer_bias)
    parent2_third_layer_weights = np.copy(parent2.third_layer_weights)
    parent2_third_layer_bias = np.copy(parent2.third_layer_bias)


    # Cross-over
    num_weights_1 = math.floor(random.random() * 90)
    child.first_layer_weights[:,:num_weights_1] = parent1_first_layer_weights[:,:num_weights_1]
    child.first_layer_weights[:,num_weights_1:] = parent2_first_layer_weights[:,num_weights_1:]

    num_bias_1 = math.floor(random.random() * 90)
    child.first_layer_bias[:num_bias_1] = parent1_first_layer_bias[:num_bias_1]
    child.first_layer_bias[num_bias_1:] = parent2_first_layer_bias[num_bias_1:]

    num_weights_2 = math.floor(random.random() * 40)
    child.second_layer_weights[:,:num_weights_2] = parent1_second_layer_weights[:,:num_weights_2]
    child.second_layer_weights[:,num_weights_2:] = parent2_second_layer_weights[:,num_weights_2:]

    num_bias_2 = math.floor(random.random() * 40)
    child.second_layer_bias[:num_bias_2] = parent1_second_layer_bias[:num_bias_2]
    child.second_layer_bias[num_bias_2:] = parent2_second_layer_bias[num_bias_2:]

    num_weights_3 = math.floor(random.random() * 10)
    child.third_layer_weights[:,:num_weights_3] = parent1_third_layer_weights[:,:num_weights_3]
    child.third_layer_weights[:,num_weights_3:] = parent2_third_layer_weights[:,num_weights_3:]

    num_bias_3 = math.floor(random.random() * 10)
    child.third_layer_bias[:num_bias_3] = parent1_third_layer_bias[:num_bias_3]
    child.third_layer_bias[num_bias_3:] = parent2_third_layer_bias[num_bias_3:]

    # Mutation

    chance_of_mutate = random.random()
    
    if chance_of_mutate < .1:
        child.first_layer_weights += (.5 * np.random.normal(0,1))
        child.first_layer_bias += (.5 * np.random.normal(0,1))
        child.second_layer_weights += (.5 * np.random.normal(0,1))
        child.second_layer_bias += (.5 * np.random.normal(0,1))
        child.third_layer_weights += (.5 * np.random.normal(0,1))
        child.third_layer_bias += (.5 * np.random.normal(0,1))


# Perform the two point crossover variation, with mutation
def two_point_w_mutation_create_offspring(parent1, parent2, child):
    parent1_first_layer_weights = np.copy(parent1.first_layer_weights)
    parent1_first_layer_bias = np.copy(parent1.first_layer_bias)
    parent1_second_layer_weights = np.copy(parent1.second_layer_weights)
    parent1_second_layer_bias = np.copy(parent1.second_layer_bias)
    parent1_third_layer_weights = np.copy(parent1.third_layer_weights)
    parent1_third_layer_bias = np.copy(parent1.third_layer_bias)

    parent2_first_layer_weights = np.copy(parent2.first_layer_weights)
    parent2_first_layer_bias = np.copy(parent2.first_layer_bias)
    parent2_second_layer_weights = np.copy(parent2.second_layer_weights)
    parent2_second_layer_bias = np.copy(parent2.second_layer_bias)
    parent2_third_layer_weights = np.copy(parent2.third_layer_weights)
    parent2_third_layer_bias = np.copy(parent2.third_layer_bias)


    # Cross-over
    num_weights_1 = math.floor(random.random() * 90)
    num_weights_1_1 = math.floor(random.random() * 90)

    # Make sure the second number is greater than the first so the crossover occurs correctly
    if num_weights_1 > num_weights_1_1:
        hold = num_weights_1
        num_weights_1 = num_weights_1_1
        num_weights_1_1 = hold

    child.first_layer_weights[:,:num_weights_1] = parent1_first_layer_weights[:,:num_weights_1]
    child.first_layer_weights[:,num_weights_1:num_weights_1_1] = parent2_first_layer_weights[:,num_weights_1:num_weights_1_1]
    child.first_layer_weights[:,num_weights_1_1:] = parent1_first_layer_weights[:,num_weights_1_1:]

    num_bias_1 = math.floor(random.random() * 90)
    num_bias_1_1 = math.floor(random.random() * 90)

    if num_bias_1 > num_bias_1_1:
        hold = num_bias_1
        num_bias_1 = num_bias_1_1
        num_bias_1_1 = hold

    child.first_layer_bias[:num_bias_1] = parent1_first_layer_bias[:num_bias_1]
    child.first_layer_bias[num_bias_1:num_bias_1_1] = parent2_first_layer_bias[num_bias_1:num_bias_1_1]
    child.first_layer_bias[num_bias_1_1:] = parent1_first_layer_bias[num_bias_1_1:]

    num_weights_2 = math.floor(random.random() * 40)
    num_weights_2_1 = math.floor(random.random() * 40)

    if num_weights_2 > num_weights_2_1:
        hold = num_weights_2
        num_weights_2 = num_weights_2_1
        num_weights_2_1 = hold

    child.second_layer_weights[:,:num_weights_2] = parent1_second_layer_weights[:,:num_weights_2]
    child.second_layer_weights[:,num_weights_2:num_weights_2_1] = parent2_second_layer_weights[:,num_weights_2:num_weights_2_1]
    child.second_layer_weights[:,num_weights_2_1:] = parent1_second_layer_weights[:,num_weights_2_1:]


    num_bias_2 = math.floor(random.random() * 40)
    num_bias_2_1 = math.floor(random.random() * 40)

    if num_bias_1 > num_bias_1_1:
        hold = num_bias_1
        num_bias_1 = num_bias_1_1
        num_bias_1_1 = hold

    child.second_layer_bias[:num_bias_2] = parent1_second_layer_bias[:num_bias_2]
    child.second_layer_bias[num_bias_2:num_bias_2_1] = parent2_second_layer_bias[num_bias_2:num_bias_2_1]
    child.second_layer_bias[num_bias_2_1:] = parent1_second_layer_bias[num_bias_2_1:]


    num_weights_3 = math.floor(random.random() * 10)
    num_weights_3_1 = math.floor(random.random() * 10)

    if num_weights_3 > num_weights_3_1:
            hold = num_weights_3
            num_weights_3 = num_weights_3_1
            num_weights_3_1 = hold

    child.third_layer_weights[:,:num_weights_3] = parent1_third_layer_weights[:,:num_weights_3]
    child.third_layer_weights[:,num_weights_3:num_weights_3_1] = parent2_third_layer_weights[:,num_weights_3:num_weights_3_1]
    child.third_layer_weights[:,num_weights_3_1:] = parent1_third_layer_weights[:,num_weights_3_1:]


    num_bias_3 = math.floor(random.random() * 90)
    num_bias_3_1 = math.floor(random.random() * 90)

    if num_bias_3 > num_bias_3_1:
        hold = num_bias_3
        num_bias_3 = num_bias_3_1
        num_bias_3_1 = hold

    child.third_layer_bias[:num_bias_3] = parent1_third_layer_bias[:num_bias_3]
    child.third_layer_bias[num_bias_3:num_bias_3_1] = parent2_third_layer_bias[num_bias_3:num_bias_3_1]
    child.third_layer_bias[num_bias_3_1:] = parent1_third_layer_bias[num_bias_3_1:]

    # Mutation

    chance_of_mutate = random.random()
    
    if chance_of_mutate < .1:
        child.first_layer_weights += (.5 * np.random.normal(0,1))
        child.first_layer_bias += (.5 * np.random.normal(0,1))
        child.second_layer_weights += (.5 * np.random.normal(0,1))
        child.second_layer_bias += (.5 * np.random.normal(0,1))
        child.third_layer_weights += (.5 * np.random.normal(0,1))
        child.third_layer_bias += (.5 * np.random.normal(0,1))

# Perform the average variation, with mutation
def average_w_mutation_create_offspring(parent1, parent2, child):

    parent1_first_layer_weights = np.copy(parent1.first_layer_weights)
    parent1_first_layer_bias = np.copy(parent1.first_layer_bias)
    parent1_second_layer_weights = np.copy(parent1.second_layer_weights)
    parent1_second_layer_bias = np.copy(parent1.second_layer_bias)
    parent1_third_layer_weights = np.copy(parent1.third_layer_weights)
    parent1_third_layer_bias = np.copy(parent1.third_layer_bias)

    parent2_first_layer_weights = np.copy(parent2.first_layer_weights)
    parent2_first_layer_bias = np.copy(parent2.first_layer_bias)
    parent2_second_layer_weights = np.copy(parent2.second_layer_weights)
    parent2_second_layer_bias = np.copy(parent2.second_layer_bias)
    parent2_third_layer_weights = np.copy(parent2.third_layer_weights)
    parent2_third_layer_bias = np.copy(parent2.third_layer_bias)

    child.first_layer_weights = (parent1_first_layer_weights + parent2_first_layer_weights)/2
    child.first_layer_bias = (parent1_first_layer_bias + parent2_first_layer_bias)/2
    child.second_layer_weights = (parent1_second_layer_weights + parent2_second_layer_weights)/2
    child.second_layer_bias = (parent1_second_layer_bias + parent2_second_layer_bias)/2
    child.third_layer_weights = (parent1_third_layer_weights + parent2_third_layer_weights)/2
    child.third_layer_bias = (parent1_third_layer_bias + parent2_third_layer_bias)/2

    chance_of_mutate = math.floor(random.random())
    
    if chance_of_mutate < .1:
        child.first_layer_weights += (.5 * np.random.normal(0,1))
        child.first_layer_bias += (.5 * np.random.normal(0,1))
        child.second_layer_weights += (.5 * np.random.normal(0,1))
        child.second_layer_bias += (.5 * np.random.normal(0,1))
        child.third_layer_weights += (.5 * np.random.normal(0,1))
        child.third_layer_bias += (.5 * np.random.normal(0,1))

# Used in the NN
def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))