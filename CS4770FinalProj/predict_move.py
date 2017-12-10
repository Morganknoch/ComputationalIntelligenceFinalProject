import numpy as np
import random
import math

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

# initialize a player
def evolutionary_player(count):
    
    first_layer_weights = np.random.normal(0, scale=1.0, size=first_layer_hidden_weights)
    first_layer_bias = np.random.normal(0, scale=1.0, size=first_layer_hidden_bias)
    second_layer_weights = np.random.normal(0, scale=1.0, size=second_layer_hidden_weights)
    second_layer_bias = np.random.normal(0, scale=1.0, size=second_layer_hidden_bias)
    third_layer_weights = np.random.normal(0, scale=1.0, size=third_layer_hidden_weights)
    third_layer_bias = np.random.normal(0, scale=1.0, size=third_layer_hidden_bias)

    return Evol_Player(count, first_layer_weights, first_layer_bias, second_layer_weights, second_layer_bias, third_layer_weights, third_layer_bias)


# Use this as the actual heuristic function, since tensorflow takes too long to run the data through
def predict_nn(board, player):
    #board should be given as a 1x32 np array
    first_hidden_output = sigmoid( np.dot( board, player.first_layer_weights) + player.first_layer_bias )
    second_hidden_output = sigmoid( np.dot(first_hidden_output, player.second_layer_weights) + player.second_layer_bias )
    third_layer_output = sigmoid( np.dot(second_hidden_output, player.third_layer_weights) + player.third_layer_bias )

    output = np.sum(third_layer_output)
    
    return output

def fogel_create_offspring(parent, child):
    child.first_layer_weights = np.copy(parent.first_layer_weights) + (.5 * np.random.normal(0,1))
    child.first_layer_bias = np.copy(parent.first_layer_bias) + (.5 * np.random.normal(0,1))
    child.second_layer_weights =  np.copy(parent.second_layer_weights) + (.5 * np.random.normal(0,1))
    child.second_layer_bias =  np.copy(parent.second_layer_bias) + (.5 * np.random.normal(0,1))
    child.third_layer_weights = np.copy(parent.third_layer_weights) + (.5 * np.random.normal(0,1))
    child.third_layer_bias = np.copy(parent.third_layer_bias) + (.5 * np.random.normal(0,1))

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

    chance_of_mutate = random.random()
    
    if chance_of_mutate < .1:
        child.first_layer_weights += (.5 * np.random.normal(0,1))
        child.first_layer_bias += (.5 * np.random.normal(0,1))
        child.second_layer_weights += (.5 * np.random.normal(0,1))
        child.second_layer_bias += (.5 * np.random.normal(0,1))
        child.third_layer_weights += (.5 * np.random.normal(0,1))
        child.third_layer_bias += (.5 * np.random.normal(0,1))

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

    chance_of_mutate = random.random()
    
    if chance_of_mutate < .1:
        child.first_layer_weights += (.5 * np.random.normal(0,1))
        child.first_layer_bias += (.5 * np.random.normal(0,1))
        child.second_layer_weights += (.5 * np.random.normal(0,1))
        child.second_layer_bias += (.5 * np.random.normal(0,1))
        child.third_layer_weights += (.5 * np.random.normal(0,1))
        child.third_layer_bias += (.5 * np.random.normal(0,1))


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

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))