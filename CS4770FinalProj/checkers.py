# Title: checkers_v6.py
# Author: Chris Larson
# CS-6700 Final Project
# All Rights Reserved (2016)

"""This is a checkers engine that generates moves using a convolutional neural network that has been trained
on ~23k masters level checkers games that were recorded from checkers competitions that took place in the 1800 & 1900's.
These games are contained in the text file 'OCA_2.0.pdn', and were parsed and encoded using parser_v7.py. The CNN is
trained using train_v6.py. The model parameters are stored in a checkpoint folder located in the 'parameters' directory."""



# playernum = -1 is the pretrained, non-evol network


import numpy as np
import pandas as pd
import predict_move
import random
import math
import copy
import time

player_count = 0
players = []
first_layer_hidden_weights = (32,90) #32,90
first_layer_hidden_bias = (1,90) #90
second_layer_hidden_weights = (90,40) #90,40 
second_layer_hidden_bias = (1,40) #40
third_layer_hidden_weights = (40,10) #40,10
third_layer_hidden_bias = (1,10) #10

from copy import deepcopy 
#1 is black, -1 is white

######################## VARIABLES ########################

turn = 'white' # keep track of whose turn it is
selected = (0, 1) # a tuple keeping track of which piece is selected
board = 0 # link to our 'main' board
move_limit = [150, 0] # move limit for each game (declares game as draw otherwise)

# artificial intelligence related
best_move = () # best move for the player as determined by strategy
black, white = (), () # black and white players

# gui variables
window_size = (256, 256) # size of board in pixels
background_image_filename = 'board_brown.png' # image for the background
title = 'pyCheckers 1.1.2.3 final' # window title
board_size = 8 # board is 8x8 squares
left = 1 # left mouse button
fps = 5 # framerate of the scene (to save cpu time)
pause = 5 # number of seconds to pause the game for after end of game
start = True # are we at the beginnig of the game?

######################## CLASSES ########################

# class representing piece on the board
class Piece(object):
	def __init__(self, color, king):
		self.color = color
		self.king = king

# class representing player
class Player(object):
	def __init__(self, type, color, strategy, ply_depth):
		self.type = type # cpu or human
		self.color = color # black or white
		self.strategy = strategy # choice of strategy: minimax, negascout, negamax, minimax w/ab
		self.ply_depth = ply_depth # ply depth for algorithms
		#self.number = number # ply depth for algorithms

######################## INITIALIZE ########################

# will initialize board with all the pieces
def init_board():
	global move_limit
	move_limit[1] = 0 # reset move limit
	
	result = [
	[ 0, 1, 0, 1, 0, 1, 0, 1],
	[ 1, 0, 1, 0, 1, 0, 1, 0],
	[ 0, 1, 0, 1, 0, 1, 0, 1],
	[ 0, 0, 0, 0, 0, 0, 0, 0],
	[ 0, 0, 0, 0, 0, 0, 0, 0],
	[-1, 0,-1, 0,-1, 0,-1, 0],
	[ 0,-1, 0,-1, 0,-1, 0,-1],
	[-1, 0,-1, 0,-1, 0,-1, 0]
	] # initial board setting
	for m in range(8):
		for n in range(8):
			if (result[m][n] == 1):
				piece = Piece('black', False) # basic black piece
				result[m][n] = piece
			elif (result[m][n] == -1):
				piece = Piece('white', False) # basic white piece
				result[m][n] = piece
	return result

# initialize players
def init_player(type, color, strategy, ply_depth):
	return Player(type, color, strategy, ply_depth)

######################## FUNCTIONS ########################

# will return array with available moves to the player on board
def avail_moves(board, player):
    moves = [] # will store available jumps and moves

    for m in range(8):
        for n in range(8):
            if board[m][n] != 0 and board[m][n].color == player: # for all the players pieces...
                # ...check for jumps first
                if can_jump([m, n], [m+1, n+1], [m+2, n+2], board) == True: moves.append([m, n, m+2, n+2])
                if can_jump([m, n], [m-1, n+1], [m-2, n+2], board) == True: moves.append([m, n, m-2, n+2])
                if can_jump([m, n], [m+1, n-1], [m+2, n-2], board) == True: moves.append([m, n, m+2, n-2])
                if can_jump([m, n], [m-1, n-1], [m-2, n-2], board) == True: moves.append([m, n, m-2, n-2])

    if len(moves) == 0: # if there are no jumps in the list (no jumps available)
        # ...check for regular moves
        for m in range(8):
            for n in range(8):
                if board[m][n] != 0 and board[m][n].color == player: # for all the players pieces...
                    if can_move([m, n], [m+1, n+1], board) == True: moves.append([m, n, m+1, n+1])
                    if can_move([m, n], [m-1, n+1], board) == True: moves.append([m, n, m-1, n+1])
                    if can_move([m, n], [m+1, n-1], board) == True: moves.append([m, n, m+1, n-1])
                    if can_move([m, n], [m-1, n-1], board) == True: moves.append([m, n, m-1, n-1])

    return moves # return the list with available jumps or moves
                
# will return true if the jump is legal
def can_jump(a, via, b, board):
    # is destination off board?
    if b[0] < 0 or b[0] > 7 or b[1] < 0 or b[1] > 7:
        return False
    # does destination contain a piece already?
    if board[b[0]][b[1]] != 0: return False
    # are we jumping something?
    if board[via[0]][via[1]] == 0: return False
    # for white piece
    if board[a[0]][a[1]].color == 'white':
        if board[a[0]][a[1]].king == False and b[0] > a[0]: return False # only move up
        if board[via[0]][via[1]].color != 'black': return False # only jump blacks
        return True # jump is possible
    # for black piece
    if board[a[0]][a[1]].color == 'black':
        if board[a[0]][a[1]].king == False and b[0] < a[0]: return False # only move down
        if board[via[0]][via[1]].color != 'white': return False # only jump whites
        return True # jump is possible

# will return true if the move is legal
def can_move(a, b, board):
    # is destination off board?
    if b[0] < 0 or b[0] > 7 or b[1] < 0 or b[1] > 7:
        return False
    # does destination contain a piece already?
    if board[b[0]][b[1]] != 0: return False
    # for white piece (not king)
    if board[a[0]][a[1]].king == False and board[a[0]][a[1]].color == 'white':
        if b[0] > a[0]: return False # only move up
        return True # move is possible
    # for black piece
    if board[a[0]][a[1]].king == False and board[a[0]][a[1]].color == 'black':
        if b[0] < a[0]: return False # only move down
        return True # move is possible
    # for kings
    if board[a[0]][a[1]].king == True: return True # move is possible

# make a move on a board, assuming it's legit
def make_move(a, b, board):
    board[b[0]][b[1]] = board[a[0]][a[1]] # make the move
    board[a[0]][a[1]] = 0 # delete the source
    
    # check if we made a king
    if b[0] == 0 and board[b[0]][b[1]].color == 'white': board[b[0]][b[1]].king = True
    if b[0] == 7 and board[b[0]][b[1]].color == 'black': board[b[0]][b[1]].king = True
    
    if (a[0] - b[0]) % 2 == 0: # we made a jump...
        board[int((a[0]+b[0])/2)][int((a[1]+b[1])/2)] = 0 # delete the jumped piece

######################## CORE FUNCTIONS ########################

def evaluate(number, board, player):

    new_board = []
    
    # will evaluate board for a player
    #send modified board to NN
    #if player is white, do flip the board
    if player == 'white':
        for i in range(7,-1,-1):
            #new_row = []
            for j in range(7,-1,-1):
                if i % 2 == 0 and j % 2 == 0:
                    pass
                elif i % 2 == 1 and j%2 ==1:
                    pass
                else:
                    if board[i][j] != 0:
                        if board[i][j].color == 'black':
                            if board[i][j].king == True:
                                new_board.append(-3)
                            else:
                                new_board.append(-1)
                        else:
                            if board[i][j].king == True:
                                new_board.append(3)
                            else:
                                new_board.append(1)
                    else:
                        new_board.append(0)
            #new_board.append(new_row)
    else:
        for i in range(8):
            #new_row = []
            for j in range(8):
                if i % 2 == 0 and j % 2 == 0:
                    pass
                elif i % 2 == 1 and j%2 ==1:
                    pass
                else:
                    if board[i][j] != 0:
                        if board[i][j].color == 'white':
                            if board[i][j].king == True:
                                new_board.append(-3)
                            else:
                                new_board.append(-1)
                        else:
                            if board[i][j].king == True:
                                new_board.append(3)
                            else:
                                new_board.append(1)
                    else:
                        new_board.append(0)

    new_board = np.array(new_board)
    new_board.shape = (1,32)

    eval = predict_move.predict_nn(new_board, players[number])

    return eval

# have we killed the opponent already?
def end_game(board):
	black, white = 0, 0 # keep track of score
	for m in range(8):
		for n in range(8):
			if board[m][n] != 0:
				if board[m][n].color == 'black': black += 1 # we see a black piece
				else: white += 1 # we see a white piece

	return black, white


''' alpha-beta(player,board,alpha,beta) '''
def alpha_beta(number, player, board, ply, alpha, beta):
	global best_move

	# find out ply depth for player
	ply_depth = 0
	if player != 'black': ply_depth = white.ply_depth
	else: ply_depth = black.ply_depth

	end = end_game(board)

	''' if(game over in current board position) '''
	if ply >= ply_depth or end[0] == 0 or end[1] == 0: # are we still playing?
		''' return winner '''
		score = evaluate(number, board, player) # return evaluation of board as we have reached final ply or end state
		return score

	''' children = all legal moves for player from this board '''
	moves = avail_moves(board, player) # get the available moves for player

	''' if(max's turn) '''
	if player == turn: # if we are to play on node...
		''' for each child '''
		for i in range(len(moves)):
			# create a deep copy of the board (otherwise pieces would be just references)
			new_board = deepcopy(board)
			make_move((moves[i][0], moves[i][1]), (moves[i][2], moves[i][3]), new_board) # make move on new board

			''' score = alpha-beta(other player,child,alpha,beta) '''
			# ...make a switch of players for minimax...
			if player == 'black': player = 'white'
			else: player = 'black'

			score = alpha_beta(number,player, new_board, ply+1, alpha, beta)

			''' if score > alpha then alpha = score (we have found a better best move) '''
			if score > alpha:
				if ply == 0: best_move = (moves[i][0], moves[i][1]), (moves[i][2], moves[i][3]) # save the move
				alpha = score
			''' if alpha >= beta then return alpha (cut off) '''
			if alpha >= beta:
				#if ply == 0: best_move = (moves[i][0], moves[i][1]), (moves[i][2], moves[i][3]) # save the move
				return alpha

		''' return alpha (this is our best move) '''
		return alpha

	else: # the opponent is to play on this node...
		''' else (min's turn) '''
		''' for each child '''
		for i in range(len(moves)):
			# create a deep copy of the board (otherwise pieces would be just references)
			new_board = deepcopy(board)
			make_move((moves[i][0], moves[i][1]), (moves[i][2], moves[i][3]), new_board) # make move on new board

			''' score = alpha-beta(other player,child,alpha,beta) '''
			# ...make a switch of players for minimax...
			if player == 'black': player = 'white'
			else: player = 'black'

			score = alpha_beta(number,player, new_board, ply+1, alpha, beta)

			''' if score < beta then beta = score (opponent has found a better worse move) '''
			if score < beta: beta = score
			''' if alpha >= beta then return beta (cut off) '''
			if alpha >= beta: return beta
		''' return beta (this is the opponent's best move) '''
		return beta

# end turn
def end_turn():
	global turn # use global variables

	if turn != 'black':	turn = 'black'
	else: turn = 'white'

# play as a computer
def cpu_play(number, player):
	global board, move_limit # global variables
    # find and print the best move for cpu

	if player.strategy == 'alpha-beta': alpha = alpha_beta(number, player.color, board, 0, -10000, +10000)
	end_turn()


	if alpha == -10000: # no more moves available... all is lost
		if player.color == white: return -1     
		else: return -1


	make_move(best_move[0], best_move[1], board) # make the move on board

# initialize players and the boardfor the game
def game_init(ply):
	global black, white # work with global variables


	black = init_player('cpu', 'black', 'alpha-beta', ply) # init black player
	white = init_player('cpu', 'white', 'alpha-beta', ply) # init white player
	board = init_board()

	return board			

######################## START OF GAME ########################

def play_game(player1, player2):
    global board

    board = game_init(2) # initialize players and board for the game
    hold1 = 0
    hold2 = 0
    white_lost = False  
    black_lost = False
    while True:
        end = end_game(board)
        print_board(board)
        if end[1] == 0 or white_lost:	
            # black wins
            player1.score += 1
            player2.score -= 2
            print('Black won')
            break
        elif end[0] == 0 or black_lost: 
            # white wins
            player1.score -= 2
            player2.score += 1
            print('White won')
            break
	    # check if we breached the threshold for number of moves	
        elif move_limit[0] == move_limit[1]: 
            #draw, do nothing 
            print('It was a draw')
            break
	    # cpu play	
        if turn != 'black' and white.type == 'cpu' and (hold1 != -1 or hold2 != -1): 
            hold1 = cpu_play(player2.number, white) # white cpu turn
            move_limit[1] += 1 # add to move limit
            print('White made a move')
            if hold1 == -1:
                white_lost = True
        elif turn != 'white' and black.type == 'cpu' and (hold1 != -1 or hold2 != -1): 
            hold2 = cpu_play(player1.number, black) # black cpu turn
            move_limit[1] += 1 # add to move limit
            print('Black made a move')
            if hold2 == -1:
                black_lost = True

def print_board(board):

    for i in range(8):
        for j in range(8):
            if board[i][j] != 0:
                if board[i][j].color == 'white':
                    if board[i][j].king == True:
                        print('O', end="")
                    else:    
                        print('o', end="")
                else:
                    if board[i][j].king == True:
                        print('X', end="")
                    else:    
                        print('x', end="")
            else:
                print('-', end="")
        print('\n')

if __name__ == '__main__': #########################################
    player_dir='test_players/'
    
    
    first_layer_hidden_weights = (32,90) #32,90
    first_layer_hidden_bias = (1,90) #90
    second_layer_hidden_weights = (90,40) #90,40 
    second_layer_hidden_bias = (1,40) #40
    third_layer_hidden_weights = (40,10) #40,10
    third_layer_hidden_bias = (1,10) #10
    

# Create 30 players in the population to begin
def create_players():
    global player_count, players
    for i in range (0, 30):
        players.append(predict_move.evolutionary_player(player_count))
        player_count += 1


# randomly play the players against one another to determine who will be parents for the next generation
def score_players():
    
    # initialize scores to 0
    for i in range(0,30):
        players[i].score = 0

    for j in range(0, 30):
        for k in range(0, 5):
            player1 = i 
            player2 = math.floor(random.random() * 30)
            while player1 == player2:
                player2 = math.floor(random.random() * 30)

        play_game(players[player1], players[player2])  

def fogel_create_offspring():
    global players
    # rank parents
    # sort players by score
    numbers_not_used = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    numbers_used = []
    hold_players = copy.deepcopy(players)
    
    for i in range(0,30):
        for j in range(0,29-i):
            if hold_players[i].score < hold_players[i+1].score:
                hold_players[i], hold_players[i+1]  = hold_players[i+1], hold_players[i]
    
    for i in range(0, 15):
        numbers_used.append(hold_players[i].number)
        numbers_not_used.pop(numbers_not_used.index(hold_players[i].number)) #remove the numbers that are being used again as parents

    # we need to create the offspring now
    i = 0
    for n in numbers_not_used:
        predict_move.fogel_create_offspring( players[numbers_used[i]], players[n] )
        i +=1


def one_point_w_mutation_create_offspring():
    global players
    # rank parents
    # sort players by score
    numbers_not_used = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    numbers_used = []
    hold_players = copy.deepcopy(players)
    
    for i in range(0,30):
        for j in range(0,29-i):
            if hold_players[i].score < hold_players[i+1].score:
                hold_players[i], hold_players[i+1]  = hold_players[i+1], hold_players[i]
    
    for i in range(0, 15):
        numbers_used.append(hold_players[i].number)
        numbers_not_used.pop(numbers_not_used.index(hold_players[i].number)) #remove the numbers that are being used again as parents
        

    # we need to create the offspring now
    for n in numbers_not_used:
        #pick two random parents
        player1 = math.floor(random.random() * 15)
        player2 = math.floor(random.random() * 15)
        while player1 == player2:
            player2 = math.floor(random.random() * 15)
        predict_move.one_point_w_mutation_create_offspring( players[numbers_used[player1]], players[numbers_used[player2]], players[n] )
        

def two_point_w_mutation_create_offspring():
    global players
    # rank parents
    # sort players by score
    numbers_not_used = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    numbers_used = []
    hold_players = copy.deepcopy(players)
    
    for i in range(0,30):
        for j in range(0,29-i):
            if hold_players[i].score < hold_players[i+1].score:
                hold_players[i], hold_players[i+1]  = hold_players[i+1], hold_players[i]
    
    for i in range(0, 15):
        numbers_used.append(hold_players[i].number)
        numbers_not_used.pop(numbers_not_used.index(hold_players[i].number)) #remove the numbers that are being used again as parents
        

    # we need to create the offspring now
    for n in numbers_not_used:
        #pick two random parents
        player1 = math.floor(random.random() * 15)
        player2 = math.floor(random.random() * 15)
        while player1 == player2:
            player2 = math.floor(random.random() * 15)
        predict_move.two_point_w_mutation_create_offspring( players[numbers_used[player1]], players[numbers_used[player2]], players[n] )


def average_w_mutation_create_offspring():
    global players
    # rank parents
    # sort players by score
    numbers_not_used = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    numbers_used = []
    hold_players = copy.deepcopy(players)
    
    for i in range(0,30):
        for j in range(0,29-i):
            if hold_players[i].score < hold_players[i+1].score:
                hold_players[i], hold_players[i+1]  = hold_players[i+1], hold_players[i]
    
    for i in range(0, 15):
        numbers_used.append(hold_players[i].number)
        numbers_not_used.pop(numbers_not_used.index(hold_players[i].number)) #remove the numbers that are being used again as parents
        

    # we need to create the offspring now
    for n in numbers_not_used:
        #pick two random parents
        player1 = math.floor(random.random() * 15)
        player2 = math.floor(random.random() * 15)
        while player1 == player2:
            player2 = math.floor(random.random() * 15)
        predict_move.average_w_mutation_create_offspring( players[numbers_used[player1]], players[numbers_used[player2]], players[n] )

def one_point_crossover_ga():
    
    print("Creating players")
    create_players()
    start_time = time.time()
    print("Finished Creating players")
    for i in range(100):
        gen_start_time = time.time()
        score_players()
        print("--------------------------Finished scoring------------------------------------")
        one_point_w_mutation_create_offspring()
        gen_end_time = time.time() - gen_start_time
        gen_end_time *= 1000
        print('Calc time for generation was ' + str(gen_end_time) + ' seconds')

    end_time = time.time() - start_time
    end_time *= 1000 

    print('End time for players was ' + str(end_time) + ' seconds')
    k = 0
    print('Saving players...')
    for i in players:
        np.savez('one_point_cross_w_mutation/player' + str(k), first_layer_weights = i.first_layer_weights, first_layer_bias = i.first_layer_bias, second_layer_weights = i.second_layer_weights, second_layer_bias = i.second_layer_bias, third_layer_weights = i.third_layer_weights, third_layer_bias = i.third_layer_bias)
        k += 1

def two_point_crossover_ga():
    
    print("Creating players")
    create_players()
    start_time = time.time()
    print("Finished Creating players")
    for i in range(100):
        gen_start_time = time.time()
        score_players()
        print("--------------------------Finished scoring------------------------------------")
        two_point_w_mutation_create_offspring()
        gen_end_time = time.time() - gen_start_time
        gen_end_time *= 1000
        print('Calc time for generation was ' + str(gen_end_time) + ' seconds')

    end_time = time.time() - start_time
    end_time *= 1000 

    print('End time for players was ' + str(end_time) + ' seconds')
    k = 0
    print('Saving players...')
    for i in players:
        np.savez('two_point_cross_w_mutation/player' + str(k), first_layer_weights = i.first_layer_weights, first_layer_bias = i.first_layer_bias, second_layer_weights = i.second_layer_weights, second_layer_bias = i.second_layer_bias, third_layer_weights = i.third_layer_weights, third_layer_bias = i.third_layer_bias)
        k += 1

def average_crossover_ga():
    
    print("Creating players")
    create_players()
    start_time = time.time()
    print("Finished Creating players")
    for i in range(100):
        gen_start_time = time.time()
        score_players()
        print("--------------------------Finished scoring------------------------------------")
        average_w_mutation_create_offspring()
        gen_end_time = time.time() - gen_start_time
        gen_end_time *= 1000
        print('Calc time for generation was ' + str(gen_end_time) + ' seconds')

    end_time = time.time() - start_time
    end_time *= 1000 

    print('End time for players was ' + str(end_time) + ' seconds')
    k = 0
    print('Saving players...')
    for i in players:
        np.savez('average_w_mutation/player' + str(k), first_layer_weights = i.first_layer_weights, first_layer_bias = i.first_layer_bias, second_layer_weights = i.second_layer_weights, second_layer_bias = i.second_layer_bias, third_layer_weights = i.third_layer_weights, third_layer_bias = i.third_layer_bias)
        k += 1
   
def fogel_ea():

    print("Creating players")
    create_players()
    start_time = time.time()
    print("Finished Creating players")
    for i in range(100):
        gen_start_time = time.time()
        score_players()
        print("--------------------------Finished scoring------------------------------------")
        fogel_create_offspring()
        gen_end_time = time.time() - gen_start_time
        gen_end_time *= 1000
        print('Calc time for generation was ' + str(gen_end_time) + ' seconds')

    end_time = time.time() - start_time
    end_time *= 1000 

    print('End time for players was ' + str(end_time) + ' seconds')
    k = 0
    print('Saving players...')
    for i in players:
        np.savez('fogel_players/fogel_player' + str(k), first_layer_weights = i.first_layer_weights, first_layer_bias = i.first_layer_bias, second_layer_weights = i.second_layer_weights, second_layer_bias = i.second_layer_bias, third_layer_weights = i.third_layer_weights, third_layer_bias = i.third_layer_bias)
        k += 1

#fogel_ea()
#one_point_crossover_ga()
two_point_crossover_ga()
#average_crossover_ga()
print('Done')


    
    
