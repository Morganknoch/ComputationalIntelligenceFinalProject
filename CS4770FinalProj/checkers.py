# Computational Intelligence Final Project 

import numpy as np
import predict_move
import random
import math
import copy
import time
from copy import deepcopy 

# Create the dictionary for the transition of the output of CNN to board positions
board_dict = {}

j = 0
k = 0
l = 32
for i in range(8):
    k = 0
    if i % 2 == 0:
        k += 1
    for m in range(4):
           
        board_dict[l] = (j,k)
        k += 2
        l -= 1
    j += 1

######################## VARIABLES ########################

turn = 'white' # keep track of whose turn it is
selected = (0, 1) # a tuple keeping track of which piece is selected
board = 0 # link to our 'main' board
move_limit = [200, 0] # move limit for each game (declares game as draw otherwise)

# artificial intelligence related
best_move = () # best move for the player as determined by strategy
black, white = (), () # black and white players

player_count = 0
players = []
first_layer_hidden_weights = (32,90) #32,90
first_layer_hidden_bias = (1,90) #90
second_layer_hidden_weights = (90,40) #90,40 
second_layer_hidden_bias = (1,40) #40
third_layer_hidden_weights = (40,10) #40,10
third_layer_hidden_bias = (1,10) #10

predict_one_cross = False
predict_two_cross = False
predict_average = False
predict_fogel = False

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

    global players, predict_one_cross, predict_two_cross, predict_average, predict_fogel 
    new_board = []
    
    # will evaluate board for a player
    #send modified board to NN
    #if player is white, flip the board
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

    if predict_one_cross:
        eval = predict_move.predict_nn(new_board, one_point_cross_players[number])
    elif predict_two_cross:
        eval = predict_move.predict_nn(new_board, two_point_cross_players[number])
    elif predict_average:
        eval = predict_move.predict_nn(new_board, average_cross_players[number])
    elif predict_fogel:
        eval = predict_move.predict_nn(new_board, fogel_players[number])
    else:
        eval = predict_move.predict_nn(new_board, players[number])

    # If CNN fails to output a valid move, choose a random legal move
    if number == -1:
        eval == 0

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

# initialize players and the board for the game
def game_init(ply):
	global black, white # work with global variables


	black = init_player('cpu', 'black', 'alpha-beta', ply) # init black player
	white = init_player('cpu', 'white', 'alpha-beta', ply) # init white player
	board = init_board()

	return board			

######################## START OF GAME ########################

# Plays the checkers game for EC players for selection and the final tournament
def play_game(player1, player2, player1List, player2List):
    global board, predict_one_cross, predict_two_cross, predict_average, predict_fogel

    board = game_init(2) # initialize players and board for the game
    hold1 = 0
    hold2 = 0
    white_lost = False  
    black_lost = False
    num_white_pieces = 12
    num_black_pieces = 12
    while True:
        end = end_game(board)
    
        num_white_pieces, num_black_pieces = count_pieces(board)

        print_board(board)
        if end[1] == 0 or (white_lost and num_white_pieces < num_black_pieces):	
            # black wins
            player1.score += 1
            player2.score -= 2
            print('Black won')
            break
        elif end[0] == 0 or (black_lost and num_black_pieces < num_white_pieces): 
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
            if player2List == 'one_point':
                predict_one_cross = True
                predict_two_cross = False
                predict_average = False
                predict_fogel = False
            elif player2List == 'two_point':
                predict_one_cross = False
                predict_two_cross = True
                predict_average = False
                predict_fogel = False
            elif player2List == 'average':
                predict_one_cross = False
                predict_two_cross = False
                predict_average = True
                predict_fogel = False
            elif player2List == 'fogel':
                predict_one_cross = False
                predict_two_cross = False
                predict_average = False
                predict_fogel = True

            hold1 = cpu_play(player2.number, white) # white cpu turn
            move_limit[1] += 1 # add to move limit
            print('White made a move')
            if hold1 == -1:
                white_lost = True
        elif turn != 'white' and black.type == 'cpu' and (hold1 != -1 or hold2 != -1): 

            if player1List == 'one_point':
                predict_one_cross = True
                predict_two_cross = False
                predict_average = False
                predict_fogel = False
            elif player1List == 'two_point':
                predict_one_cross = False
                predict_two_cross = True
                predict_average = False
                predict_fogel = False
            elif player1List == 'average':
                predict_one_cross = False
                predict_two_cross = False
                predict_average = True
                predict_fogel = False
            elif player1List == 'fogel':
                predict_one_cross = False
                predict_two_cross = False
                predict_average = False
                predict_fogel = True


            hold2 = cpu_play(player1.number, black) # black cpu turn
            move_limit[1] += 1 # add to move limit
            print('Black made a move')
            if hold2 == -1:
                black_lost = True

# Used to play between an EC player and the CNN
def cnn_play_game(player1, player1List):
    global board, predict_one_cross, predict_two_cross, predict_average, predict_fogel

    board = game_init(2) # initialize players and board for the game
    hold1 = 0
    hold2 = 0
    white_lost = False  
    black_lost = False
    num_white_pieces = 12
    num_black_pieces = 12
    while True:
        end = end_game(board)
    
        num_white_pieces, num_black_pieces = count_pieces(board)

        print_board(board)
        if end[1] == 0 or (white_lost and num_white_pieces < num_black_pieces):	
            # black wins
            player1.score += 1
            
            print('Black won')
            break
        elif end[0] == 0 or (black_lost and num_black_pieces < num_white_pieces): 
            # white wins
            player1.score -= 2
            
            print('White won')
            break
	    # check if we breached the threshold for number of moves	
        elif move_limit[0] == move_limit[1]: 
            #draw, do nothing 
            print('It was a draw')
            
            break
	    # CNN play	
        if turn != 'black' and white.type == 'cpu' and (hold1 != -1 or hold2 != -1): 
            
            predict_one_cross = False
            predict_two_cross = False
            predict_average = False
            predict_fogel = False
            
            # Get moves from CNN, check if they are legal, if not use the minimax algorithm to determine the next
            new_board = []

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
            new_board.shape = (8,4)
           
            moves_list, probs = predict_move.predict_cnn(new_board)            
            #print(moves_list)
            can_make_move = False
            for m in moves_list:
                if m[0] != '' and m[1] !='' and m[0] < 32 and m[1] < 32 and m[0] > -1 and m[1] > -1:
                    can_make_move = can_move(board_dict[m[0]],board_dict[m[1]], board)
                    if can_make_move:
                        # make move
                        make_move(board_dict[m[0]],board_dict[m[1]], board)
                    break
            
            if not can_make_move:
                hold1 = cpu_play(-1, white) # white cpu turn

            move_limit[1] += 1 # add to move limit
            end_turn()
            print('White made a move')
            if hold1 == -1:
                white_lost = True
        elif turn != 'white' and black.type == 'cpu' and (hold1 != -1 or hold2 != -1): 

            if player1List == 'one_point':
                predict_one_cross = True
                predict_two_cross = False
                predict_average = False
                predict_fogel = False
            elif player1List == 'two_point':
                predict_one_cross = False
                predict_two_cross = True
                predict_average = False
                predict_fogel = False
            elif player1List == 'average':
                predict_one_cross = False
                predict_two_cross = False
                predict_average = True
                predict_fogel = False
            elif player1List == 'fogel':
                predict_one_cross = False
                predict_two_cross = False
                predict_average = False
                predict_fogel = True


            hold2 = cpu_play(player1.number, black) # black cpu turn
            move_limit[1] += 1 # add to move limit
            print('Black made a move')
            if hold2 == -1:
                black_lost = True


# Counts the number of pieces for each player on the board
def count_pieces(board):
    white_pieces = 0
    black_pieces = 0
    for i in range(8):
        for j in range(8):
            if board[i][j] != 0:
                if board[i][j].color == 'white':
                    white_pieces += 1
                elif board[i][j].color == 'black':
                    black_pieces += 1
    return white_pieces, black_pieces

# Prints the board to the console
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

# Initialize the weight and bias dimensions
if __name__ == '__main__': 
    
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

        play_game(players[player1], players[player2], "","")  


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
    for i in range(10):
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
    for i in range(10):
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
    for i in range(10):
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
    for i in range(10):
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

one_point_cross_players = []
two_point_cross_players = []
average_cross_players = []
fogel_players = []

def final_tournament():

    global players
    # Reload one point cross players

    for i in range(30):
        player_dict = np.load('one_point_cross_w_mutation/player' + str(i) + '.npz')
        one_point_cross_players.append(predict_move.Evol_Player(i, player_dict['first_layer_weights'], player_dict['first_layer_bias'], player_dict['second_layer_weights'], player_dict['second_layer_bias'], player_dict['third_layer_weights'], player_dict['third_layer_bias']))
    
    # Reload two point cross players
    for i in range(30):
        player_dict = np.load('two_point_cross_w_mutation/player' + str(i) + '.npz')
        two_point_cross_players.append(predict_move.Evol_Player(i, player_dict['first_layer_weights'], player_dict['first_layer_bias'], player_dict['second_layer_weights'], player_dict['second_layer_bias'], player_dict['third_layer_weights'], player_dict['third_layer_bias']))
    
    # Reload average cross players
    for i in range(30):
        player_dict = np.load('average_w_mutation/player' + str(i) + '.npz')
        average_cross_players.append(predict_move.Evol_Player(i, player_dict['first_layer_weights'], player_dict['first_layer_bias'], player_dict['second_layer_weights'], player_dict['second_layer_bias'], player_dict['third_layer_weights'], player_dict['third_layer_bias']))
    
    # Reload Fogel Players
    for i in range(30):
        player_dict = np.load('fogel_players/fogel_player' + str(i) + '.npz')
        fogel_players.append(predict_move.Evol_Player(i, player_dict['first_layer_weights'], player_dict['first_layer_bias'], player_dict['second_layer_weights'], player_dict['second_layer_bias'], player_dict['third_layer_weights'], player_dict['third_layer_bias']))
    

    # Find best out of one point players for final tournament
    for i in range(0,30):
        one_point_cross_players[i].score = 0

    players = one_point_cross_players

    for j in range(0, 30):
        for k in range(0, 5):
            player1 = i 
            player2 = math.floor(random.random() * 30)
            while player1 == player2:
                player2 = math.floor(random.random() * 30)

        play_game(one_point_cross_players[player1], one_point_cross_players[player2], 'one_point', 'one_point') 

    # Find the best player out of all 30
    for i in range(0,30):
        for j in range(0,29-i):
            if one_point_cross_players[i].score < one_point_cross_players[i+1].score:
                one_point_cross_players[i], one_point_cross_players[i+1]  = one_point_cross_players[i+1], one_point_cross_players[i]


    final_one_point_player = one_point_cross_players[0]

    # Find best out of two point players for final tournament
    for i in range(0,30):
        two_point_cross_players[i].score = 0

    players = two_point_cross_players

    for j in range(0, 30):
        for k in range(0, 5):
            player1 = i 
            player2 = math.floor(random.random() * 30)
            while player1 == player2:
                player2 = math.floor(random.random() * 30)

        play_game(two_point_cross_players[player1], two_point_cross_players[player2], 'two_point', 'two_point') 

    # Find the best player out of all 30
    for i in range(0,30):
        for j in range(0,29-i):
            if two_point_cross_players[i].score < two_point_cross_players[i+1].score:
                two_point_cross_players[i], two_point_cross_players[i+1]  = two_point_cross_players[i+1], two_point_cross_players[i]

    final_two_point_player = two_point_cross_players[0]

    # Find best out of average players for final tournament
    for i in range(0,30):
        average_cross_players[i].score = 0

    players = average_cross_players

    for j in range(0, 30):
        for k in range(0, 5):
            player1 = i 
            player2 = math.floor(random.random() * 30)
            while player1 == player2:
                player2 = math.floor(random.random() * 30)

        play_game(average_cross_players[player1], average_cross_players[player2], 'average', 'average') 

    # Find the best player out of all 30
    for i in range(0,30):
        for j in range(0,29-i):
            if average_cross_players[i].score < average_cross_players[i+1].score:
                average_cross_players[i], average_cross_players[i+1]  = average_cross_players[i+1], average_cross_players[i]

    final_average_player = average_cross_players[0]

    # Find best out of fogel players for final tournament
    for i in range(0,30):
        fogel_players[i].score = 0

    players = fogel_players

    for j in range(0, 30):
        for k in range(0, 5):
            player1 = i 
            player2 = math.floor(random.random() * 30)
            while player1 == player2:
                player2 = math.floor(random.random() * 30)

        play_game(fogel_players[player1], fogel_players[player2], 'fogel', 'fogel') 

    # Find the best player out of all 30
    for i in range(0,30):
        for j in range(0,29-i):
            if fogel_players[i].score < fogel_players[i+1].score:
                fogel_players[i], fogel_players[i+1]  = fogel_players[i+1], fogel_players[i]

    final_fogel_player = fogel_players[0]

    one_point_wins = 0
    all_one_point_wins = []
    two_point_wins = 0
    all_two_point_wins = []
    average_wins = 0
    all_average_wins = []
    fogel_wins = 0
    all_fogel_wins = []
    draw = 0

    # FINAL TOURNAMENT PLAY: Play one hundred games per player
    # One vs Two
    a = copy.deepcopy(tournament_play(final_one_point_player,final_two_point_player, 'one_point', 'two_point'))
    all_one_point_wins.append(a)
    b = copy.deepcopy(a)
    hold = b[1]
    b[1] = b[0]
    b[0] = hold
    all_two_point_wins.append(b)
    
    # Two vs One
    a = copy.deepcopy(tournament_play(final_two_point_player, final_one_point_player, 'two_point', 'one_point'))
    all_two_point_wins.append(a)
    b = copy.deepcopy(a)
    hold = b[1]
    b[1] = b[0]
    b[0] = hold
    all_one_point_wins.append(b)

    # One vs Average
    a = copy.deepcopy(tournament_play(final_one_point_player,final_average_player, 'one_point', 'average'))
    all_one_point_wins.append(a)
    b = copy.deepcopy(a)
    hold = b[1]
    b[1] = b[0]
    b[0] = hold
    all_average_wins.append(b)

    # Average vs One
    a = copy.deepcopy(tournament_play(final_average_player, final_one_point_player, 'average','one_point'))
    all_average_wins.append(a)
    b = copy.deepcopy(a)
    hold = b[1]
    b[1] = b[0]
    b[0] = hold
    all_one_point_wins.append(b)


    # One vs Fogel
    a = copy.deepcopy(tournament_play(final_one_point_player,final_fogel_player, 'one_point', 'fogel'))
    all_one_point_wins.append(a)
    b = copy.deepcopy(a)
    hold = b[1]
    b[1] = b[0]
    b[0] = hold
    all_fogel_wins.append(b)

    # Fogel vs One
    a = copy.deepcopy(tournament_play(final_fogel_player, final_one_point_player, 'fogel', 'one_point'))
    all_fogel_wins.append(a)
    b = copy.deepcopy(a)
    hold = b[1]
    b[1] = b[0]
    b[0] = hold
    all_one_point_wins.append(b)

    # Two vs Average
    a = copy.deepcopy(tournament_play(final_two_point_player, final_average_player, 'two_point', 'average'))
    all_two_point_wins.append(a)
    b = copy.deepcopy(a)
    hold = b[1]
    b[1] = b[0]
    b[0] = hold
    all_average_wins.append(b)

    # Average vs Two
    a = copy.deepcopy(tournament_play(final_average_player, final_two_point_player, 'average', 'two_point'))
    all_average_wins.append(a)
    b = copy.deepcopy(a)
    hold = b[1]
    b[1] = b[0]
    b[0] = hold
    all_two_point_wins.append(b)

    # Two vs Fogel
    a = copy.deepcopy(tournament_play(final_two_point_player, final_fogel_player, 'two_point', 'fogel'))
    all_two_point_wins.append(a)
    b = copy.deepcopy(a)
    hold = b[1]
    b[1] = b[0]
    b[0] = hold
    all_fogel_wins.append(b)

    # Fogel vs Two
    a = copy.deepcopy(tournament_play(final_fogel_player, final_two_point_player, 'fogel', 'two_point'))
    all_fogel_wins.append(a)
    b = copy.deepcopy(a)
    hold = b[1]
    b[1] = b[0]
    b[0] = hold
    all_two_point_wins.append(b)

    # Average vs Fogel
    a = copy.deepcopy(tournament_play(final_average_player, final_fogel_player, 'average', 'fogel'))
    all_average_wins.append(a)
    b = copy.deepcopy(a)
    hold = b[1]
    b[1] = b[0]
    b[0] = hold
    all_fogel_wins.append(b)

    # Fogel vs Average
    a = copy.deepcopy(tournament_play(final_fogel_player, final_average_player, 'fogel', 'average'))
    all_fogel_wins.append(a)
    b = copy.deepcopy(a)
    hold = b[1]
    b[1] = b[0]
    b[0] = hold
    all_average_wins.append(b)


    # CNN tournament
    for i in range(50):
        cnn_play_game(one_point_cross_players[0], 'one_point')
        cnn_play_game(one_point_cross_players[0], 'two_point')
        cnn_play_game(one_point_cross_players[0], 'average')
        cnn_play_game(one_point_cross_players[0], 'fogel')
    
    print('-----------------One point record------------------')
    print(all_one_point_wins)
    print('-----------------Two point record------------------')
    print(all_two_point_wins)
    print('-----------------Average record------------------')
    print(all_average_wins)
    print('-----------------Fogel record------------------')
    print(all_fogel_wins)


# Handles the tournament for the EC players
def tournament_play(player1, player2, l1, l2):

    player1wins = 0
    player2wins = 0
    draw = 0

    for i in range(50):
        player1.score = 0
        player2.score = 0
        play_game(player1, player2, l1,l2)
        
        if player1.score == 1:
            player1wins += 1
            print('Player one won')
        elif player2.score == 1:
            player2wins += 1
            print('Player two won')
        else:
            draw += 1
    
    return [player1wins, player2wins, draw]


begin_time = time.time()
fogel_ea()
players = []
one_point_crossover_ga()
players = []
two_point_crossover_ga()
players = []
average_crossover_ga()
final_tournament()
final_time = time.time() - begin_time
print('Done: Ran for ' + str(final_time) + 'seconds')
