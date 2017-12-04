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

player_count = 0
players = []

class Board(object):

    global jumps, empty, odd_list

    def __init__(self):
        self.state = pd.read_csv(filepath_or_buffer='board_init.csv', header=-1, index_col=None)
        self.invalid_attempts = 0

    def board_state(self, player_type):
        if player_type == 'white':
            return -self.state.iloc[::-1, ::-1]
        elif player_type == 'black':
            return self.state

    def print_board(self):

        print('  32  31  30  29')
        print('28  27  26  25')
        print('  24  23  22  21')
        print('20  19  18  17')
        print('  16  15  14  13')
        print('12  11  10  09')
        print('  08  07  06  05')
        print('04  03  02  01')
        print('\n')
        for j in range(8):
            for i in range(4):
                if j % 2 == 0:
                    print(' ', end=""),
                if self.state[3 - i][7 - j] == 1:
                    print('x', end=""),
                elif self.state[3 - i][7 - j] == 3:
                    print('X', end=""),
                elif self.state[3 - i][7 - j] == 0:
                    print('-', end=""),
                elif self.state[3 - i][7 - j] == -1:
                    print('o', end=""),
                else:
                    print('O', end=""),
                if j % 2 != 0:
                    print(' ', end=""),
            print('')

    def find_jumps(self, player_type):

        valid_jumps = list()

        if player_type == 'black':
            king_value = black_king
            chkr_value = black_chkr
            chkr_directions = [1, 2]
        else:
            king_value = white_king
            chkr_value = white_chkr
            chkr_directions = [0, 3]

        board_state = self.state.as_matrix()
        board_state = np.reshape(board_state, (32,))

        for position in range(32):
            piece = board_state[position]
            neighbors_list = neighbors[position]
            next_neighbors_list = next_neighbors[position]

            if piece == chkr_value:
                for direction in chkr_directions:
                    neighbor = neighbors_list[direction]
                    next_neighbor = next_neighbors_list[direction]
                    if neighbor == iv or next_neighbor == iv:
                        pass
                    elif board_state[next_neighbor] == empty and (board_state[neighbor] == -chkr_value or board_state[neighbor] == -king_value):
                        valid_jumps.append([position, next_neighbor])

            elif piece == king_value:
                for direction in range(4):
                    neighbor = neighbors_list[direction]
                    next_neighbor = next_neighbors_list[direction]
                    if neighbor == iv or next_neighbor == iv:
                        pass
                    elif board_state[next_neighbor] == empty and (board_state[neighbor] == -chkr_value or board_state[neighbor] == -king_value):
                        valid_jumps.append([position, next_neighbor])

        print(valid_jumps)
        return valid_jumps

    def get_positions(self, move, player_type):

        # Extract starting position, and direction to move
        ind = np.argwhere(move == 1)[0]
        position = ind[0]
        direction = ind[1]

        jumps_available = self.find_jumps(player_type=player_type)

        neighbor = neighbors[position][direction]
        next_neighbor = next_neighbors[position][direction]

        if [position, next_neighbor] in jumps_available:
            return position, next_neighbor, 'jump'
        else:
            return position, neighbor, 'standard'

    def generate_move(self, playernum, player_type, output_type):

        board_state = self.state
        if playernum != -1:
            holddir = player_dir + str(playernum) + 'model.ckpt'
            moves, probs = predict_move.predict_cnn(board_state, output=output_type, params_dir=holddir)
        else:
            moves, probs = predict_move.predict_cnn(board_state, output=output_type, params_dir=params_dir)
        moves_list = list()

        for i in range(1, 6):
            ind = np.argwhere(moves == i)[0]
            move = np.zeros([32, 4])
            move[ind[0], ind[1]] = 1
            pos_init, pos_final, move_type = self.get_positions(move, player_type=player_type)
            moves_list.append([pos_init, pos_final])

        return moves_list, probs

    def update(self, positions, player_type, move_type):

        # Extract the initial and final positions into ints
        pos_init, pos_final = positions[0], positions[1]

        if pos_init == '' or pos_final == '':
            return True

        if player_type == 'black':
            king_pos = black_king_pos
            king_value = black_king
            chkr_value = black_chkr
            pos_init, pos_final = int(pos_init), int(pos_final)
        else:
            king_pos = white_king_pos
            king_value = white_king
            chkr_value = white_chkr
            pos_init, pos_final = int(pos_init) - 1, int(pos_final) - 1

        # print(pos_init, pos_final)
        board_vec = self.state.copy()
        board_vec = np.reshape(board_vec.as_matrix(), (32,))

        if (board_vec[pos_init] == chkr_value or board_vec[pos_init] == king_value) and board_vec[pos_final] == empty:
            board_vec[pos_final] = board_vec[pos_init]
            board_vec[pos_init] = empty

            # Assign kings
            if pos_final in king_pos:
                board_vec[pos_final] = king_value

            # Remove eliminated pieces
            if move_type == 'jump':
                eliminated = int(jumps.iloc[pos_init, pos_final])
                print('Position eliminated: %d' % (eliminated + 1))
                assert board_vec[eliminated] == -chkr_value or -king_value
                board_vec[eliminated] = empty

            # Update the board
            board_vec = pd.DataFrame(np.reshape(board_vec, (8, 4)))
            self.state = board_vec
            return False

        else:
            return True

if __name__ == '__main__':
    # Define board entries and valid positions
    empty = 0
    black_chkr = 1
    black_king = 3
    black_king_pos = [28, 29, 30, 31]
    white_chkr = -black_chkr
    white_king = -black_king
    white_king_pos = [0, 1, 2, 3]
    valid_positions = range(32)
    odd_list = [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27]
    even_list = [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31]
    jumps = pd.read_csv(filepath_or_buffer='jumps.csv', header=-1, index_col=None)
    params_dir = 'parameters/convnet_150k_full/model.ckpt-150001'
    player_dir = 'test_players/'
    # Entries for neighbors are lists, with indices corresponding to direction as defined in parser_v7.py ...
    iv = ''
    neighbors = {0: [iv, 5, 4, iv],
                 1: [iv, 6, 5, iv],
                 2: [iv, 7, 6, iv],
                 3: [iv, iv, 7, iv],
                 4: [0, 8, iv, iv],
                 5: [1, 9, 8, 0],
                 6: [2, 10, 9, 1],
                 7: [3, 11, 10, 2],
                 8: [5, 13, 12, 4],
                 9: [6, 14, 13, 5],
                 10: [7, 15, 14, 6],
                 11: [iv, iv, 15, 7],
                 12: [8, 16, iv, iv],
                 13: [9, 17, 16, 8],
                 14: [10, 18, 17, 9],
                 15: [11, 19, 18, 10],
                 16: [13, 21, 20, 12],
                 17: [14, 22, 21, 13],
                 18: [15, 23, 22, 14],
                 19: [iv, iv, 23, 15],
                 20: [16, 24, iv, iv],
                 21: [17, 25, 24, 16],
                 22: [18, 26, 25, 17],
                 23: [19, 27, 26, 18],
                 24: [21, 29, 28, 20],
                 25: [22, 30, 29, 21],
                 26: [23, 31, 30, 22],
                 27: [iv, iv, 31, 23],
                 28: [24, iv, iv, iv],
                 29: [25, iv, iv, 24],
                 30: [26, iv, iv, 25],
                 31: [27, iv, iv, 26]
                 }

    next_neighbors = {0: [iv, 9, iv, iv],
                      1: [iv, 10, 8, iv],
                      2: [iv, 11, 9, iv],
                      3: [iv, iv, 10, iv],
                      4: [iv, 13, iv, iv],
                      5: [iv, 14, 12, iv],
                      6: [iv, 15, 13, iv],
                      7: [iv, iv, 14, iv],
                      8: [1, 17, iv, iv],
                      9: [2, 18, 16, 0],
                      10: [3, 19, 17, 1],
                      11: [iv, iv, 18, 2],
                      12: [5, 21, iv, iv],
                      13: [6, 22, 20, 4],
                      14: [7, 23, 21, 5],
                      15: [iv, iv, 22, 6],
                      16: [9, 25, iv, iv],
                      17: [10, 26, 24, 8],
                      18: [11, 27, 25, 9],
                      19: [iv, iv, 26, 10],
                      20: [13, 29, iv, iv],
                      21: [14, 30, 28, 12],
                      22: [15, 31, 29, 13],
                      23: [iv, iv, 30, 14],
                      24: [17, iv, iv, iv],
                      25: [18, iv, iv, 16],
                      26: [19, iv, iv, 17],
                      27: [iv, iv, iv, 18],
                      28: [21, iv, iv, iv],
                      29: [22, iv, iv, 20],
                      30: [23, iv, iv, 21],
                      31: [iv, iv, iv, 22]
                      }


def evolutionary_play(player1, player2):

    # Alpha-numeric encoding of player turn: white = 1, black = -1
    turn = -1

    # Count number of invalid move attempts
    invalid_move_attempts = 0
    jumps_not_predicted = 0
    move_count = 0
    game_aborted = False

    player1num = player1.number
    player2num = player2.number

    # Initialize board object
    board = Board()

    #print('====================================================================================================================================================')
    #print('CNN Checkers Engine')
    #print('Created by Chris Larson')
    #print('All Rights Reserved (2016)')
    #print('\n')
    #print('You are playing the white pieces, computer is playing black.')
    #print('There is no GUI for this game. Feel free to run an external program in 2-player mode alongside this game.')
    #print('\n')
    #print('Procedure:')
    #print('1. The computer generates its own moves and prints them to the screen. The user can execute these moves in an external program.')
    #print("2. The computer will then prompt the user to enter a sequence of board positions separated by a comma, indicating the move they want to make.")
    #print("For example, the entry '7, 10' would indicate moving the piece located at position 7 to position 10. The entry '7, 14, 23' would")
    #print("indicate a mulitiple-jump move by the piece located at position 7 eliminating the opposing checkers at positions 10 and 18.")
    #print("3. To end the game, specify the result as follows: 'black wins' for a black win, 'white wins' for a white win, or 'draw' for a draw.")
    #print('\n')

    # Start game
    #raw_input("To begin, press Enter:")
    end_game = False
    winner = ''
    while True:

        # White turn, player1
        if turn == 1:
            board.print_board()
            player_type = 'white'



            # Call model to generate move
            moves_list, probs = board.generate_move(playernum = player1num, player_type=player_type, output_type='top-5')
            #print(np.array(moves_list) + 1)
            print(probs)

            # Check for available jumps, cross check with moves
            available_jumps = board.find_jumps(player_type=player_type)

            first_move = True

            # Handles situation where there is a jump available to black
            if len(available_jumps) > 0:

                move_type = 'jump'
                jump_available = True

                while jump_available:

                    # For one jump available
                    if len(available_jumps) == 1:
                        count = 1
                        move_predicted = False

                        for move in moves_list:
                            if move == available_jumps[0]:
                                # print("There is one jump available. This move was choice %d." % count)
                                move_predicted = True
                                break
                            else:
                                count += 1

                        if not move_predicted:
                            # print('Model did not output the available jumps. Forced move.')
                            jumps_not_predicted += 1

                        initial_position = available_jumps[0][0]
                        if not (first_move or final_position == initial_position):
                            break
                        final_position = available_jumps[0][1]
                        initial_piece = np.reshape(board.state.as_matrix(), (32,))[initial_position]
                        move_illegal = board.update(available_jumps[0], player_type=player_type, move_type=move_type)

                        if move_illegal:
                            # print('Find Jumps function returned invalid move: %s' % (np.array(available_jumps[0]) + 1))
                            game_aborted = True
                            player1.score -=2
                        else:
                            print("Black move: %s" % (np.array(available_jumps[0]) + 1))
                            available_jumps = board.find_jumps(player_type=player_type)
                            final_piece = np.reshape(board.state.as_matrix(), (32,))[final_position]
                            if len(available_jumps) == 0 or final_piece != initial_piece:
                                jump_available = False

                    # When diffent multiple jumps are available
                    else:
                        move_predicted = False
                        for move in moves_list:
                            if move in available_jumps:

                                initial_position = move[0]
                                if not (first_move or final_position == initial_position):
                                    break
                                final_position = move[1]
                                initial_piece = np.reshape(board.state.as_matrix(), (32,))[initial_position]
                                move_illegal = board.update(move, player_type=player_type, move_type=move_type)

                                if move_illegal:
                                    pass
                                    # print('Model and Find jumps function predicted an invalid move: %s' % (np.array(move) + 1))
                                else:
                                    print("Black move: %s" % (np.array(move) + 1))
                                    move_predicted = True
                                    available_jumps = board.find_jumps(player_type=player_type)
                                    final_piece = np.reshape(board.state.as_matrix(), (32,))[final_position]
                                    if len(available_jumps) == 0 or final_piece != initial_piece:
                                        jump_available = False
                                    break

                        if not move_predicted:
                            # print('Model did not output any of the available jumps. Move picked randomly among valid options.')
                            jumps_not_predicted += 1
                            ind = np.random.randint(0, len(available_jumps))

                            initial_position = available_jumps[ind][0]
                            if not (first_move or final_position == initial_position):
                                break
                            final_position = available_jumps[ind][1]
                            initial_piece = np.reshape(board.state.as_matrix(), (32,))[initial_position]
                            move_illegal = board.update(available_jumps[ind], player_type=player_type, move_type=move_type)

                            if move_illegal:
                                # print('Find Jumps function returned invalid move: %s' % (np.array(available_jumps[ind]) + 1))
                                game_aborted = True 
                                player1.score -=2
                            else:
                                available_jumps = board.find_jumps(player_type=player_type)
                                final_piece = np.reshape(board.state.as_matrix(), (32,))[final_position]
                                if len(available_jumps) == 0 or final_piece != initial_piece:
                                    jump_available = False

                    first_move = False


            # For standard moves
            else:
                move_type = 'standard'
                move_illegal = True
                while move_illegal:

                    count = 1
                    for move in moves_list:

                        move_illegal = board.update(move, player_type=player_type, move_type=move_type)

                        if move_illegal:
                            # print('model predicted invalid move (%s)' % (np.array(move) + 1))
                            print(probs[count - 1])
                            invalid_move_attempts += 1
                            count += 1
                        else:
                            print('Black move: %s' % (np.array(move) + 1))
                            break

                    if move_illegal:
                        game_aborted = True
                        player1.score -=2
                        print("The model failed to provide a valid move. Game aborted.")
                        #print(np.array(moves_list) + 1)
                        print(probs)
                        break

            
            try:
                move[::-1][:4][::-1]
            except:
                pass
            else:
                if move[::-1][:4][::-1] == 'wins':
                    winner = 'black'
                    end_game = True
                    break
                elif move == 'draw':
                    winner = 'draw'
                    end_game = True
                    break




        # Black turn
        elif turn == -1:

            print('\n' * 2)
            print('=======================================================')
            print("Black's turn")
            board.print_board()
            player_type = 'black'

            # Call model to generate move
            moves_list, probs = board.generate_move(playernum = player2num, player_type=player_type, output_type='top-5')
            #print(np.array(moves_list) + 1)
            print(probs)

            # Check for available jumps, cross check with moves
            available_jumps = board.find_jumps(player_type=player_type)

            first_move = True

            # Handles situation where there is a jump available to black
            if len(available_jumps) > 0:

                move_type = 'jump'
                jump_available = True

                while jump_available:

                    # For one jump available
                    if len(available_jumps) == 1:
                        count = 1
                        move_predicted = False

                        for move in moves_list:
                            if move == available_jumps[0]:
                                # print("There is one jump available. This move was choice %d." % count)
                                move_predicted = True
                                break
                            else:
                                count += 1

                        if not move_predicted:
                            # print('Model did not output the available jumps. Forced move.')
                            jumps_not_predicted += 1

                        initial_position = available_jumps[0][0]
                        if not (first_move or final_position == initial_position):
                            break
                        final_position = available_jumps[0][1]
                        initial_piece = np.reshape(board.state.as_matrix(), (32,))[initial_position]
                        move_illegal = board.update(available_jumps[0], player_type=player_type, move_type=move_type)

                        if move_illegal:
                            # print('Find Jumps function returned invalid move: %s' % (np.array(available_jumps[0]) + 1))
                            game_aborted = True
                            player2.score -=2
                        else:
                            print("Black move: %s" % (np.array(available_jumps[0]) + 1))
                            available_jumps = board.find_jumps(player_type=player_type)
                            final_piece = np.reshape(board.state.as_matrix(), (32,))[final_position]
                            if len(available_jumps) == 0 or final_piece != initial_piece:
                                jump_available = False

                    # When diffent multiple jumps are available
                    else:
                        move_predicted = False
                        for move in moves_list:
                            if move in available_jumps:

                                initial_position = move[0]
                                if not (first_move or final_position == initial_position):
                                    break
                                final_position = move[1]
                                initial_piece = np.reshape(board.state.as_matrix(), (32,))[initial_position]
                                move_illegal = board.update(move, player_type=player_type, move_type=move_type)

                                if move_illegal:
                                    pass
                                    # print('Model and Find jumps function predicted an invalid move: %s' % (np.array(move) + 1))
                                else:
                                    print("Black move: %s" % (np.array(move) + 1))
                                    move_predicted = True
                                    available_jumps = board.find_jumps(player_type=player_type)
                                    final_piece = np.reshape(board.state.as_matrix(), (32,))[final_position]
                                    if len(available_jumps) == 0 or final_piece != initial_piece:
                                        jump_available = False
                                    break

                        if not move_predicted:
                            # print('Model did not output any of the available jumps. Move picked randomly among valid options.')
                            jumps_not_predicted += 1
                            ind = np.random.randint(0, len(available_jumps))

                            initial_position = available_jumps[ind][0]
                            if not (first_move or final_position == initial_position):
                                break
                            final_position = available_jumps[ind][1]
                            initial_piece = np.reshape(board.state.as_matrix(), (32,))[initial_position]
                            move_illegal = board.update(available_jumps[ind], player_type=player_type, move_type=move_type)

                            if move_illegal:
                                # print('Find Jumps function returned invalid move: %s' % (np.array(available_jumps[ind]) + 1))
                                game_aborted = True
                                player2.score -=2
                            else:
                                available_jumps = board.find_jumps(player_type=player_type)
                                final_piece = np.reshape(board.state.as_matrix(), (32,))[final_position]
                                if len(available_jumps) == 0 or final_piece != initial_piece:
                                    jump_available = False

                    first_move = False

            # For standard moves
            else:
                move_type = 'standard'
                move_illegal = True
                while move_illegal:

                    count = 1
                    for move in moves_list:

                        move_illegal = board.update(move, player_type=player_type, move_type=move_type)

                        if move_illegal:
                            # print('model predicted invalid move (%s)' % (np.array(move) + 1))
                            print(probs[count - 1])
                            invalid_move_attempts += 1
                            count += 1
                        else:
                            print('Black move: %s' % (np.array(move) + 1))
                            break

                    if move_illegal:
                        game_aborted = True
                        print("The model failed to provide a valid move. Game aborted.")
                        player2.score -=2
                        #print(np.array(moves_list) + 1)
                        print(probs)
                        break
            try:
                if move[::-1][:4][::-1] == 'wins':
                    winner = 'black'
                    end_game = True
                    break

                elif move == 'draw':
                    winner = 'draw'
                    end_game = True
                    break
            except:
                pass

        if game_aborted:
            print('Game aborted.')
            break

        if end_game:
            print('The game has ended')
            break
        move_count += 1
        turn *= -1

    # Print out game stats
    end_board = board.state.as_matrix()
    num_black_chkr = len(np.argwhere(end_board == black_chkr))
    num_black_king = len(np.argwhere(end_board == black_king))
    num_white_chkr = len(np.argwhere(end_board == white_chkr))
    num_white_king = len(np.argwhere(end_board == white_king))
    if winner == 'draw':
        print('The game ended in a draw.')
    elif winner == 'white':
        #first player won
        player1.score +=1
        player2.score -=2
    else:
        player1.score -=2
        player2.score +=1
    print('Total number of moves: %d' % move_count)
    print('Remaining white pieces: (checkers: %d, kings: %d)' % (num_white_chkr, num_white_king))
    print('Remaining black pieces: (checkers: %d, kings: %d)' % (num_black_chkr, num_black_king))


#play()

# Create 30 players in the population to begin
def create_players():
    global player_count
    for i in range (0, 30):
        players.append(predict_move.evolutionary_player(player_dir=player_dir, player_count=player_count))
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

            evolutionary_play(players[player1], players[player2])  

def create_offspring():
    
    # rank parents
    # sort players by score
    numbers_not_used = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    hold_players = copy.deepcopy(players)
    
    for i in range(0,30):
        for j in range(0,29-i):
            if hold_players[i].score < hold_players[i+1].score:
                hold_players[i], hold_players[i+1]  = hold_players[i+1], hold_players[i]
                

    for i in range(0,30):
        print(hold_players[i].number)


    parents = {}
    for i in range(0, 15):
        parents[hold_players[i].number] = hold_players[i]
        numbers_not_used.pop(numbers_not_used.index(hold_players[i].number)) #remove the numbers that are being used again as parents
        

    # we need to create the offspring now
    i = 0
    for n in numbers_not_used:
        players[n] = predict_move.fogel_create_offspring(player_dir=player_dir, parent_num = i, offspring_num = n)


#def one_point_crossover_ga():
    
#    create_players()
#    score_players()
#    create_offspring()

   
def fogel_ea():

    create_players()
    for i in range(25):
        #score_players()
        create_offspring()

fogel_ea()

