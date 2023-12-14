
from Board import Board
import random
import json
from ast import literal_eval
from Board import Board
import matplotlib.pyplot as plt

class Player:
    def set_board(self, the_board):
        self.board = the_board
    
    def game_completed(self):
        pass
    
    def get_next_move(self):
        pass

def reward_function(state_info1, state_info2):
    if state_info2[1] == 0 and state_info2[3] == 0:
        return 500
    if state_info2[0] == 0 and state_info2[2] == 0:
        return -500
    return state_info2[0]-state_info1[0] + 2*(state_info2[2]-state_info1[2])-(state_info2[1]-state_info1[1])-2*(state_info2[3]-state_info1[3])


class Q_Learning_AI(Player):
    def __init__(self, the_player_id, the_learning_rate, the_discount_factor, info_location=None, the_random_move_probability=0, the_board=None):
        self.random_move_probability = the_random_move_probability
        self.learning_rate = the_learning_rate    
        self.discount_factor = the_discount_factor
        self.player_id = the_player_id
        self.board = the_board
        self.pre_last_move_state = None
        self.post_last_move_state = None 
        if not info_location is None:
            self.load_transition_information(info_location)
        else:
            self.transitions = {}

    def set_random_move_probability(self, probability):
        self.random_move_probability = probability


    def set_learning_rate(self, the_learning_rate):
        self.learning_rate = the_learning_rate
    

    def get_states_from_boards_spots(self, boards_spots):
        piece_counters = [[0,0,0,0,0,0,0] for j in range(len(boards_spots))] 
        for k in range(len(boards_spots)):
            for j in range(len(boards_spots[k])):
                for i in range(len(boards_spots[k][j])):
                    if boards_spots[k][j][i] != 0:
                        piece_counters[k][boards_spots[k][j][i]-1] = piece_counters[k][boards_spots[k][j][i]-1] + 1
                        if (self.player_id and (boards_spots[k][j][i] == 1 or boards_spots[k][j][i] == 3)) or (not self.player_id and (boards_spots[k][j][i] == 2 or boards_spots[k][j][i] == 4)):
                            if i==0 and j%2==0:
                                piece_counters[k][4] = piece_counters[k][4] + 1
                            elif i==3 and j%2==1:
                                piece_counters[k][4] = piece_counters[k][4] + 1
                                
                            piece_counters[k][5] = piece_counters[k][5] + j
                        else: 
                            piece_counters[k][6] = piece_counters[k][6] + j
            
            if piece_counters[k][0] + piece_counters[k][2] != 0:
                piece_counters[k][5] = int(piece_counters[k][5] / (piece_counters[k][0] + piece_counters[k][2]))
            else:
                piece_counters[k][5] = 0
            if piece_counters[k][1] + piece_counters[k][3] != 0:
                piece_counters[k][6] = int(piece_counters[k][6] / (piece_counters[k][1] + piece_counters[k][3]))
            else:
                piece_counters[k][6] = 0

        return [tuple(counter) for counter in piece_counters]
                                 

    def get_desired_transition_between_states(self, possible_state_array, initial_transition_value=10):
        cur_state = tuple(self.get_states_from_boards_spots([self.board.spots])[0])
        done_transitions = {}
        for state in possible_state_array:#%%%%%%%%%%%%%%%%%%%%%% FOR (1)
            if done_transitions.get((cur_state, tuple(state))) is None:
                if self.transitions.get((cur_state, tuple(state))) is None:
                    self.transitions.update({(cur_state, tuple(state)):initial_transition_value})
                done_transitions.update({(cur_state, tuple(state)):self.transitions.get((cur_state, tuple(state)))})
                
            
        if random != 0 and random.random() < self.random_move_probability:
            try:
                return list(done_transitions.keys())[random.randint(0, len(done_transitions)-1)]
            except:   
                return []
    
        try:
            reverse_dict = {j:i for i,j in done_transitions.items()}
            return reverse_dict.get(max(reverse_dict))
        except:
            return []    
   
   
    def game_completed(self):
        cur_state = self.get_states_from_boards_spots([self.board.spots])[0]
        transition = (self.pre_last_move_state ,self.post_last_move_state)

        self.transitions[transition] = self.transitions[transition] + self.learning_rate * reward_function(transition[0],cur_state)

        self.pre_last_move_state = None
        self.post_last_move_state = None



    def get_transitions_information(self):
        start_of_transitions = {}
        max_value = float("-inf")
        min_value = float("inf")
        total_value = 0
        for k,v in self.transitions.items():
            if start_of_transitions.get(k[0]) is None:
                start_of_transitions.update({k[0]:0})
            if v > max_value:
                max_value = v
            if v < min_value:
                min_value = v
            total_value = total_value + v
            
        return [len(self.transitions), len(start_of_transitions), float(total_value/len(self.transitions)), max_value, min_value]
    
    
    def print_transition_information(self, info):
        print("Total number of transitions: ".ljust(35), info[0])        
        print("Total number of visited states: ".ljust(35), info[1])
        print("Average value for transition: ".ljust(35), info[2])
        print("Maximum value for transition: ".ljust(35), info[3])
        print("Minimum value for transition: ".ljust(35), info[4])
    
        
    def save_transition_information(self, file_name="data.json"):
        with open(file_name, 'w') as fp:
            json.dump({str(k): v for k,v in self.transitions.items()}, fp)
        
        
    def load_transition_information(self, file_name):
        with open(file_name, 'r') as fp:
            self.transitions = {literal_eval(k): v for k,v in json.load(fp).items()}
        
        
    def get_optimal_potential_value(self, depth):
        answer = float("-inf")
        cur_state = self.get_states_from_boards_spots([self.board.spots])[0]
        for k,v in self.transitions.items():
            if v > answer and k[0] == cur_state:
                answer = v
        
        if answer == float("-inf"):
            return None
        return answer



    def get_next_move(self):#, new_board):
        if self.pre_last_move_state is not None:
            cur_state = self.get_states_from_boards_spots([self.board.spots])[0]
    
            transition = (self.pre_last_move_state ,self.post_last_move_state)
            try:
                max_future_state = self.get_optimal_potential_value(1)
                self.transitions[transition] = self.transitions[transition] + self.learning_rate * (reward_function(transition[0],cur_state)+ self.discount_factor* max_future_state - self.transitions[transition])
            except:
                self.transitions[transition] = self.transitions[transition] + self.learning_rate * (reward_function(transition[0],cur_state))
        
        self.pre_last_move_state = self.get_states_from_boards_spots([self.board.spots])[0]
        possible_next_moves = self.board.get_possible_next_moves()
        possible_next_states = self.get_states_from_boards_spots(self.board.get_potential_spots_from_moves(possible_next_moves))
        
        self.post_last_move_state = self.get_desired_transition_between_states(possible_next_states)[1]   
        
        considered_moves = []
        for j in range(len(possible_next_states)):
            if tuple(possible_next_states[j]) == self.post_last_move_state:
                considered_moves.append(possible_next_moves[j])
                
        return considered_moves[random.randint(0,len(considered_moves)-1)]

def get_number_of_pieces_and_kings(spots, player_id=None):
    piece_counter = [0,0,0,0]  
    for row in spots:
        for element in row:
            if element != 0:
                piece_counter[element-1] = piece_counter[element-1] + 1
    
    if player_id == True:
        return [piece_counter[0], piece_counter[2]]
    elif player_id == False:
        return [piece_counter[1], piece_counter[3]]
    else:
        return piece_counter
    

class Alpha_beta(Player):
    def __init__(self, the_player_id, the_depth, the_board=None):
        self.board = the_board
        self.depth = the_depth
        self.player_id = the_player_id

    def alpha_beta(self, board, depth, alpha, beta, maximizing_player):
        if board.is_game_over():
            if get_number_of_pieces_and_kings(board.spots, board.player_turn) == [0,0]:
                if maximizing_player:
                    #Using integers instead of float("inf") so it's less than float("inf") not equal to
                    return -10000000, None
                else:
                    return 10000000, None
            elif get_number_of_pieces_and_kings(board.spots, not board.player_turn) == [0,0]:
                if maximizing_player:
                    return 1000000, None
                else:
                    return -1000000, None
            else:
                return 0, None

        if depth == 0:
            players_info = get_number_of_pieces_and_kings(board.spots)
            if board.player_turn != maximizing_player:
                return  players_info[1] + 2 * players_info[3] - (players_info[0] + 2 * players_info[2]), None
            return  players_info[0] + 2 * players_info[2] - (players_info[1] + 2 * players_info[3]), None
        possible_moves = board.get_possible_next_moves()

        potential_spots = board.get_potential_spots_from_moves(possible_moves)
        desired_move_index = None
        if maximizing_player:
            v = float('-inf')
            for j in range(len(potential_spots)):
                cur_board = Board(potential_spots[j], not board.player_turn)
                alpha_beta_results = self.alpha_beta(cur_board, depth - 1, alpha, beta, False)
                if v < alpha_beta_results[0]: 
                    v = alpha_beta_results[0]
                    alpha = max(alpha, v)
                    desired_move_index = j
                if beta <= alpha: 
                    break
            if desired_move_index is None:
                return v, None
            return v, possible_moves[desired_move_index]
        else:
            v = float('inf')
            for j in range(len(potential_spots)):
                cur_board = Board(potential_spots[j], not board.player_turn)
                alpha_beta_results = self.alpha_beta(cur_board, depth - 1, alpha, beta, True)
                if v > alpha_beta_results[0]:  
                    v = alpha_beta_results[0]
                    desired_move_index = j
                    beta = min(beta, v)
                if beta <= alpha:
                    break
            if desired_move_index is None:
                return v, None
            return v, possible_moves[desired_move_index]
    
    def get_next_move(self):
        return self.alpha_beta(self.board, self.depth, float('-inf'), float('inf'), self.player_id)[1]
        

def play_n_games(player1, player2, num_games, move_limit):
    game_board = Board()
    player1.set_board(game_board)
    player2.set_board(game_board)
     
    players_move = player1
    outcome_counter = [[-1,-1,-1,-1,-1,-1] for j in range(num_games)] 
    for j in range(num_games):
        #print(j)
        move_counter = 0
        while not game_board.is_game_over() and move_counter < move_limit:
            game_board.make_move(players_move.get_next_move())
             
            move_counter = move_counter + 1
            if players_move is player1:
                players_move = player2
            else:
                players_move = player1
            game_board.print_board()
            print("=================================")
        else:
            piece_counter = get_number_of_pieces_and_kings(game_board.spots)
            if piece_counter[0] != 0 or piece_counter[2] != 0:
                if piece_counter[1] != 0 or piece_counter[3] != 0:
                    if move_counter == move_limit:
                        outcome_counter[j][0] = 3
                    else:
                        outcome_counter[j][0] = 2
                else:
                    outcome_counter[j][0] = 0
            else:
                outcome_counter[j][0] = 1
            outcome_counter[j][1] = move_counter
            outcome_counter[j][2] = piece_counter[0]
            outcome_counter[j][3] = piece_counter[1]
            outcome_counter[j][4] = piece_counter[2]
            outcome_counter[j][5] = piece_counter[3]
             
            player1.game_completed()
            player2.game_completed()
            game_board.print_board()
            game_board.reset_board()
     
    return outcome_counter


def pretty_outcome_display(outcomes):
    game_wins = [0,0,0,0]
    total_moves = 0
    max_moves_made = float("-inf")
    min_moves_made = float("inf")
    for outcome in outcomes:
        total_moves = total_moves + outcome[1]
        if outcome[1] < min_moves_made:
            min_moves_made = outcome[1]
        if outcome[1] > max_moves_made:
            max_moves_made = outcome[1]
        
        game_wins[outcome[0]] = game_wins[outcome[0]] + 1
    
    print("Games Played: ".ljust(35), len(outcomes))
    print("Player 1 wins: ".ljust(35), game_wins[0])
    print("Player 2 wins: ".ljust(35), game_wins[1])
    print("Games exceeded move limit: ".ljust(35), game_wins[3])
    print("Games tied: ".ljust(35), game_wins[2])
    print("Total moves made: ".ljust(35), total_moves)  
    print("Average moves made: ".ljust(35), total_moves/len(outcomes))
    print("Max moves made: ".ljust(35), max_moves_made)
    print("Min moves made: ".ljust(35), min_moves_made)
    

def plot_end_game_information(outcome, interval, title="End of Game Results"):
    player1_wins = [0 for _ in range(int(len(outcome)/interval))]
    player2_wins = [0 for _ in range(int(len(outcome)/interval))]
    ties = [0 for _ in range(int(len(outcome)/interval))]
    move_limit = [0 for _ in range(int(len(outcome)/interval))]
    
    for j in range(int(len(outcome)/interval)):
        for i in range(interval):
            if outcome[j*interval + i][0] == 0:
                player1_wins[j] = player1_wins[j] + 1
            elif outcome[j*interval + i][0] == 1:
                player2_wins[j] = player2_wins[j] + 1
            elif outcome[j*interval + i][0] == 2:
                ties[j] = ties[j] + 1
            else:
                move_limit[j] = move_limit[j] + 1
                
    plt.figure(title)
    
    p1_win_graph, = plt.plot(player1_wins, label = "Player 1 wins")
    p2_win_graph, = plt.plot(player2_wins, label = "Player 2 wins")
    tie_graph, = plt.plot(ties, label = "Ties")
    move_limit_graph, = plt.plot(move_limit, label = "Move limit reached")
    
    plt.ylabel("Occurance per " +str(interval) + " games")
    plt.xlabel("Interval")
    
    plt.legend(handles=[p1_win_graph, p2_win_graph, tie_graph, move_limit_graph])


 
LEARNING_RATE = .005  
DISCOUNT_FACTOR = .3
NUM_GAMES_TO_TRAIN = 1
NUM_TRAINING_ROUNDS = 10
NUM_VALIDATION_GAMES = 1
NUM_GAMES_TO_TEST = 0
TRAINING_RANDOM_MOVE_PROBABILITY = .25
ALPHA_BETA_DEPTH = 2
TRAINING_MOVE_LIMIT = 500
VALIDATION_MOVE_LIMIT = 1000
TESTING_MOVE_LIMIT = 2000
PLAYER1 = Q_Learning_AI(True, LEARNING_RATE, DISCOUNT_FACTOR, the_random_move_probability=TRAINING_RANDOM_MOVE_PROBABILITY)#, info_location="data.json")
PLAYER2 = Alpha_beta(False, ALPHA_BETA_DEPTH)
PLAYER4 = Alpha_beta(False, 3)

training_info = []
validation_info = []
for j in range(NUM_TRAINING_ROUNDS):
    training_info.extend(play_n_games(PLAYER1, PLAYER2, NUM_GAMES_TO_TRAIN, TRAINING_MOVE_LIMIT))
    PLAYER1.print_transition_information(PLAYER1.get_transitions_information())
    PLAYER1.set_random_move_probability(0)
    PLAYER1.set_learning_rate(0)
    validation_info.extend(play_n_games(PLAYER1, PLAYER4, NUM_VALIDATION_GAMES, VALIDATION_MOVE_LIMIT))
    print("Round " + str(j+1) + " completed!")
    PLAYER1.set_random_move_probability(TRAINING_RANDOM_MOVE_PROBABILITY)
    PLAYER1.set_learning_rate(LEARNING_RATE)
    print("")
    PLAYER1.save_transition_information()
 
plt.show()
 
pretty_outcome_display(training_info)
print("")
pretty_outcome_display(validation_info)
PLAYER1.print_transition_information(PLAYER1.get_transitions_information())
PLAYER1.save_transition_information()


def switch_board_players(board):

    board.player_turn = not board.player_turn
    for row_index in len(board.spots):
        for col_index in len(board.spots[row_index]):
            if board.spots[row_index][col_index] == 1:
                board.spots[row_index][col_index] = 2
            elif board.spots[row_index][col_index] == 2:
                board.spots[row_index][col_index] = 1
            elif board.spots[row_index][col_index] == 3:
                board.spots[row_index][col_index] = 4
            elif board.spots[row_index][col_index] == 4:
                board.spots[row_index][col_index] = 3

def test_board_functions_not_next_move():
    old_spots = [[1,4,1,1],[1,0,1,1],[1,1,0,1],[0,0,0,0],[0,0,0,0],[0,1,2,2],[2,2,2,2],[2,2,2,2]]
    
    board = Board()
    alpha_beta_ai = Alpha_beta(False, 6)
    alpha_beta_ai.set_board(board)
    
    alpha_beta_ai.get_next_move()
    board.make_move([[2,0],[3,0]])
    board.make_move([[5,0],[4,1]])
    alpha_beta_ai.get_next_move()
    board.make_move([[2,2],[3,1]])
    board.make_move([[1,0],[2,0]])
    alpha_beta_ai.get_next_move()
    board.make_move([[0,1],[1,0]])
    board.make_move([[4,1],[2,2],[0,1]])
    alpha_beta_ai.get_next_move()
    board.make_move([[5,1],[4,1]])
    board.make_move([[3,0],[5,1]])
    alpha_beta_ai.get_next_move()

    if board.spots == old_spots:
        pass
    else:
        pass

