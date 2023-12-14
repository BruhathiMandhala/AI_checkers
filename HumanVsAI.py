import copy
import sys
import csv
import time
import traceback
from collections import deque
from multiprocessing import Pool
from Board import open_file, load_weights
from Board import *
from agents import *
import numpy as np

NUM_WEIGHTS_REM = 5
WEIGHTS_SAVE_FREQ = 50
WRITE_FREQ = 100
TEST_FREQ = 100
TEST_GAMES = 100
NOTIFY_FREQ = 50
CHANGE_AGENT_FREQ = 10

class GameState:

    def __init__(self, prev_state=None, the_player_turn=True):
        if prev_state is None:
            prev_spots = None
        else:
            prev_spots = copy.deepcopy(prev_state.board.spots)

        self.board = Board(prev_spots, the_player_turn)
        self.max_moves_done = False

    def get_num_agents(self):
        return 2

    def get_legal_actions(self):
        return self.board.get_possible_next_moves()


    def generate_successor(self, action, switch_player_turn=True):
        successor_state = GameState(self, self.board.player_turn)
        successor_state.board.make_move(action, switch_player_turn)

        return successor_state

    def is_first_agent_turn(self):
        return self.board.player_turn


    def is_game_over(self):
        return self.board.is_game_over() or self.max_moves_done

    def is_first_agent_win(self):
        if self.max_moves_done:
            return False

        if not self.is_game_over() or self.is_first_agent_turn():
            return False

        return True

    def is_second_agent_win(self):
        if self.max_moves_done:
            return False

        if not self.is_game_over() or not self.is_first_agent_turn():
            return False

        return True


    def print_board(self):
        self.board.print_board()


    def player_info(self):
        return self.board.P1 if self.board.player_turn else self.board.P2


    def player_symbol(self, index):
        if index == 1:
            return self.board.P1_SYMBOL
        else:
            return self.board.P2_SYMBOL


    def get_pieces_and_kings(self, player=None):
        spots = self.board.spots
        count = [0,0,0,0]   
        for x in spots:
            for y in x:
                if y != 0:
                    count[y-1] = count[y-1] + 1

        if player is not None:
            if player:
                return [count[0], count[2]]  #Player 1
            else:
                return [count[1], count[3]]  #Player 2
        else:
            return count

    def set_max_moves_done(self, done=True):
        self.max_moves_done = done

    def num_attacks(self):
        piece_locations = self.board.get_piece_locations()

        capture_moves = reduce(lambda x, y: x + y, list(map(self.board.get_capture_moves, piece_locations)), [])
        num_pieces_in_attack = 0

        pieces_in_attack = set()
        for move in capture_moves:
            for i, loc in enumerate(move):
                if (i+1) < len(move):
                    loc_2 = move[i+1]
                    pieces_in_attack.add(( (loc_2[0] + loc[0]) / 2, (loc_2[1] + loc[1]) / 2 + loc[0] % 2,))

        num_pieces_in_attack = len(pieces_in_attack)
        return num_pieces_in_attack

class ClassicGameRules:
    def __init__(self, max_moves=200):
        self.max_moves = max_moves
        self.quiet = False

    def new_game(self, first_agent, second_agent, first_agent_turn, quiet=False):
        init_state = GameState(the_player_turn=first_agent_turn)

        self.quiet = quiet
        game = Game(first_agent, second_agent, init_state, self)

        return game


def load_agent(agent_type, agent_learn, weights=None, depth=3):
    if agent_type == 'k':
        return Human()
    elif agent_type == 'ab':
        return AlphaBetaAgent(depth=depth)
    elif agent_type == 'ql':
        is_learning_agent = True if agent_learn else False
        return QLearningAgent(is_learning_agent=is_learning_agent, weights=weights)
    else:
        raise Exception('Invalid agent ' + str(agent_type))


def default(str):
    return str + ' [Default: %default]'


def read_command(argv):
    options= {'num_games': 1, 'first_agent': 'ab', 'first_agent_learn': 1, 'second_agent': 'k', 'second_agent_learn': 1, 'turn': 1, 'update_param': 0, 'quiet': 0, 'first_save': './data/first_save', 'second_save': './data/second_save', 'first_weights': './data/first_weights', 'second_weights': './data/second_weights', 'first_results': './data/first_results', 'second_results': './data/second_results', 'first_m_results': './data/first_m_results', 'second_m_results': './data/second_m_results', 'play_against_self': 0}
    #print(options)
    args = dict()
    args['num_games'] = options['num_games']
    first_weights = load_weights(options['first_weights'])
    args['first_agent'] = load_agent(options['first_agent'], options['first_agent_learn'], first_weights)
    second_weights = load_weights(options['second_weights'])
    args['second_agent'] = load_agent(options['second_agent'], options['second_agent_learn'], second_weights)
    args['first_agent_turn'] = options['turn'] == 1
    args['update_param'] = options['update_param']
    args['quiet'] = True if options['quiet'] else False
    args['first_file_name'] = options['first_save']
    args['second_file_name'] = options['second_save']
    args['first_weights_file_name'] = options['first_weights']
    args['second_weights_file_name'] = options['second_weights']
    args['first_result_file_name'] = options['first_results']
    args['second_result_file_name'] = options['second_results']
    args['first_m_result_file_name'] = options['first_m_results']
    args['second_m_result_file_name'] = options['second_m_results']
    args['play_against_self'] = options['play_against_self'] == 1
    return args


def run_test(rules, first_agent, second_agent, first_agent_turn, quiet=True):
    game = rules.new_game(first_agent, second_agent, first_agent_turn, quiet=True)
    num_moves, game_state = game.run()
    return num_moves, game_state


def multiprocess(rules, first_agent, second_agent, first_agent_turn, quiet=True):
    results = []
    result_f = [[], []]
    result_s = [[], []]
    pool = Pool(4)
    kwds = {'quiet': quiet}
    for i in range(TEST_GAMES):
        results.append(pool.apply_async(run_test, [rules, first_agent, second_agent, 
            first_agent_turn], kwds))
    pool.close()
    pool.join()
    for result in results:
        val = result.get()
        num_moves, game_state = val[0], val[1]
    return result_f, result_s


def run_games(first_agent, second_agent, first_agent_turn, num_games, update_param=0, quiet=False, 
                first_file_name="./data/first_save", second_file_name="./data/second_save", 
                first_weights_file_name="./data/first_weights", 
                second_weights_file_name="./data/second_weights",
                first_result_file_name="./data/first_results",
                second_result_file_name="./data/second_results", 
                first_m_result_file_name="./data/first_m_results",
                second_m_result_file_name="./data/second_m_results", 
                play_against_self=False):
    try:
        print(first_agent)
        print(second_agent)
        print('starting game', 0)
        for i in range(num_games):

            if (i+1) % NOTIFY_FREQ == 0:
                print('Starting game', (i+1))
            rules = ClassicGameRules()
            game = rules.new_game(first_agent, second_agent, first_agent_turn, quiet=quiet)

            num_moves, game_state = game.run()
            if (i+1) % TEST_FREQ == 0:
                result_f = []
                result_s = []
                print('strting', TEST_GAMES, 'tests')

                result_f, result_s = \
                multiprocess(rules, first_agent, second_agent, first_agent_turn, quiet=True)
    except Exception as e:
        print(sys.exc_info()[0])
        traceback.print_tb(e.__traceback__)


if __name__ == '__main__':

    start_time = time.time()
    args = read_command(['-f','ab','-s','k'])
    #print(args)
    run_games(**args)
    #print(**args)
    print(time.time() - start_time)