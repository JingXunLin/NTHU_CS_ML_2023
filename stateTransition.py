from TetrisBattle.envs.tetris_env import TetrisSingleEnv
from TetrisBattle.tetris import Tetris, get_infos, Piece, Buffer, hardDrop, rotateCollide, collideDown, collide, rotate
from TetrisBattle.envs.tetris_interface import TetrisSingleInterface, TetrisDoubleInterface
from TetrisBattle.tetris import Tetris
from TetrisBattle.tetris import Player
from TetrisBattle.envs.tetris_interface import ComEvent
from copy import deepcopy

from TetrisManager import TetrisSaver
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from TetrisBattle.settings import *
import numpy as np
from gym_tetris.ai.QNetwork import QNetwork
#from tqdm.notebook import tqdm

import hashlib
import pickle

EPISODE = 100


class TetrisState():
    
    
    def __init__(self, tetris, board, coords): #Call method: Tetris(Player(info_dict), gridchoice, tetris.get_board())

        self.tetris = tetris
        self.board = board
        self.cols = len(self.board)
        self.rows = len(self.board[0])
        self.buffer = tetris.buffer
        # self.last_piece = None
        # self.piece = None #new
        self.coords = coords
    '''
    def get_org_infos(self,action):
        ob, reward, end, infos = self.env.step(action)
        
        return infos
    '''
    def get_state(self,infos,rows_cleared): #Call method: get_state(infos)   need to add 'from TetrisBattle.tetris import Tetris'
        
        #infos = self.get_org_infos(action)
        states = {'Rows_cleared': 0, 
                'Bumpiness': 0,
                'Holes': 0,
                'Landing_height': 0,
                'Row_transitions': 0,
                'Column_transitions': 0,
                'Cumulative_wells': 0,
                'Eroded_piece_cells': 0,
                'Aggregate_height': 0}
        '''
        infos: {'is_fallen', 'height_sum', 'diff_sum', 'max_height', 'holes', 'n_used_block',
                'scores', 'cleared', 'penalty', 'reward_notdie', 'sent', 'episode'}
        '''
        eroded_piece_cells = 0
        landing_height = 0
        
        
        if self.tetris.is_fallen:
            landing_height = self.rows - self.coords['py']
            eroded_piece_cells = self.tetris.cleared #TO fix    
            
        states['Rows_cleared'] = self.tetris.cleared
        states['Bumpiness'] = self.get_bumpiness()
        states['Holes'] = self.get_hole_count()
        states['Landing_height'] = landing_height
        states['Row_transitions'] = self.get_row_transitions()
        states['Column_transitions'] = self.get_column_transitions()
        states['Cumulative_wells'] = self.get_cumulative_wells()
        states['Eroded_piece_cells'] = eroded_piece_cells
        states['Aggregate_height'] = self.get_aggregate_height()
        
        return [states["Rows_cleared"], states["Bumpiness"], states["Holes"], states["Landing_height"], states["Row_transitions"], states["Column_transitions"],
                states["Cumulative_wells"], states["Eroded_piece_cells"], states["Aggregate_height"]]
    
    def get_possible_states(self,infos):
        
        # if len(self.buffer.next_list) > 1: #Otherwise might pop empty list
        #     self.last_piece = self.buffer.new_block()
        #     self.piece = self.buffer.next_list[0] #new
        
        # if self.piece is None:
        #     return []
        
        states = []
        #temporary code:
        rows_cleared = self.get_cleared_rows()
        '''
        TODO
        '''
                
        return self.get_state(infos, rows_cleared)
    
    def is_row(self, y):
        """Returns if the row is a fully filled one."""
        return all(self.board[x][y] != 0 for x in range(self.cols))
    
    def remove_row(self, y):
        """Removes a row from the board."""
        removed_row = self.board.pop(y)
        self.board.insert(0, [0] * len(removed_row))
        return removed_row

    def insert_row(self, y, row):
        """Inserts a row into the board."""
        self.board = self.board[:-1]
        self.board.insert(y, row)
    
    def get_cleared_rows(self):
        """Returns the the amount of rows cleared."""
        return list(filter(lambda y: self.is_row(y), range(self.rows)))
    
    def get_bumpiness(self):
        """Returns the total of the difference between the height of each column."""
        bumpiness = 0
        last_height = -1
        for x in range(self.cols):
            current_height = 0
            for y in range(self.rows):
                if self.board[x][y] != 0:
                    current_height = self.rows - y
                    break
            if last_height != -1:
                bumpiness += abs(last_height - current_height)
            last_height = current_height
        return bumpiness

    
    def get_hole_count(self):
        """returns the number of empty cells covered by a full cell."""
        hole_count = 0
        for x in range(self.cols):
            below = False
            for y in range(self.rows):
                empty = self.board[x][y] == 0
                if not below and not empty:
                    below = True
                elif below and empty:
                    hole_count += 1

        return hole_count
    
    def get_row_transitions(self):
        """Returns the number of horizontal cell transitions."""
        total = 0
        for y in range(self.rows):
            row_count = 0
            last_empty = False
            for x in range(self.cols):
                empty = self.board[x][y] == 0
                if last_empty != empty:
                    row_count += 1
                    last_empty = empty

            if last_empty:
                row_count += 1

            if last_empty and row_count == 2:
                continue

            total += row_count
        return total
    
    def get_column_transitions(self):
        """Returns the number of vertical cell transitions."""
        total = 0
        for x in range(self.cols):
            column_count = 0
            last_empty = False
            for y in reversed(range(self.rows)):
                empty = self.board[x][y] == 0
                if last_empty and not empty:
                    column_count += 2
                last_empty = empty

            if last_empty and column_count == 1:
                continue

            total += column_count
        return total
    
    def get_cumulative_wells(self):
        """Returns the sum of all wells."""
        wells = [0 for i in range(self.cols)]
        for y in range(self.rows):
            left_empty = True
            for x in range(self.cols):
                code = self.board[x][y]
                if code == 0:
                    well = False
                    right_empty = self.cols > x + 1 >= 0 and self.board[x + 1][y] == 0
                    if left_empty or right_empty:
                        well = True
                    wells[x] = 0 if well else wells[x] + 1
                    left_empty = True
                else:
                    left_empty = False
        return sum(wells)
    
    def get_aggregate_height(self):
        aggregate_height = 0
        for x in range(self.cols):
            for y in range(self.rows):
                if self.board[x][y] != 0:
                    aggregate_height += self.rows - y
                    break
        return aggregate_height


def TetrisMove(tetris, action, com_event, last_infos):
    com_event.set([action])
    for evt in com_event.get():
        tetris.trigger(evt)

    tetris.increment_timer()
    tetris.move()
    tetris.check_fallen()
    if tetris.is_fallen:
        tetris.clear()
        tetris.new_block()
    tetris.increment_timer()
    infos = {'is_fallen': tetris.is_fallen}

    if tetris.is_fallen:
        height_sum, diff_sum, max_height, holes = get_infos(tetris.get_board())

        # store the different of each information due to the move
        infos['diff_sum'] =  diff_sum - last_infos['diff_sum']
        infos['holes'] =  holes - last_infos['holes'] 
        infos['is_fallen'] =  tetris.is_fallen 
        infos['cleared'] =  tetris.cleared
    return infos

def move(tetris, requireAct, last_infos):
    # print("RA:", requireAct)
    lastX = tetris.px
    com_event = ComEvent()
    for _ in range(requireAct[1]):
        TetrisMove(tetris, 3, com_event, last_infos)
    infos = None
    while True:
        if tetris.px > requireAct[0]:
            TetrisMove(tetris, 6, com_event, last_infos)
        elif tetris.px < requireAct[0]:
            TetrisMove(tetris, 5, com_event, last_infos)
        else:
            coords_buffer.append({"px":tetris.px,"py":tetris.py})
            infos = TetrisMove(tetris, 2, com_event, last_infos)
            break
        # print("pX:", lastX, tetris.px)
        if tetris.px == lastX:
            return False, []
        lastX = tetris.px
    test = TetrisState(tetris, tetris.get_board(),coords_buffer.pop())
    testStates = test.get_possible_states(infos)
    # print("infos:",infos)
    # if len(testStates) != 0: print(testStates) 
    return True, testStates

def get_predict(network, env):
    obs = []
    global NATRUAL_FALL_FREQ
    tmp = NATRUAL_FALL_FREQ
    NATRUAL_FALL_FREQ = 100
    player = env.game_interface.tetris_list[env.game_interface.now_player]
    tetris = player["tetris"]
    ts = TetrisSaver()
    ts.save(tetris)
    for ro in range(0,4):
        for x in range(-2, 9):
            newTetris = deepcopy(tetris)
            # ts.load(newTetris)

            success, simState= move(newTetris, (x, ro), env.game_interface.last_infos)
            if success:
                obs.append([(x, ro), simState])
    obs.sort(key=lambda x:(x[0][1], x[0][0]))
    obs = np.array(obs)
    print(obs)
    action, state = network.act(obs)
    NATRUAL_FALL_FREQ = tmp
    return action

if __name__ == "__main__":
    network = QNetwork(discount=1, epsilon=0, epsilon_min=0, epsilon_decay=0)
    network.load()
    env = TetrisSingleEnv(gridchoice="none", obs_type="grid", mode="human") # env: gym environment for Tetris
    random.seed(50)
    action_meaning_table = env.get_action_meanings() # meaning of action number

    ob = env.reset() # ob: current map depends on obs type(image: return ndarray of pixels of screen; grid: return information (20, 34) grid) (note that originally is (20, 34, 1) )
    
    coords_buffer = []

    for i in range(EPISODE):
        # action_meaning = {
        #     0: "NOOP",
        #     1: "hold",
        #     2: "drop",
        #     3: "rotate_right",
        #     4: "rotate_left",
        #     5: "right",
        #     6: "left",
        #     7: "down" 
        # }
        ### Predict
        # infos: please refer to tetris_interface -> act method
          
        
        ###
        predict = get_predict(network, env)
        player = env.game_interface.tetris_list[env.game_interface.now_player]
        tetris = player["tetris"]
        print("px:",tetris.px,"py:",tetris.py)
        print(predict)
        action = input()
        #print("action:", action_meaning_table[action]) # action chosen
        for i in range(predict[1]):
            ob, reward, done, infos = env.step(3)
        while True:
            if tetris.px > predict[0]:
                ob, reward, done, infos = env.step(6)
            elif tetris.px < predict[0]:
                ob, reward, done, infos = env.step(5)
            else:
                ob, reward, done, infos = env.step(2)
                break

        # ob, reward, done, infos = env.step(action) # execute action we chose in game interface and get now status information and reward
        
        #print states and infos
        # if infos['is_fallen']==0: 
        #     coords_buffer.append({"px":tetris.px,"py":tetris.py})
        # else: 
        #     test = TetrisState(tetris, tetris.get_board(),coords_buffer.pop())
        #     testStates = test.get_possible_states(infos)
        #     print("infos:",infos)
        #     if len(testStates) != 0: print(testStates) 
         
        
        if done:
            ob = env.reset() # if end, info["episode"] will tell accumulated rewards
    