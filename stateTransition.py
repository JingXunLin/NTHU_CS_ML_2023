from TetrisBattle.envs.tetris_env import TetrisSingleEnv
from TetrisBattle.tetris import Tetris, get_infos, Piece, Buffer, hardDrop, rotateCollide, collideDown, collide, rotate
from TetrisBattle.envs.tetris_interface import TetrisSingleInterface, TetrisDoubleInterface
from copy import deepcopy


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
#from tqdm.notebook import tqdm

EPISODE = 100


class TetrisState():
    
    
    def __init__(self, tetris, board, coords): #Call method: Tetris(Player(info_dict), gridchoice, tetris.get_board())

        self.tetris = tetris
        self.board = board
        self.cols = len(self.board)
        self.rows = len(self.board[0])
        self.buffer = tetris.buffer
        self.last_piece = None
        self.piece = None #new
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
        
        
        if infos['is_fallen']:
            landing_height = self.rows - self.coords['py']
            #eroded_piece_cells = infos['cleared'] * sum(y in rows_cleared for x, y in self.last_piece.return_pos(self.coords['px'], self.coords['py']))   
            
        states['Rows_cleared'] = infos['cleared']
        states['Bumpiness'] = infos['diff_sum']
        states['Holes'] = infos['holes']
        states['Landing_height'] = landing_height
        states['Row_transitions'] = self.get_row_transitions()
        states['Column_transitions'] = self.get_column_transitions()
        states['Cumulative_wells'] = self.get_cumulative_wells()
        states['Eroded_piece_cells'] = eroded_piece_cells
        states['Aggregate_height'] = self.get_aggregate_height()
        
        return states
    
    def get_possible_states(self,infos):
        
        if len(self.buffer.next_list) > 1: #Otherwise might pop empty list
            self.last_piece = self.buffer.new_block()
            self.piece = self.buffer.next_list[0] #new
        
        if self.piece is None:
            return []
        
        states = []
        #temporary code:
        rows_cleared = self.get_cleared_rows()
        states.append(((0,0), self.get_state(infos, rows_cleared)))
        '''
        TODO
        '''
                
        return states
    
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
    
if __name__ == "__main__":

    env = TetrisSingleEnv(gridchoice="none", obs_type="grid", mode="human") # env: gym environment for Tetris

    action_meaning_table = env.get_action_meanings() # meaning of action number

    ob = env.reset() # ob: current map depends on obs type(image: return ndarray of pixels of screen; grid: return information (20, 34) grid) (note that originally is (20, 34, 1) )
    
    coords_buffer = []

    for i in range(EPISODE):

        #env.take_turns() # change player in interface

        action = env.random_action() # chose action number (meaning of it is said in line 8 list)
        #print("action:", action_meaning_table[action]) # action chosen

        ob, reward, done, infos = env.step(action) # execute action we chose in game interface and get now status information and reward
        # infos: please refer to tetris_interface -> act method
        
        player = env.game_interface.tetris_list[env.game_interface.now_player]
        tetris = player["tetris"]
        
        print("px:",tetris.px,"py:",tetris.py)
        
        #print states and infos
        if infos['is_fallen']==0: 
            coords_buffer.append({"px":tetris.px,"py":tetris.py})
        else: 
            test = TetrisState(tetris, tetris.get_board(),coords_buffer.pop())
            testStates = test.get_possible_states(infos)
            print("infos:",infos)
            if len(testStates) != 0: print(testStates)    
        
        
        if done:
            ob = env.reset() # if end, info["episode"] will tell accumulated rewards
    