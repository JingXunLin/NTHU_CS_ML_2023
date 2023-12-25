from TetrisBattle.envs.tetris_env import TetrisDoubleEnv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from TetrisBattle.getState import *

class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()

        self.conv1 = nn.Sequential(nn.Linear(4, 64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Linear(64, 1))

        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x
def do_action(action, env, tetris):
    #ret_done = False
    tetris.block.current_shape_id = action[1]
    tetris.px = action[0]
    # _, reward, done, infos = env.step(0)
    #print(infos)
    #assert(tetris.px == action[0])
    #while infos['is_fallen'] == 0:
    _, reward, done, infos = env.step(2)
    return _, reward, done, infos

if __name__ == "__main__":

    env = TetrisDoubleEnv(gridchoice="none", obs_type="grid", mode="human") # env: gym environment for Tetris
    model1 = DeepQNetwork()
    model1.load_state_dict(torch.load('tetris'))
    tetris1 = env.game_interface.tetris_list[0]["tetris"]
    model2 = DeepQNetwork()
    model2.load_state_dict(torch.load('tetris'))
    tetris2 = env.game_interface.tetris_list[1]["tetris"]
    env.reset()
    while True:
        if env.game_interface.now_player == 0:
            tetris = tetris1
        else:
            tetris = tetris2
        next_steps = get_next_states(tetris)
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.tensor(next_states,dtype=torch.float)
        if env.game_interface.now_player == 0:
            predictions = model1(next_states)[:, 0]
        else:
            predictions = model2(next_states)[:, 0]
        
        index = torch.argmax(predictions).item()
        action = next_actions[index]
        _, reward, done, infos  = do_action(action, env, tetris)

        if done:
            break
        env.take_turns()
