from TetrisBattle.envs.tetris_env import TetrisSingleEnv
import random


if __name__ == "__main__":

    env = TetrisSingleEnv(gridchoice="none", obs_type="grid", mode="human") # env: gym environment for Tetris

    ob = env.reset() # ob: current map depends on obs type(inage: return ndarray of pixels of screen; grid: return information (20, 34) grid) (note that originally is (20, 34, 1) )
    for i in range(100):

         # change player in interface

        action = env.random_action() # chose action number (meaning of it is said in line 8 list)
        
        ob, reward, done, infos = env.step(action)
        player = env.game_interface.now_player
        print(env.game_interface.tetris_list[player]["tetris"].px, env.game_interface.tetris_list[player]["tetris"].py)
        


        if done:
            ob = env.reset() # if end, info["episode"] will tell accumulated rewards
            break