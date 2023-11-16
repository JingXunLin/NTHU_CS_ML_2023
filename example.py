from TetrisBattle.envs.tetris_env import TetrisSingleEnv


if __name__ == "__main__":

    env = TetrisSingleEnv(gridchoice="none", obs_type="grid", mode="rgb_array") # env: gym environment for Tetris

    action_meaning_table = env.get_action_meanings() # meaning of action number

    ob = env.reset() # ob: current map depends on obs type(inage: return ndarray of pixels of screen; grid: return information (20, 34) grid) (note that originally is (20, 34, 1) )
    # information grid: (see grid explain.png)
    # now table: 0.3 ghost block, 0.7 real block
    # hold block: the i-th entry is 1 means hold block type is i
    # next block: for j-th row, the i-th entry is 1 means j-th next block type is i
    # other infos: 
    # 0-th: self.sent / 100
    # 1-th: self.combo / 10
    # 2-th: self.pre_back2back
    # 3-th: self._attacked / GRID_DEPTH

    for i in range(1):

        env.take_turns() # change player in interface

        action = env.random_action() # chose action number (meaning of it is said in line 8 list)
        print("action:", action_meaning_table[action]) # action chosen

        ob, reward, done, infos = env.step(action) # execute action we chose in game interface and get now status information and reward
        # infos: please refer to tetris_interface -> act method

        print("reward:", reward) # reward gain

        if len(infos) != 0: # if end, info["episode"] will tell accumulated rewards
            print(infos)

        if done:
            ob = env.reset() # if end, info["episode"] will tell accumulated rewards