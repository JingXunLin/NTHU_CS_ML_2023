from TetrisBattle.envs.tetris_env import TetrisSingleEnv


if __name__ == "__main__":

    env = TetrisSingleEnv(gridchoice="none", obs_type="grid", mode="rgb_array") # env: gym environment for Tetris

    action_meaning_table = env.get_action_meanings() # meaning of action number

    ob = env.reset() # ob: game interface
    # we can get all status information we need in ob & infos

    for i in range(20):

        env.take_turns() # take_turn() only work in double mode

        action = env.random_action() # chose action number (meaning of it is said in line 8 list)
        print("action:", action_meaning_table[action]) # action chosen

        ob, reward, done, infos = env.step(action) # execute action we chose in game interface and get now status information and reward

        print("reward:", reward) # reward gain

        if len(infos) != 0: # if end, info["episode"] will tell accumulated rewards
            print(infos)

        if done:
            ob = env.reset() # if end, info["episode"] will tell accumulated rewards