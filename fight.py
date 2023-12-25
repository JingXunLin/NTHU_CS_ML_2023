from TetrisBattle.envs.tetris_env import TetrisDoubleEnv,TetrisSingleEnv
from Group37_model import *

if __name__ == "__main__":
    #env = TetrisSingleEnv(gridchoice="none", obs_type="grid", mode="human")
    env = TetrisDoubleEnv(gridchoice="none", obs_type="grid", mode="human")
    model1 = DeepQNetwork()
    tetris1 = env.game_interface.tetris_list[0]["tetris"]
    model2 = DeepQNetwork()
    tetris2 = env.game_interface.tetris_list[1]["tetris"]
    env.reset()
    while True:
        tetris = env.game_interface.tetris_list[env.game_interface.now_player]["tetris"]

        next_states = get_next_states(tetris)
        next_actions, next_states = zip(*next_states.items())
        next_states = torch.tensor(next_states,dtype=torch.float)
        if env.game_interface.now_player == 0:
            predictions = model1(next_states)[:, 0]
        else:
            predictions = model2(next_states)[:, 0]
        
        index = torch.argmax(predictions).item()
        action = next_actions[index]
        _, reward, done, infos  = do_action(action, env, tetris)
        print(action)
        if done:
            break
        env.take_turns()
