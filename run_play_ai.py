import gym
import pygame
import gym_tetris
from TetrisBattle.envs.tetris_env import TetrisSingleEnv
from gym_tetris.ai.QNetwork import QNetwork


def main():
    env = gym.make("tetris-v1", action_mode=1)
    # env2 = TetrisSingleEnv(gridchoice="none", obs_type="grid", mode="human")
    # network = QNetwork(discount=1, epsilon=0, epsilon_min=0, epsilon_decay=0)
    # network.load()

    # obs = env.reset()
    # obs2 = env2.reset()
    print(env.observation_space)
    return
    running = True
    display = True
    while running:
        action, state = network.act(obs)
        obs, reward, done, info = env.step(action)

        if display:
            env.render()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    display = not display
        if done:
            obs = env.reset()

    env.close()


if __name__ == '__main__':
    main()
