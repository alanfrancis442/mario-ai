# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack
from preprocess import SkipFrame, GrayScaleObservation, ResizeObservation

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros

env = gym_super_mario_bros.make(
    "SuperMarioBros-1-1-v0", render_mode="human", apply_api_compatibility=True
)


# Limit the action-space to
#   0. walk right
#   1. jump right
# env = JoypadSpace(env, [["right"], ["right", "A"]])


env.reset()
next_state, reward, done, trunc, info = env.step(action=0)
print(f"{next_state.shape},\n {reward},\n {done},\n {info}")
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=(84, 84))
env = FrameStack(env, num_stack=4)

done = True
for step in range(5000):
    if done:
        state = env.reset()
        env.step(env.action_space.sample())
    env.render()

env.close()
