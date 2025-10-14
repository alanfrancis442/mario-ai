import torch
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import time

from environment import SkipFrame, GrayScaleObservation, ResizeObservation
from main import Mario
from gym.wrappers import FrameStack
# 1. Path to your saved checkpoint
#    Update this to the actual path of your saved model.
CHECKPOINT_PATH = "./checkpoints/2025-10-12T00-21-50/mario_net_0.chkpt"

# 2. Set up the Environment (MUST be identical to your training setup)
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', apply_api_compatibility=True, render_mode='human')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Apply the same wrappers as in training
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)

# 3. Initialize the Agent
state_dim = (4, 84, 84) # The shape of the stacked frames
action_dim = env.action_space.n
mario = Mario(state_dim=state_dim, action_dim=action_dim, save_dir=None)

# 4. Load the saved weights into the agent's network
checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu')) # Use 'cpu' if you don't have a GPU
mario.net.load_state_dict(checkpoint['model'])

# 5. Set the agent to Evaluation Mode
#    This is crucial! It tells the model to only use its learned knowledge.
mario.exploration_rate = 0.0 # Turn off exploration completely
mario.net.eval()

print("Loaded model! Starting gameplay...")

# 6. The "Play" Loop
state, info = env.reset()

while True:
    # The environment renders the screen for you to watch
    env.render()
    
    # Let the agent choose the best action based on the current state
    action = mario.act(state)
    
    # Perform the action in the game
    next_state, reward, terminated, truncated, info = env.step(action)
    
    # Update the state for the next loop iteration
    state = next_state
    
    # A small delay to make the gameplay watchable
    time.sleep(0.02)
    
    # If Mario dies or completes the level, the episode is done
    if terminated or truncated:
        print("Episode finished. Resetting...")
        state, info = env.reset()