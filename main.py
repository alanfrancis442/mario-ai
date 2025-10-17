from tensordict import TensorDict
import torch
import torch.nn as nn
import numpy as np
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.data import LazyMemmapStorage

import gym
from gym.spaces import Box
import numpy as np
import torch
import torchvision.transforms as T
from gym.wrappers import FrameStack

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
import time

from preprocess import SkipFrame, GrayScaleObservation, ResizeObservation


class MarioNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim
        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = self._build_cnn_layers(c, output_dim)
        self.target = self._build_cnn_layers(c, output_dim)

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

    def _build_cnn_layers(self, input_dim, output_dim):
        """Construct the convolutional layers"""
        self.conv1 = nn.Conv2d(
            in_channels=input_dim, out_channels=32, kernel_size=8, stride=4
        )
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        return nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.ReLU(),
            self.conv3,
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )


class MarioAgent:
    def __init__(self, state_dim, action_dim, save_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.net = MarioNet(self.state_dim, self.action_dim).float()
        self.net.to(self.device)
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.save_every = 5e5

        self.memory = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(
                max_size=50_000, device="cuda" if torch.cuda.is_available() else "cpu"
            ),
        )
        self.batch_size = 32

    def act(self, state):
        """Given a state, choose an epsilon-greedy action"""

        # explore
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.action_dim)
        # exploit
        else:
            state = (
                state[0].__array__() if isinstance(state, tuple) else state.__array__()
            )
            with torch.no_grad():
                state = torch.tensor(state, device=self.device).unsqueeze(0)
                action_values = self.net(state, model="online")
                action_idx = torch.argmax(action_values, dim=1).item()

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate, self.exploration_rate_min)
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """Add the experience to memory"""

        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x

        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        self.memory.add(
            TensorDict(
                {
                    "state": torch.tensor(state),
                    "next_state": torch.tensor(next_state),
                    "action": torch.tensor([action]),
                    "reward": torch.tensor([reward]),
                    "done": torch.tensor([done]),
                },
                batch_size=[],
            )
        )

    def recall(self):
        """Sample experiences from memory"""
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (
            batch.get(key)
            for key in ("state", "next_state", "action", "reward", "done")
        )
        return state, next_state, action, reward, done


class Mario(MarioAgent):
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        self.gamma = 0.99
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.burnin = 1e4
        self.learn_every = 3
        self.sync_every = 1e4

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]
        return current_Q

    @torch.no_grad()
    def td_target(self, next_state, reward, done):
        next_state_Q = self.net(next_state, model="target")
        best_action = torch.argmax(next_state_Q, dim=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save_model(self):
        save_path = (
            self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    def learn(self):
        if self.curr_step % self.learn_every != 0:
            return None, None

        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save_model()

        if self.curr_step < self.burnin:
            return None, None

        state, next_state, action, reward, done = self.recall()

        # get TD estimate
        td_est = self.td_estimate(state, action)
        # get TD target
        td_tgt = self.td_target(next_state, reward, done)
        # backpropagate loss
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)


if __name__ == "__main__":
    CHECKPOINT_PATH = "./checkpoints/2025-10-12T00-21-50/mario_net_0.chkpt"  # 2. Set up the Environment (MUST be identical to your training setup)
    env = gym_super_mario_bros.make(
        "SuperMarioBros-1-1-v0", apply_api_compatibility=True, render_mode="human"
    )
    env = JoypadSpace(env, [["right"], ["right", "A"]])

    # Apply the same wrappers as in training
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)

    # 3. Initialize the Agent
    state_dim = (4, 84, 84)  # The shape of the stacked frames
    action_dim = env.action_space.n
    mario = Mario(state_dim=state_dim, action_dim=action_dim, save_dir=None)

    # 4. Load the saved weights into the agent's network
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        weights_only=False,
    )  # Use 'cpu' if you don't have a GPU
    mario.net.load_state_dict(checkpoint["model"])
    mario.net.to(mario.device)

    # 5. Set the agent to Evaluation Mode
    #    This is crucial! It tells the model to only use its learned knowledge.
    mario.exploration_rate = 0.0  # Turn off exploration completely
    mario.net.eval()
    torch.set_grad_enabled(False)
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
