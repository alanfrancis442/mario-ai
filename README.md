# 🎮 Mario AI - Deep Q-Learning Agent

An AI agent that learns to play Super Mario Bros using Deep Q-Learning (DQN) with PyTorch.

## 📋 Table of Contents
- [Overview](#overview)
- [How It Works](#how-it-works)
- [Mathematical Concepts](#mathematical-concepts)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Training Process](#training-process)
- [Results](#results)

## 🎯 Overview

This project implements a Deep Q-Network (DQN) agent that learns to play Super Mario Bros through reinforcement learning. The agent observes the game screen, learns from its experiences, and improves its gameplay over time without any human intervention.

### Key Features
- **Deep Q-Learning**: Uses neural networks to approximate Q-values
- **Experience Replay**: Stores and samples past experiences for stable learning
- **Target Network**: Separate network for stable Q-value targets
- **Epsilon-Greedy Exploration**: Balances exploration vs exploitation
- **GPU Acceleration**: All computations run on CUDA-enabled GPUs

## 🧠 How It Works

### High-Level Process

1. **Observation**: The agent receives the current game state (4 stacked grayscale frames)
2. **Decision**: Based on the state, the agent chooses an action (move right, jump, etc.)
3. **Action**: The environment executes the action and returns:
   - New state
   - Reward (progress, coins, enemies defeated)
   - Done flag (level complete or Mario dies)
4. **Learning**: The agent stores the experience and learns from past experiences
5. **Repeat**: Continue until Mario completes the level or dies

### State Representation

The game screen is preprocessed into a compact representation:
- **Grayscale**: RGB → Grayscale (reduces complexity)
- **Resize**: 240×256 → 84×84 pixels
- **Frame Stacking**: 4 consecutive frames stacked together
- **Skip Frames**: Only process every 4th frame (speeds up learning)

**Final State Shape**: `(4, 84, 84)` - 4 channels (frames) of 84×84 images

### Action Space

The agent can perform 2 actions:
1. **Move Right**: Continue running right
2. **Move Right + Jump**: Run right while jumping

## 📐 Mathematical Concepts

### 1. Markov Decision Process (MDP)

The game is modeled as an MDP with:
- **States (S)**: Game screen observations
- **Actions (A)**: Available moves (right, jump)
- **Rewards (R)**: Points earned, progress made
- **Transition Probability P(s'|s,a)**: Probability of reaching state s' from state s by taking action a
- **Discount Factor (γ)**: How much future rewards matter (γ = 0.9)

### 2. Q-Learning

The core algorithm learns a **Q-function** Q(s, a) that estimates the expected cumulative reward for taking action `a` in state `s`.

**Bellman Equation:**
```
Q(s, a) = R(s, a) + γ · max(Q(s', a'))
```

Where:
- `R(s, a)` = immediate reward
- `γ` = discount factor (0.9)
- `max(Q(s', a'))` = maximum Q-value in next state

### 3. Deep Q-Network (DQN)

Instead of storing Q-values in a table (impossible for complex states), we use a neural network to approximate Q(s, a).

**Network Architecture:**
```
Input: (4, 84, 84) stacked frames
↓
Conv2D(32 filters, 8×8, stride=4) + ReLU
↓
Conv2D(64 filters, 4×4, stride=2) + ReLU
↓
Conv2D(64 filters, 3×3, stride=1) + ReLU
↓
Flatten → (3136 features)
↓
Linear(3136 → 512) + ReLU
↓
Linear(512 → action_dim)
↓
Output: Q-values for each action
```

### 4. Temporal Difference (TD) Learning

The agent learns by minimizing the **TD error** - the difference between predicted and actual Q-values.

**TD Estimate:**
```
Q_predicted = Q_online(s, a)
```

**TD Target:**
```
Q_target = r + γ · max(Q_target(s', a'))
```

**Loss Function (Smooth L1 Loss):**
```
Loss = SmoothL1(Q_predicted - Q_target)
```

The Smooth L1 Loss is less sensitive to outliers than MSE:
```
SmoothL1(x) = {
    0.5 × x²           if |x| < 1
    |x| - 0.5          otherwise
}
```

### 5. Experience Replay

Stores experiences as tuples: `(state, action, reward, next_state, done)`

**Benefits:**
- Breaks correlation between consecutive samples
- Reuses experiences multiple times (sample efficiency)
- Stabilizes training by learning from diverse experiences

**Memory Size**: 50,000 experiences  
**Batch Size**: 32 samples per learning step

### 6. Target Network

Uses two networks:
- **Online Network**: Updated every learning step
- **Target Network**: Copy of online network, updated every 10,000 steps

**Why?** Prevents the "moving target" problem where Q-values chase each other, causing instability.

### 7. Epsilon-Greedy Exploration

Balances exploration (trying new actions) vs exploitation (using learned knowledge).

```
action = {
    random_action           with probability ε (explore)
    argmax(Q(s, a))        with probability 1-ε (exploit)
}
```

**Epsilon Decay:**
- Start: ε = 1.0 (100% exploration)
- Decay: ε × 0.99999975 per step
- Minimum: ε = 0.1 (10% exploration)

### 8. Reward Discount Factor (γ)

The discount factor determines how much the agent values future rewards.

```
Cumulative Reward = r₀ + γ·r₁ + γ²·r₂ + γ³·r₃ + ...
```

With γ = 0.9:
- Immediate reward: 100% value
- 1 step ahead: 90% value
- 2 steps ahead: 81% value
- 10 steps ahead: 35% value

This encourages the agent to prioritize near-term rewards while still considering future outcomes.

## 🏗️ Project Architecture

### File Structure

```
mario-ai/
├── main.py              # Main training/inference script
├── Net.py               # Neural network architecture (legacy)
├── environment.py       # Environment setup utilities
├── preprocess.py        # Frame preprocessing wrappers
├── requirement.txt      # Python dependencies
├── checkpoints/         # Saved model checkpoints
│   └── YYYY-MM-DDTHH-MM-SS/
│       └── mario_net_X.chkpt
└── main.ipynb          # Jupyter notebook for experimentation
```

### Key Classes

#### `MarioNet` (Neural Network)
- **Online Network**: Used for action selection
- **Target Network**: Used for stable Q-value targets
- CNN architecture for processing game frames

#### `MarioAgent` (Base Agent)
- Handles action selection (epsilon-greedy)
- Manages experience replay buffer
- Stores and recalls experiences

#### `Mario` (Learning Agent)
- Inherits from `MarioAgent`
- Implements TD learning
- Updates networks via backpropagation
- Manages model checkpointing

### Preprocessing Pipeline

```python
env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
env = JoypadSpace(env, [["right"], ["right", "A"]])  # Simplify actions
env = SkipFrame(env, skip=4)                          # Process every 4th frame
env = GrayScaleObservation(env)                       # RGB → Grayscale
env = ResizeObservation(env, shape=84)                # Resize to 84×84
env = FrameStack(env, num_stack=4)                    # Stack 4 frames
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- NVIDIA drivers and CUDA toolkit

### Install Dependencies

```bash
pip install -r requirement.txt
```

**Key Dependencies:**
- `torch` - PyTorch for neural networks
- `torchrl` - Reinforcement learning utilities
- `gym-super-mario-bros` - Mario environment
- `nes-py` - NES emulator
- `tensordict` - Efficient tensor storage

## 💻 Usage

### Training a New Agent

```python
from main import Mario
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from preprocess import SkipFrame, GrayScaleObservation, ResizeObservation
from gym.wrappers import FrameStack
from pathlib import Path

# Setup environment
env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
env = JoypadSpace(env, [["right"], ["right", "A"]])
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)

# Initialize agent
save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

mario = Mario(
    state_dim=(4, 84, 84),
    action_dim=env.action_space.n,
    save_dir=save_dir
)

# Training loop
episodes = 1000
for episode in range(episodes):
    state, info = env.reset()
    
    while True:
        action = mario.act(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        
        mario.cache(state, next_state, action, reward, terminated)
        q_value, loss = mario.learn()
        
        state = next_state
        
        if terminated or truncated:
            break
```

### Running a Trained Agent

```bash
python main.py
```

The script will:
1. Load the checkpoint from `./checkpoints/2025-10-12T00-21-50/mario_net_0.chkpt`
2. Set exploration rate to 0 (pure exploitation)
3. Run Mario with the learned policy
4. Display the gameplay in real-time

**To use a different checkpoint**, modify the `CHECKPOINT_PATH` variable in `main.py`.

## 🎓 Training Process

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `learning_rate` | 0.00025 | Adam optimizer learning rate |
| `gamma` | 0.9 | Reward discount factor |
| `exploration_start` | 1.0 | Initial epsilon (exploration rate) |
| `exploration_min` | 0.1 | Minimum epsilon |
| `exploration_decay` | 0.99999975 | Epsilon decay per step |
| `batch_size` | 32 | Samples per training batch |
| `memory_size` | 50,000 | Replay buffer capacity |
| `burnin` | 10,000 | Steps before learning starts |
| `learn_every` | 3 | Learn every N steps |
| `sync_every` | 10,000 | Sync target network every N steps |
| `save_every` | 500,000 | Save checkpoint every N steps |

### Training Stages

1. **Exploration Phase (Steps 0-10,000)**
   - Agent explores randomly
   - Fills replay buffer with diverse experiences
   - No learning occurs (burnin period)

2. **Early Learning (Steps 10,000-100,000)**
   - High exploration rate (ε ≈ 0.9-0.6)
   - Agent learns basic movements
   - Frequent failures

3. **Skill Development (Steps 100,000-500,000)**
   - Moderate exploration (ε ≈ 0.6-0.3)
   - Agent learns to avoid enemies
   - Begins to complete levels

4. **Mastery (Steps 500,000+)**
   - Low exploration (ε ≈ 0.3-0.1)
   - Consistent level completion
   - Optimizes for speed and score

### Learning Dynamics

**Each step:**
1. Observe state `s`
2. Choose action `a` (epsilon-greedy)
3. Execute action, observe `r`, `s'`, `done`
4. Store experience `(s, a, r, s', done)` in replay buffer
5. Every 3 steps: sample batch, compute loss, update online network
6. Every 10,000 steps: copy online → target network
7. Every 500,000 steps: save checkpoint

## 📊 Results

### Model Checkpoints

The project includes several trained checkpoints:
- `2025-10-12T00-20-29/mario_net_0.chkpt` - Early training
- `2025-10-12T00-21-50/mario_net_0.chkpt` - Intermediate
- `2025-10-14T21-49-33/mario_net_0.chkpt` - Advanced
- `2025-10-16T00-25-28/mario_net_0.chkpt` - Latest

### Expected Performance

- **Untrained**: Random movements, dies quickly
- **After 100K steps**: Can move right consistently, occasional jumps
- **After 500K steps**: Avoids basic obstacles, sometimes completes level
- **After 1M+ steps**: Consistently completes Level 1-1

## 🔧 Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` (default: 32 → 16)
- Reduce `memory_size` (default: 50,000 → 10,000)
- Use CPU instead: modify device detection in `MarioAgent.__init__`

### Slow Training
- Ensure GPU is being used (check "cuda" vs "cpu" print)
- Increase `skip` parameter in `SkipFrame` (4 → 8)
- Reduce `learn_every` frequency

### Agent Not Learning
- Check if `burnin` period has passed (10,000 steps)
- Verify reward signal is non-zero
- Ensure exploration rate is decaying properly

## 📚 References

### Papers
1. **Playing Atari with Deep Reinforcement Learning** (Mnih et al., 2013)
   - Original DQN paper
   - https://arxiv.org/abs/1312.5602

2. **Human-level control through deep reinforcement learning** (Mnih et al., 2015)
   - Improved DQN with experience replay and target networks
   - https://www.nature.com/articles/nature14236

### Concepts
- **Markov Decision Process**: Framework for modeling sequential decision-making
- **Q-Learning**: Off-policy TD control algorithm
- **Bellman Equation**: Recursive relationship for optimal value functions
- **Temporal Difference Learning**: Learn from differences between consecutive predictions
- **Function Approximation**: Use neural networks to represent value functions

## 🤝 Contributing

Feel free to submit issues or pull requests to improve:
- Training stability
- Network architecture
- Hyperparameter tuning
- Documentation

## 📄 License

This project is for educational purposes. Super Mario Bros is owned by Nintendo.

## 🙏 Acknowledgments

- OpenAI Gym team for the RL framework
- PyTorch team for the deep learning library
- nes-py developers for the NES emulator
- DeepMind for pioneering DQN research

---

**Happy Learning! 🎮🤖**
