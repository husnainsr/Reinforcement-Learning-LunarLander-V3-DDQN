# 🚀 Lunar Lander with Double DQN

A comprehensive Reinforcement Learning project implementing Double Deep Q-Network (Double DQN) to solve the Lunar Lander environment from OpenAI Gym.

## 🎯 Project Overview

This project demonstrates the application of value-based reinforcement learning methods to solve the continuous control problem of landing a spacecraft. The agent learns to control the lander's thrusters to achieve smooth landings while minimizing fuel consumption.

### Key Features
- **Double DQN Implementation**: Reduces overestimation bias in Q-learning
- **Experience Replay**: Improves sample efficiency and stability
- **Target Network**: Stabilizes training by using separate target Q-network
- **Comprehensive Visualization**: Real-time training plots and landing animations
- **Model Persistence**: Save and load trained models
- **Performance Metrics**: Detailed analysis of learning progress

## 🛠️ Technical Implementation

### Value-Based Method: Double DQN
- **Algorithm**: Double Deep Q-Network (van Hasselt et al., 2016)
- **Network Architecture**: Deep Neural Network with experience replay
- **Exploration Strategy**: Epsilon-greedy with decay
- **Optimization**: Adam optimizer with learning rate scheduling

### Environment Details
- **State Space**: 8-dimensional continuous space (position, velocity, angle, etc.)
- **Action Space**: 4 discrete actions (do nothing, fire left, fire main, fire right)
- **Reward Structure**: +100 for successful landing, penalties for crashes and fuel usage

## 📁 Project Structure

```
lunar_lander_dqn/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── config/
│   └── config.yaml          # Hyperparameters and settings
├── src/
│   ├── agent/               # DQN and Double DQN implementations
│   ├── models/              # Neural network architectures
│   ├── utils/               # Utilities (replay buffer, plotting, etc.)
│   └── training/            # Training loop and logic
├── main.py                  # Main training script
├── evaluate.py              # Model evaluation script
├── results/                 # Training results and plots
├── saved_models/            # Trained model weights
└── notebooks/               # Jupyter notebooks for analysis
```

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Training
```bash
python main.py
```

### 3. Evaluation
```bash
python evaluate.py --model saved_models/best_model.pth
```

## 📊 Expected Results

The agent should achieve:
- **Episode Score**: 200+ (successful landing)
- **Training Time**: ~1000-2000 episodes
- **Success Rate**: >90% after training
- **Landing Precision**: Smooth, controlled landings

## 🔬 Research Background

This implementation is based on:
- **DQN**: Mnih et al. (2015) - Human-level control through deep reinforcement learning
- **Double DQN**: van Hasselt et al. (2016) - Deep reinforcement learning with double Q-learning
- **Experience Replay**: Lin (1992) - Self-improving reactive agents

## 📈 Performance Visualization

The project includes comprehensive visualization:
- Real-time training progress
- Episode rewards over time
- Loss curves
- Landing success rate
- Q-value distributions

## 🤝 Contributing

This project is designed for educational purposes and research. Feel free to experiment with:
- Different network architectures
- Hyperparameter tuning
- Additional RL algorithms
- Environment modifications

## 📚 References

1. van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double Q-learning.
2. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning.
3. Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction.

## 🏆 Project Outcomes

This project demonstrates:
- **Theoretical Understanding**: Implementation of state-of-the-art RL algorithms
- **Practical Application**: Real-world control problem solving
- **Research Skills**: Experimental design and result analysis
- **Software Engineering**: Professional code structure and documentation 