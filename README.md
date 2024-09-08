# Dino Run with Rainbow DQN and Variants

## 🦖 Overview

This project implements the classic Chrome Dino Run game using various Deep Q-Network (DQN) variants, culminating in the Rainbow DQN algorithm. It showcases the power of reinforcement learning in mastering simple yet challenging games.

## 🌈 Features

- 🎮 Implementation of Chrome's Dino Run game
- 🧠 Multiple DQN variants:
  - Double DQN
  - Dueling DQN
  - DQN with Cyclic Exploration Rate (CER)
  - Noisy DQN
  - Prioritized Experience Replay (PER) DQN
  - Rainbow DQN (combining all above improvements)
- 📊 Performance visualization and analysis
- 🔢 Comparative study of different DQN variants

## 🛠️ Installation

Ensure you have Python 3.7+ installed. Then follow these steps:

```bash
# Clone the repository
git clone https://github.com/312707047/Dino-Run-with-Rainbow-DQN
cd Dino-Run-with-Rainbow-DQN

# Install the required packages
pip install -r requirements.txt
```

## 🖥️ Usage

To run the Dino Run game with a specific DQN variant, please modify `Dino_run/torch_main.py`

## 🧠 Models

### Double DQN
Reduces overestimation bias in Q-value estimation.

### Dueling DQN
Separates state-value and advantage functions for more efficient learning.

### CER DQN 
Implements a CER DQN for taking every current observation to update the agent.

### Noisy DQN
Adds parametric noise to the weights for better exploration.

### PER DQN (Prioritized Experience Replay)
Prioritizes important transitions in the replay buffer for more efficient learning.

### Rainbow DQN
Combines all the above improvements for state-of-the-art performance.

## 📚 References

1. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.
2. Van Hasselt, H., Guez, A., & Silver, D. (2016). Deep Reinforcement Learning with Double Q-learning. AAAI.
3. Wang, Z., et al. (2016). Dueling Network Architectures for Deep Reinforcement Learning. ICML.
4. Fortunato, M., et al. (2017). Noisy Networks for Exploration. arXiv preprint arXiv:1706.10295.
5. Schaul, T., et al. (2015). Prioritized Experience Replay. arXiv preprint arXiv:1511.05952.
6. Hessel, M., et al. (2017). Rainbow: Combining Improvements in Deep Reinforcement Learning. arXiv preprint arXiv:1710.02298.
