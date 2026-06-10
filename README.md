# Dino Run with Rainbow DQN

Chrome Dino reinforcement learning project comparing Deep Q-Network variants up to Rainbow DQN.

The project implements a Chrome Dino Gym environment workflow and several DQN agents, then compares how different algorithmic improvements affect training performance.

## What Is Included

- Chrome Dino Gym environment wrapper under `Dino_run/gym_chrome_dino/`
- PyTorch DQN agents in `Dino_run/torch_agents.py`
- Model definitions in `Dino_run/torch_model.py`
- Rainbow DQN training entry point in `Dino_run/torch_main.py`
- Training logs, plots, and saved model artifacts under `Dino_run/log/` and `Dino_run/models/`
- Project report files: `Dino run.pdf`, `Dino run.docx`, and `107302002.pptx`

## Implemented Agents

- DQN
- Double DQN
- Dueling DQN
- CER DQN
- Noisy DQN
- Prioritized Experience Replay DQN
- Rainbow DQN

## Installation

```bash
git clone https://github.com/novis10813/Dino-Run-with-Rainbow-DQN.git
cd Dino-Run-with-Rainbow-DQN
```

This repository was originally developed with Python 3.7-era RL and browser automation dependencies. There is no current `requirements.txt`, so dependency setup may need adjustment for your local Python, PyTorch, Selenium, Chrome, and ChromeDriver versions.

## Usage

The main PyTorch entry point is:

```bash
cd Dino_run
python torch_main.py
```

By default, `torch_main.py` creates the Chrome Dino environment and trains `RainbowDQN`. To run a different agent, edit the commented agent blocks in `Dino_run/torch_main.py`.

The environment uses Chrome/ChromeDriver for browser-based gameplay. The repository includes Windows ChromeDriver artifacts, but you may need to replace them with a driver that matches your operating system and Chrome version.

## Results

Training logs and plots are available under `Dino_run/log/`, including per-agent CSV files and comparison images.

## References

1. Mnih et al. (2015), "Human-level control through deep reinforcement learning".
2. Van Hasselt, Guez, and Silver (2016), "Deep Reinforcement Learning with Double Q-learning".
3. Wang et al. (2016), "Dueling Network Architectures for Deep Reinforcement Learning".
4. Fortunato et al. (2017), "Noisy Networks for Exploration".
5. Schaul et al. (2015), "Prioritized Experience Replay".
6. Hessel et al. (2017), "Rainbow: Combining Improvements in Deep Reinforcement Learning".

## Status

This is an educational reinforcement learning project and experiment archive. It is useful for reading agent implementations and comparing DQN variants, but may require dependency updates before it runs on a modern machine.
