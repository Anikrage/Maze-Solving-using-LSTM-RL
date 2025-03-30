# Procedural Maze Solver with Memory-Based LSTM RL

This project combines procedural maze generation with a memory-based reinforcement learning (RL) agent that uses an LSTM network to solve the maze. The interactive web application is built with Streamlit and features animated visualizations of the training and maze-solving processes.

## Features

- **Procedural Maze Generation:**  
  Generates a perfect maze using a Depth-First Search (DFS) algorithm. The maze is represented as a 2D grid, with clear entrance and exit points.

- **Custom Maze Environment:**  
  Implements a Gymnasium-compatible environment (`MazeEnv`) that simulates maze navigation. The environment returns the maze grid with the agent’s position marked.

- **Memory-Based RL Agent:**  
  Utilizes an LSTM-based RL agent (using Recurrent PPO from `sb3-contrib`) to solve the maze. The LSTM component allows the agent to handle partial observability and sequential decision-making.

- **Training Visualization:**  
  Displays the training progress by plotting reward per episode over time.

- **Maze Solving Animation:**  
  Provides animated visualizations (using Matplotlib and imageio) to demonstrate the agent’s step-by-step solution path.

- **Interactive Web Interface:**  
  Streamlit is used to provide an interactive interface where users can generate mazes, train the agent, and view animations of the solving process.

## Project Structure

