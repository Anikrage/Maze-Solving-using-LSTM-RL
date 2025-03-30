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


## Usage

    Run the Streamlit App:

    streamlit run app.py

    Interact with the App:

        Generate Maze:
        Use the slider to adjust the maze size and click "Generate Maze" to create a new maze.

        Train LSTM Agent:
        Click "Train LSTM Agent" to train the RL agent on the generated maze. A plot of the training rewards per episode is displayed.

        Solve Maze:
        Once trained, click "Solve Maze" to generate a solution path (using a placeholder BFS method) and visualize it on the maze.

        Animate Solve Maze:
        Click "Animate Solve Maze" to see an animated step-by-step visualization of the maze being solved.

## Project Components

    Maze Generation:
    Implemented in maze_generator.py using a recursive DFS approach. The generator creates a perfect maze with one unique path between any two points.

    Maze Environment:
    Defined in environment.py, this Gymnasium-compatible environment sets up the maze, defines the action space (up, right, down, left), and provides methods for resetting and stepping through the maze.

    RL Agent Training:
    The training routine in train_agent.py leverages Recurrent PPO from sb3-contrib to train an LSTM-based policy. The training loop collects rewards and updates the agent's policy.

    Visualization:
    The application visualizes the maze, training progress, and solution path. Animation is created by generating frames with Matplotlib and compiling them into a GIF with imageio.

## Future Work

    Agent Improvement:
    Explore more advanced memory-based architectures and fine-tune the LSTM network for better performance.

    Enhanced Maze Complexity:
    Introduce varying levels of maze complexity and additional features (e.g., dynamic obstacles).

    Real-Time Training Visualization:
    Implement real-time visual updates during agent training for better insight into the learning process.

Happy Maze Solving!