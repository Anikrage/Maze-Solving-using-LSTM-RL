import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import imageio.v2 as imageio

from maze_generator import MazeGenerator
from train_agent import train_agent
from environment import MazeEnv

st.title("Procedural Maze Solver with LSTM RL")

# Slider for maze size (maze dimensions will be 2*size+1)
size = st.slider("Maze Size", 5, 20, 10, step=1)

# Initialize session state variables
if "maze" not in st.session_state:
    st.session_state.maze = None
    st.session_state.env = None
    st.session_state.agent_trained = False
    st.session_state.model = None
    st.session_state.solve_path = None

# Button: Generate Maze
if st.button("Generate Maze"):
    generator = MazeGenerator(size)
    st.session_state.maze = generator.generate_dfs_maze()
    st.session_state.env = MazeEnv(st.session_state.maze)

    # Visualize the generated maze
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(st.session_state.maze, cmap="gray")
    ax.scatter(st.session_state.env.start[1], st.session_state.env.start[0],
               c="blue", marker="o", label="Start")
    ax.scatter(st.session_state.env.goal[1], st.session_state.env.goal[0],
               c="red", marker="x", label="Goal")
    ax.legend()
    st.pyplot(fig)
    
    # Reset training and solution if a new maze is generated
    st.session_state.agent_trained = False
    st.session_state.model = None
    st.session_state.solve_path = None

# Button: Train LSTM Agent (ensure maze exists)
if st.session_state.maze is not None:
    if st.button("Train LSTM Agent"):
        with st.spinner("Training..."):
            model, training_progress = train_agent(st.session_state.env)
            st.session_state.model = model
            st.session_state.agent_trained = True
        st.success("Training Complete!")
        
        # Plot training progress (reward per episode)
        fig, ax = plt.subplots()
        ax.plot(training_progress, label="Reward per Episode")
        ax.set_title("Training Progress")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.legend()
        st.pyplot(fig)

# Button: Solve Maze (ensure agent is trained)
if st.session_state.agent_trained:
    if st.button("Solve Maze"):
        # Get a solution path (this uses a placeholder random policy)
        path = st.session_state.env.solve_with_trained_agent()
        st.session_state.solve_path = path

        # Visualize the static solution path on the maze
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(st.session_state.maze, cmap="gray")
        path_array = np.array(path)
        ax.plot(path_array[:, 1], path_array[:, 0], 'r-', linewidth=2)
        ax.scatter(path_array[0, 1], path_array[0, 0], c="green", marker="o", s=100, label="Start")
        ax.scatter(path_array[-1, 1], path_array[-1, 0], c="blue", marker="x", s=100, label="Goal")
        ax.legend()
        st.pyplot(fig)

# Button: Animate Maze Solving (ensure a solution exists)
if st.session_state.solve_path is not None:
    if st.button("Animate Solve Maze"):
        path = st.session_state.solve_path
        frames = []
        # Create an animation frame for each step of the solution path.
        for i in range(1, len(path) + 1):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(st.session_state.maze, cmap="gray")
            current_path = np.array(path[:i])
            ax.plot(current_path[:, 1], current_path[:, 0], 'r-', linewidth=2)
            ax.scatter(current_path[0, 1], current_path[0, 0],
                       c="green", marker="o", s=100, label="Start")
            # Mark the current agent position with an "x"
            ax.scatter(current_path[-1, 1], current_path[-1, 0],
                       c="blue", marker="x", s=100, label="Current")
            ax.legend()
            
            # Save the current frame to a buffer
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            image = imageio.imread(buf)
            frames.append(image)
            plt.close(fig)
        
        # Save the frames as an animated GIF in memory.
        gif_buf = io.BytesIO()
        imageio.mimsave(gif_buf, frames, format="GIF", duration=0.3)
        st.image(gif_buf.getvalue(), caption="Maze Solve Animation")
