import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import imageio.v2 as imageio
import time
from datetime import datetime
from collections import deque
from maze_generator import MazeGenerator
from train_agent import train_agent, curriculum_training
from environment import MazeEnv
from sb3_contrib import RecurrentPPO

# Set page configuration
st.set_page_config(
    page_title="Procedural Maze Solver with LSTM RL",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("Procedural Maze Solver with LSTM RL")
st.markdown("""
This application generates procedural mazes and uses Reinforcement Learning with LSTM networks to solve them.
Build a maze, train an agent, and watch it navigate through the labyrinth!
""")

# Sidebar for configuration options
with st.sidebar:
    st.header("Configuration")
    
    # Maze configuration
    st.subheader("Maze Settings")
    size = st.slider("Maze Size", 5, 30, 10, step=1, help="Size of the maze (will be 2*size+1)")
    maze_algorithm = st.selectbox(
        "Generation Algorithm",
        ["dfs", "prims", "recursive_division"],
        index=0,
        format_func=lambda x: {"dfs": "DFS", "prims": "Prim's", "recursive_division": "Recursive Division"}[x],
        help="Algorithm used to generate the maze"
    )
    seed = st.number_input("Random Seed", min_value=0, max_value=9999, value=42,
                          help="Seed for reproducible maze generation")
    
    # Training configuration
    st.subheader("Training Settings")
    training_episodes = st.slider("Training Episodes", 500, 5000, 1000, step=100,
                               help="Number of episodes to train the agent")
    algorithm = st.selectbox(
        "RL Algorithm",
        ["recurrent_ppo", "custom_lstm"],
        index=0,
        format_func=lambda x: {"recurrent_ppo": "RecurrentPPO", "custom_lstm": "Custom LSTM"}[x],
        help="Reinforcement learning algorithm to use"
    )
    reward_type = st.selectbox(
        "Reward Structure",
        ["default", "distance"],
        index=0,
        format_func=lambda x: {"default": "Default", "distance": "Distance-based"}[x],
        help="Type of reward function to use during training"
    )
    
    # Add curriculum learning option
    curriculum = st.checkbox("Use Curriculum Learning", value=True, 
                            help="Start with simple mazes and gradually increase complexity")
    if curriculum:
        curr_col1, curr_col2 = st.columns(2)
        with curr_col1:
            start_size = st.slider("Starting Maze Size", 3, 10, 5, 
                                  help="Size of the smallest maze to start training")
        with curr_col2:
            max_size = st.slider("Maximum Maze Size", 5, 20, min(15, size), 
                                help="Size of the largest maze to train on")
        
        episodes_per_size = st.slider("Episodes per Size", 100, 1000, 200, step=50,
                                     help="Number of training episodes for each maze size")
    
    # Model management
    st.subheader("Model Management")
    model_name = st.text_input("Model Name", "maze_solver_model",
                              help="Name for saving/loading the trained model")
    models_dir = "./models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Get list of all model files (.zip for SB3, .pt for custom LSTM)
    saved_models = [f for f in os.listdir(models_dir) if f.endswith(".zip") or f.endswith(".pt")]
    if saved_models:
        load_col1, load_col2 = st.columns(2)
        with load_col1:
            model_type = st.selectbox(
                "Model Type",
                ["RecurrentPPO", "Custom LSTM"],
                format_func=lambda x: x,
                help="Type of model to load"
            )
        
        with load_col2:
            # Filter models based on type
            filtered_models = [f for f in saved_models if 
                             (model_type == "RecurrentPPO" and f.endswith(".zip")) or
                             (model_type == "Custom LSTM" and f.endswith(".pt"))]
            
            if filtered_models:
                load_model = st.selectbox("Select Model", ["None"] + filtered_models,
                                        help="Load a previously trained model")
            else:
                st.warning(f"No {model_type} models found in {models_dir}")
                load_model = "None"

# Initialize session state variables
if "maze" not in st.session_state:
    st.session_state.maze = None
    st.session_state.env = None
    st.session_state.agent_trained = False
    st.session_state.model = None
    st.session_state.solve_path = None
    st.session_state.bfs_path = None
    st.session_state.agent_path = None
    st.session_state.training_progress = None
    st.session_state.model_maze_size_mismatch = False

# Main area with tabs
tab1, tab2, tab3, tab4 = st.tabs(["Generate Maze", "Train Agent", "Solve Maze", "Analysis"])

with tab1:
    st.header("Maze Generation")
    
    if st.button("Generate New Maze", key="generate_maze"):
        with st.spinner("Generating maze..."):
            # Create maze generator with selected algorithm
            generator = MazeGenerator(size, seed=seed)
            st.session_state.maze = generator.generate_maze(algorithm=maze_algorithm)
            
            # Create environment
            st.session_state.env = MazeEnv(st.session_state.maze, reward_type="dense" if reward_type == "distance" else "sparse")
            
            # Reset training and solution if a new maze is generated
            st.session_state.agent_trained = False
            st.session_state.model = None
            st.session_state.solve_path = None
            st.session_state.bfs_path = None
            st.session_state.agent_path = None
            st.session_state.model_maze_size_mismatch = False
            
            st.success("Maze generated successfully!")
    
    # Visualize the generated maze
    if st.session_state.maze is not None:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(st.session_state.maze, cmap="binary")
        ax.scatter(st.session_state.env.start[1], st.session_state.env.start[0],
                 c="green", marker="o", s=100, label="Start")
        ax.scatter(st.session_state.env.goal[1], st.session_state.env.goal[0],
                 c="red", marker="x", s=100, label="Goal")
        ax.set_title(f"Generated Maze ({maze_algorithm} algorithm, size {size})")
        ax.legend()
        
        # Display the figure in Streamlit
        st.pyplot(fig)
        
        # Calculate and store optimal path
        st.session_state.bfs_path = st.session_state.env.solve_with_bfs()
        if st.session_state.bfs_path:
            st.write(f"BFS solution length: {len(st.session_state.bfs_path)} steps")
        else:
            st.warning("This maze has no solution!")

with tab2:
    st.header("Train LSTM Agent")
    
    # Check if maze exists
    if st.session_state.maze is None:
        st.warning("Please generate a maze first in the 'Generate Maze' tab.")
    else:
        # Model loading section
        if 'load_model' in locals() and load_model != "None":
            if st.button("Load Selected Model"):
                with st.spinner(f"Loading model {load_model}..."):
                    model_path = os.path.join(models_dir, load_model)
                    try:
                        if model_type == "RecurrentPPO":
                            # Check if maze size matches the expected input dimensions
                            model = RecurrentPPO.load(model_path)
                            expected_shape = model.observation_space.shape
                            actual_shape = st.session_state.env.observation_space.shape
                            
                            if expected_shape != actual_shape:
                                # Extract the expected maze size
                                expected_size = (expected_shape[0] - 1) // 2
                                actual_size = (actual_shape[0] - 1) // 2
                                st.warning(f"Model expects maze size {expected_size} but current maze is size {actual_size}. "
                                         f"Consider generating a new maze with size {expected_size}.")
                                st.session_state.model_maze_size_mismatch = True
                            else:
                                st.session_state.model_maze_size_mismatch = False
                            
                            st.session_state.model = model
                        else:  # Custom LSTM
                            from lstm_rl_agent import RL_Agent
                            # Initialize with correct dimensions
                            input_dim = np.prod(st.session_state.env.observation_space.shape)
                            action_dim = st.session_state.env.action_space.n
                            agent = RL_Agent(input_dim=input_dim, action_dim=action_dim)
                            # Add weights_only=False for PyTorch 2.6 compatibility
                            agent.load(model_path, eval_mode=True, weights_only=False)
                            st.session_state.model = agent
                            st.session_state.model_maze_size_mismatch = False
                            
                        st.session_state.agent_trained = True
                        st.success(f"Model loaded successfully from {model_path}")
                    except Exception as e:
                        st.error(f"Error loading model: {e}")
        
        # Train button
        if not st.session_state.agent_trained:
            train_button = st.button("Train Agent", key="train_agent")
            if train_button:
                progress_bar = st.progress(0)
                status_text = st.empty()
                reward_plot = st.empty()
                
                # Training parameters
                if curriculum:
                    with st.spinner("Starting curriculum learning from small mazes..."):
                        # Setup the callback for progress updates
                        class StreamlitCallback:
                            def __init__(self, total_sizes):
                                self.episode = 0
                                self.size_index = 0
                                self.total_sizes = total_sizes
                                self.rewards = []
                                
                            def __call__(self, size, episode, mean_reward, epsilon):
                                self.episode += 1
                                total_episodes = episodes_per_size * self.total_sizes
                                current_episode = (size - start_size) * episodes_per_size + episode
                                progress = min(1.0, current_episode / total_episodes)
                                
                                # Update progress bar
                                progress_bar.progress(progress)
                                
                                # Update status text
                                status_text.text(f"Training on maze size {size}: Episode {episode}/{episodes_per_size}, "
                                               f"Reward: {mean_reward:.2f}, Epsilon: {epsilon:.4f}")
                                
                                # Store reward for plotting
                                self.rewards.append(mean_reward)
                                
                                # Update plot every 10 episodes
                                if len(self.rewards) % 10 == 0:
                                    fig, ax = plt.subplots(figsize=(10, 4))
                                    ax.plot(self.rewards)
                                    ax.set_title("Curriculum Training Progress")
                                    ax.set_xlabel("Training Steps")
                                    ax.set_ylabel("Mean Reward")
                                    ax.grid(True, alpha=0.3)
                                    reward_plot.pyplot(fig)
                                    plt.close(fig)
                        
                        # Initialize callback
                        callback = StreamlitCallback(max_size - start_size + 1)
                        
                        # Start curriculum training
                        agent, rewards = curriculum_training(
                            start_size=start_size,
                            max_size=max_size,
                            episodes_per_size=episodes_per_size,
                            reward_type="dense" if reward_type == "distance" else "sparse",
                            callback=callback
                        )
                        
                        # Save the curriculum-trained model
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        model_filename = f"{model_name}_curriculum_{timestamp}.pt"
                        model_path = os.path.join(models_dir, model_filename)
                        
                        try:
                            agent.save(model_path)
                            st.success(f"Curriculum training complete! Model saved as {model_filename}")
                        except Exception as e:
                            st.error(f"Error saving model: {e}")
                            st.success("Training complete but model could not be saved.")
                        
                        st.session_state.model = agent
                        st.session_state.agent_trained = True
                        st.session_state.training_progress = rewards
                else:
                    # Original training for single maze
                    total_timesteps = training_episodes * 200  # Estimate steps per episode
                    
                    with st.spinner("Training agent on current maze..."):
                        # Train the agent
                        model, training_progress = train_agent(
                            st.session_state.env,
                            algorithm=algorithm,
                            total_timesteps=total_timesteps
                        )
                        
                        st.session_state.model = model
                        st.session_state.agent_trained = True
                        
                        # Convert RecurrentPPO progress if needed
                        if algorithm == "recurrent_ppo" and isinstance(training_progress, deque) and training_progress:
                            if isinstance(training_progress[0], dict):
                                st.session_state.training_progress = [ep_info['r'] for ep_info in training_progress]
                            else:
                                st.session_state.training_progress = training_progress
                        else:
                            st.session_state.training_progress = training_progress
                        
                        # Save the trained model
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        if algorithm == "recurrent_ppo":
                            model_filename = f"{model_name}_{timestamp}.zip"
                        else:
                            model_filename = f"{model_name}_{timestamp}.pt"
                        
                        model_path = os.path.join(models_dir, model_filename)
                        try:
                            model.save(model_path)
                            st.success(f"Training complete! Model saved as {model_filename}")
                        except Exception as e:
                            st.error(f"Error saving model: {e}")
                            st.success("Training complete but model could not be saved.")
        
        # Plot training progress
        if hasattr(st.session_state, 'training_progress') and st.session_state.training_progress:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Check if training_progress contains dictionaries (RecurrentPPO)
            if isinstance(st.session_state.training_progress, deque) and len(st.session_state.training_progress) > 0 and isinstance(st.session_state.training_progress[0], dict):
                # Extract only the reward values from each episode info dictionary
                rewards = [ep_info['r'] for ep_info in st.session_state.training_progress]
                ax.plot(rewards, label="Reward per Episode")
            else:
                # For custom_lstm, training_progress is already a list of rewards
                ax.plot(st.session_state.training_progress, label="Reward per Episode")
            
            ax.set_title("Training Progress")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Reward")
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)
        
        elif st.session_state.agent_trained:
            st.success("Agent already trained and ready to solve mazes!")

with tab3:
    st.header("Solve Maze")
    
    # Check for maze size mismatch warning
    if hasattr(st.session_state, 'model_maze_size_mismatch') and st.session_state.model_maze_size_mismatch:
        st.error("Cannot solve with current maze size. Please generate a new maze with the correct size.")
    
    # Check if maze and trained agent exist
    elif st.session_state.maze is None:
        st.warning("Please generate a maze first in the 'Generate Maze' tab.")
    elif not st.session_state.agent_trained:
        st.warning("Please train an agent first in the 'Train Agent' tab.")
    else:
        solve_col1, solve_col2 = st.columns(2)
        
        with solve_col1:
            if st.button("Solve with Agent", key="solve_agent"):
                with st.spinner("Solving maze with trained agent..."):
                    try:
                        # Check agent type and solve appropriately
                        if isinstance(st.session_state.model, RecurrentPPO):
                            # Handle potential observation shape mismatch
                            expected_shape = st.session_state.model.observation_space.shape
                            actual_shape = st.session_state.env.observation_space.shape
                            if expected_shape != actual_shape:
                                st.error(f"Model expects maze size {(expected_shape[0]-1)//2} but current maze is size {(actual_shape[0]-1)//2}")
                                st.info("Please generate a new maze with the correct size")
                            else:
                                path = st.session_state.env.solve_with_trained_agent(st.session_state.model)
                                st.session_state.agent_path = path
                        else:
                            # Custom LSTM agent
                            state, _ = st.session_state.env.reset()
                            done = truncated = False
                            path = [st.session_state.env.state]
                            hidden = None
                            steps = 0
                            max_steps = st.session_state.env.max_steps
                            
                            while not (done or truncated) and steps < max_steps:
                                # Get action with minimal exploration
                                action, hidden = st.session_state.model.q_network.get_action(state, hidden, epsilon=0.01)
                                next_state, reward, done, truncated, _ = st.session_state.env.step(action)
                                state = next_state
                                path.append(st.session_state.env.state)
                                steps += 1
                            
                            st.session_state.agent_path = path
                        
                        # Show success or failure message
                        if 'agent_path' in st.session_state and st.session_state.agent_path:
                            if st.session_state.agent_path[-1] == st.session_state.env.goal:
                                st.success(f"Maze solved in {len(st.session_state.agent_path)} steps!")
                            else:
                                st.error("Agent failed to solve the maze.")
                    except Exception as e:
                        st.error(f"Error during maze solving: {e}")
        
        with solve_col2:
            if st.button("Solve with BFS (Optimal)", key="solve_bfs"):
                with st.spinner("Finding optimal solution with BFS..."):
                    path = st.session_state.env.solve_with_bfs()
                    st.session_state.bfs_path = path
                    
                    if path:
                        st.success(f"Optimal solution found: {len(path)} steps")
                    else:
                        st.error("Could not find a solution for this maze.")
        
        # Display solutions if available
        if st.session_state.agent_path or st.session_state.bfs_path:
            st.subheader("Visualization")
            
            # Create visualization with both paths if available
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(st.session_state.maze, cmap="binary")
            
            # Plot optimal BFS path
            if st.session_state.bfs_path:
                bfs_path_array = np.array(st.session_state.bfs_path)
                ax.plot(bfs_path_array[:, 1], bfs_path_array[:, 0], 'g-', linewidth=2, alpha=0.7, label="Optimal (BFS)")
            
            # Plot agent path
            if st.session_state.agent_path:
                agent_path_array = np.array(st.session_state.agent_path)
                ax.plot(agent_path_array[:, 1], agent_path_array[:, 0], 'r--', linewidth=2, label="Agent")
            
            # Mark start and goal
            ax.scatter(st.session_state.env.start[1], st.session_state.env.start[0],
                     c="green", marker="o", s=100, label="Start")
            ax.scatter(st.session_state.env.goal[1], st.session_state.env.goal[0],
                     c="red", marker="x", s=100, label="Goal")
            
            ax.set_title("Solution Paths")
            ax.legend()
            
            st.pyplot(fig)
            
            # Animation button
            if st.session_state.agent_path:
                if st.button("Animate Agent Solution", key="animate_solution"):
                    path = st.session_state.agent_path
                    frames = []
                    
                    with st.spinner("Creating animation..."):
                        # Create an animation frame for each step
                        for i in range(1, len(path) + 1):
                            fig, ax = plt.subplots(figsize=(8, 8))
                            ax.imshow(st.session_state.maze, cmap="binary")
                            
                            # Plot full BFS path for reference
                            if st.session_state.bfs_path:
                                bfs_path_array = np.array(st.session_state.bfs_path)
                                ax.plot(bfs_path_array[:, 1], bfs_path_array[:, 0],
                                     'g-', linewidth=1, alpha=0.3, label="Optimal")
                            
                            # Plot agent path so far
                            current_path = np.array(path[:i])
                            ax.plot(current_path[:, 1], current_path[:, 0], 'r-', linewidth=2)
                            
                            # Mark start and current position
                            ax.scatter(path[0][1], path[0][0],
                                     c="green", marker="o", s=100, label="Start")
                            ax.scatter(path[i-1][1], path[i-1][0],
                                     c="blue", marker="x", s=100, label="Current")
                            ax.scatter(st.session_state.env.goal[1], st.session_state.env.goal[0],
                                     c="red", marker="s", s=80, label="Goal")
                            
                            # Add progress text
                            ax.set_title(f"Step {i}/{len(path)}")
                            if i == len(path):
                                if path[-1] == st.session_state.env.goal:
                                    ax.text(0.5, 0.01, "Goal Reached!", transform=ax.transAxes,
                                         ha="center", va="bottom", fontsize=14, color="green")
                            
                            ax.legend()
                            
                            # Save frame
                            buf = io.BytesIO()
                            plt.savefig(buf, format="png")
                            buf.seek(0)
                            image = imageio.imread(buf)
                            frames.append(image)
                            plt.close(fig)
                        
                        # Create and save GIF
                        gif_path = f"agent_solution_{int(time.time())}.gif"
                        imageio.mimsave(gif_path, frames, duration=0.2)
                        
                        # Display the GIF
                        st.image(gif_path, caption="Agent Navigation Animation")
                        
                        # Provide download button
                        with open(gif_path, "rb") as file:
                            btn = st.download_button(
                                label="Download Animation",
                                data=file,
                                file_name=gif_path,
                                mime="image/gif"
                            )

with tab4:
    st.header("Performance Analysis")
    
    if st.session_state.maze is None:
        st.warning("Please generate a maze first in the 'Generate Maze' tab.")
    elif not st.session_state.agent_trained:
        st.warning("Please train an agent first in the 'Train Agent' tab.")
    elif not hasattr(st.session_state, 'agent_path') or st.session_state.agent_path is None:
        st.warning("Please solve the maze with the agent in the 'Solve Maze' tab.")
    else:
        # Calculate metrics
        optimal_path_length = len(st.session_state.bfs_path) - 1 if st.session_state.bfs_path else float('inf')
        agent_path_length = len(st.session_state.agent_path) - 1 if st.session_state.agent_path else float('inf')
        
        # Display metrics
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            st.metric("Optimal Solution Length", optimal_path_length)
        
        with metrics_col2:
            st.metric("Agent Solution Length", agent_path_length)
        
        with metrics_col3:
            if optimal_path_length > 0 and agent_path_length > 0 and optimal_path_length != float('inf'):
                efficiency = optimal_path_length / agent_path_length
                st.metric("Efficiency", f"{efficiency:.2f}", f"{(efficiency-1)*100:.1f}%")
        
        # Add more detailed analysis
        st.subheader("Detailed Analysis")
        
        # Create a heatmap of state visitations
        if st.session_state.agent_path:
            visitation_grid = np.zeros_like(st.session_state.maze)
            for pos in st.session_state.agent_path:
                visitation_grid[pos] += 1
            
            fig, ax = plt.subplots(figsize=(8, 8))
            heatmap = ax.imshow(visitation_grid, cmap="hot", alpha=0.7)
            ax.set_title("Agent State Visitation Heatmap")
            plt.colorbar(heatmap, ax=ax, label="Visit Count")
            st.pyplot(fig)
        
        # Show path comparison statistics
        if st.session_state.bfs_path and st.session_state.agent_path:
            # Calculate wasted moves (revisiting cells)
            unique_visits = len(set(tuple(pos) for pos in st.session_state.agent_path))
            total_visits = len(st.session_state.agent_path)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Unique cells visited:** {unique_visits}")
                st.write(f"**Total steps taken:** {total_visits}")
            
            with col2:
                st.write(f"**Wasted moves:** {total_visits - unique_visits}")
                st.write(f"**Revisit percentage:** {(total_visits - unique_visits)/total_visits*100:.1f}%")
            
            # Check if agent solution matches optimal
            if st.session_state.agent_path == st.session_state.bfs_path:
                st.success("Agent found the optimal solution! 🎉")
            else:
                st.info("Agent solution differs from the optimal path.")
            
            # Decision point analysis
            decision_points = []
            for i in range(1, len(st.session_state.agent_path)-1):
                prev_pos = st.session_state.agent_path[i-1]
                curr_pos = st.session_state.agent_path[i]
                next_pos = st.session_state.agent_path[i+1]
                
                # Check if direction changed
                prev_dir = (curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])
                next_dir = (next_pos[0] - curr_pos[0], next_pos[1] - curr_pos[1])
                
                if prev_dir != next_dir:
                    decision_points.append(curr_pos)
            
            st.write(f"**Decision points (direction changes):** {len(decision_points)}")
            
            # Optional: Plot decision points on the maze
            if st.checkbox("Show Decision Points"):
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(st.session_state.maze, cmap="binary")
                
                # Plot agent path
                agent_path_array = np.array(st.session_state.agent_path)
                ax.plot(agent_path_array[:, 1], agent_path_array[:, 0], 'r-', linewidth=2, alpha=0.7)
                
                # Plot decision points
                decision_points_array = np.array(decision_points)
                if len(decision_points) > 0:  # Check if there are any decision points
                    ax.scatter(decision_points_array[:, 1], decision_points_array[:, 0],
                             c="blue", marker="o", s=80, label="Decision Points")
                
                ax.set_title("Agent Path with Decision Points")
                ax.legend()
                st.pyplot(fig)

# Add a footer with info about the project
st.markdown("---")
st.markdown("""
This project demonstrates procedural maze generation and reinforcement learning using LSTM networks.
The agent learns to navigate through mazes of varying complexity using deep reinforcement learning.
Current Date: Tuesday, April 15, 2025, 9:46 PM IST
""")
