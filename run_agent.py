import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from environment import MazeEnv
from maze_generator import MazeGenerator
import os
import argparse
import time
import imageio.v2 as imageio
import io

def run_sb3_agent(env, model_path, render=False, num_episodes=1, deterministic=True, save_gif=False):
    """Run a trained Stable Baselines 3 agent"""
    try:
        # Try to load as RecurrentPPO first
        model = RecurrentPPO.load(model_path)
        print("Loaded RecurrentPPO model")
    except:
        try:
            # Fall back to PPO if RecurrentPPO fails
            model = PPO.load(model_path)
            print("Loaded PPO model")
        except Exception as e:
            print(f"Failed to load model: {e}")
            return None

    # Check if maze shape matches model's expected observation shape
    expected_shape = model.observation_space.shape
    actual_shape = env.observation_space.shape
    
    if expected_shape != actual_shape:
        expected_size = (expected_shape[0] - 1) // 2
        actual_size = (actual_shape[0] - 1) // 2
        print(f"Warning: Model expects maze size {expected_size} but current maze is size {actual_size}")
        print(f"Consider regenerating maze with correct size or using 'custom' agent type")

    all_paths = []
    all_rewards = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = truncated = False
        ep_reward = 0
        path = [env.state]

        while not (done or truncated):
            # Handle potential shape mismatch
            if obs.shape != expected_shape:
                print(f"Skipping episode due to shape mismatch: {obs.shape} vs {expected_shape}")
                break
                
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            path.append(env.state)
            ep_reward += reward

            if render:
                env.render()

        all_paths.append(path)
        all_rewards.append(ep_reward)
        print(f"Episode {ep+1}/{num_episodes}, Reward: {ep_reward:.2f}, Path length: {len(path)}")

    # Visualize the best path
    if all_paths and all_rewards:
        best_ep = np.argmax(all_rewards)
        best_path = all_paths[best_ep]
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(env.maze, cmap="binary")
        path_array = np.array(best_path)
        ax.plot(path_array[:, 1], path_array[:, 0], 'r-', linewidth=2)
        ax.scatter(path_array[0, 1], path_array[0, 0], c="green", marker="o", s=100, label="Start")
        ax.scatter(path_array[-1, 1], path_array[-1, 0], c="blue", marker="x", s=100, label="End")
        ax.set_title(f"Agent Solution (Reward: {all_rewards[best_ep]:.2f})")
        ax.legend()
        plt.tight_layout()
        plt.savefig("agent_solution.png")
        plt.close()

        # Create animated GIF if requested
        if save_gif:
            frames = []
            for i in range(1, len(best_path) + 1):
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(env.maze, cmap="binary")
                current_path = np.array(best_path[:i])
                ax.plot(current_path[:, 1], current_path[:, 0], 'r-', linewidth=2)
                ax.scatter(current_path[0, 1], current_path[0, 0], c="green", marker="o", s=100, label="Start")
                ax.scatter(current_path[-1, 1], current_path[-1, 0], c="blue", marker="x", s=100, label="Current")
                if i == len(best_path) and best_path[-1] == env.goal:
                    ax.text(0.5, 0.01, "Goal Reached!", transform=ax.transAxes,
                          ha="center", va="bottom", fontsize=14, color="green")
                ax.set_title(f"Step {i}/{len(best_path)}")
                ax.legend()
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                frames.append(imageio.imread(buf))
                plt.close(fig)

            # Save as GIF
            imageio.mimsave("agent_solution.gif", frames, duration=0.2)
            print("Created animated solution GIF: agent_solution.gif")

    return all_paths, all_rewards

def run_custom_agent(env, model_path, render=False, num_episodes=1, deterministic=True, save_gif=False):
    """Run a trained custom LSTM agent"""
    from lstm_rl_agent import RL_Agent

    # Get environment dimensions
    input_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.n

    # Create agent
    agent = RL_Agent(input_dim=input_dim, action_dim=action_dim)

    try:
        # Use weights_only=False for PyTorch 2.6 compatibility
        agent.load(model_path, eval_mode=True, weights_only=False)
        print("Loaded custom LSTM agent")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

    all_paths = []
    all_rewards = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        done = truncated = False
        ep_reward = 0
        path = [env.state]
        hidden = None

        while not (done or truncated):
            action, hidden = agent.q_network.get_action(state, hidden, epsilon=0.01)
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state
            path.append(env.state)
            ep_reward += reward

            if render:
                env.render()

        all_paths.append(path)
        all_rewards.append(ep_reward)
        print(f"Episode {ep+1}/{num_episodes}, Reward: {ep_reward:.2f}, Path length: {len(path)}")

    # Visualization (same as for SB3 agent)
    best_ep = np.argmax(all_rewards)
    best_path = all_paths[best_ep]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(env.maze, cmap="binary")
    path_array = np.array(best_path)
    ax.plot(path_array[:, 1], path_array[:, 0], 'r-', linewidth=2)
    ax.scatter(path_array[0, 1], path_array[0, 0], c="green", marker="o", s=100, label="Start")
    ax.scatter(path_array[-1, 1], path_array[-1, 0], c="blue", marker="x", s=100, label="End")
    ax.set_title(f"Custom Agent Solution (Reward: {all_rewards[best_ep]:.2f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig("custom_agent_solution.png")
    plt.close()

    # Create animated GIF if requested (same as for SB3 agent)
    if save_gif:
        frames = []
        for i in range(1, len(best_path) + 1):
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(env.maze, cmap="binary")
            current_path = np.array(best_path[:i])
            ax.plot(current_path[:, 1], current_path[:, 0], 'r-', linewidth=2)
            ax.scatter(current_path[0, 1], current_path[0, 0], c="green", marker="o", s=100, label="Start")
            ax.scatter(current_path[-1, 1], current_path[-1, 0], c="blue", marker="x", s=100, label="Current")
            if i == len(best_path) and best_path[-1] == env.goal:
                ax.text(0.5, 0.01, "Goal Reached!", transform=ax.transAxes,
                      ha="center", va="bottom", fontsize=14, color="green")
            ax.set_title(f"Step {i}/{len(best_path)}")
            ax.legend()
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            frames.append(imageio.imread(buf))
            plt.close(fig)

        # Save as GIF
        imageio.mimsave("custom_agent_solution.gif", frames, duration=0.2)
        print("Created animated solution GIF: custom_agent_solution.gif")

    return all_paths, all_rewards

def compare_with_optimal(env, agent_path, agent_type="sb3"):
    """Compare agent solution with optimal BFS solution"""
    # Get optimal path
    optimal_path = env.solve_with_bfs()
    
    if optimal_path is None:
        print("No optimal path found for this maze!")
        return None, None

    # Get agent path based on type
    if agent_type == "sb3":
        agent_paths, agent_rewards = run_sb3_agent(env, agent_path, render=False, num_episodes=1)
    else:
        agent_paths, agent_rewards = run_custom_agent(env, agent_path, render=False, num_episodes=1)
        
    if not agent_paths:
        print("Agent failed to find a path!")
        return optimal_path, None
        
    agent_path = agent_paths[0]

    # Calculate metrics
    optimal_length = len(optimal_path) - 1  # Subtract 1 for start position
    agent_length = len(agent_path) - 1  # Subtract 1 for start position
    
    if agent_length == 0:
        print("Agent path has zero length - likely an error occurred")
        return optimal_path, agent_path
        
    print(f"Optimal path length: {optimal_length}")
    print(f"Agent path length: {agent_length}")
    print(f"Efficiency: {optimal_length / agent_length:.2f} (1.0 is optimal)")

    # Visualize both paths
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(env.maze, cmap="binary")

    # Plot optimal path
    optimal_array = np.array(optimal_path)
    ax.plot(optimal_array[:, 1], optimal_array[:, 0], 'g-', linewidth=2, label="Optimal (BFS)")

    # Plot agent path
    agent_array = np.array(agent_path)
    ax.plot(agent_array[:, 1], agent_array[:, 0], 'r--', linewidth=2, label="Agent")

    ax.scatter(env.start[1], env.start[0], c="green", marker="o", s=100, label="Start")
    ax.scatter(env.goal[1], env.goal[0], c="blue", marker="x", s=100, label="Goal")
    ax.set_title("Path Comparison: Optimal vs Agent")
    ax.legend()
    plt.tight_layout()
    plt.savefig("path_comparison.png")
    plt.close()

    return optimal_path, agent_path

def main():
    parser = argparse.ArgumentParser(description="Run a trained maze solver agent")
    parser.add_argument("--maze_size", type=int, default=10, help="Size of the maze to generate")
    parser.add_argument("--model_path", type=str, default="models/maze_solver_agent", help="Path to the trained model")
    parser.add_argument("--algorithm", type=str, choices=["dfs", "prims", "recursive_division"],
                       default="dfs", help="Maze generation algorithm")
    parser.add_argument("--agent_type", type=str, choices=["sb3", "custom"],
                       default="sb3", help="Type of agent to run")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--save_gif", action="store_true", help="Save an animated GIF of the solution")
    parser.add_argument("--compare", action="store_true", help="Compare with optimal solution")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    # Generate maze
    generator = MazeGenerator(args.maze_size, seed=args.seed)
    maze = generator.generate_maze(algorithm=args.algorithm)

    # Create environment
    env = MazeEnv(maze, reward_type="dense")  # Use dense rewards for better evaluation

    # Run agent
    if args.agent_type == "sb3":
        paths, rewards = run_sb3_agent(env, args.model_path, render=args.render,
                                     num_episodes=args.episodes, save_gif=args.save_gif)
    else:
        paths, rewards = run_custom_agent(env, args.model_path, render=args.render,
                                       num_episodes=args.episodes, save_gif=args.save_gif)

    # Compare with optimal if requested
    if args.compare and paths is not None:
        compare_with_optimal(env, args.model_path, agent_type=args.agent_type)

if __name__ == "__main__":
    main()
