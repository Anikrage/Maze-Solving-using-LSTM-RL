import numpy as np
import argparse
from maze_generator import MazeGenerator
from environment import MazeEnv
from train_agent import train_agent, curriculum_training

def main():
    parser = argparse.ArgumentParser(description="Train a maze solver with improved settings")
    parser.add_argument("--maze_size", type=int, default=10, help="Size of the maze")
    parser.add_argument("--algorithm", type=str, choices=["recurrent_ppo", "custom_lstm"], 
                        default="custom_lstm", help="Algorithm to use")
    parser.add_argument("--reward_type", type=str, choices=["sparse", "dense"], 
                        default="dense", help="Reward structure")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--curriculum", action="store_true", help="Use curriculum learning")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    if args.curriculum:
        print("Starting curriculum learning...")
        agent, rewards = curriculum_training(
            start_size=5,
            max_size=args.maze_size,
            episodes_per_size=200,
            reward_type=args.reward_type
        )
    else:
        # Generate maze
        print(f"Generating {args.maze_size}x{args.maze_size} maze...")
        generator = MazeGenerator(args.maze_size, seed=args.seed)
        maze = generator.generate_maze(algorithm="dfs")
        
        # Create environment
        env = MazeEnv(maze, reward_type=args.reward_type)
        
        # Train agent
        print(f"Training {args.algorithm} agent for {args.episodes} episodes...")
        agent, rewards = train_agent(
            env, 
            algorithm=args.algorithm,
            total_timesteps=args.episodes * 200  # Estimate 200 steps per episode
        )
    
    print("Training complete!")

if __name__ == "__main__":
    main()
