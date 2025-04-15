import numpy as np
import matplotlib.pyplot as plt
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import os
import time
from lstm_rl_agent import RL_Agent
from maze_generator import MazeGenerator
from environment import MazeEnv
from collections import deque

def create_env_wrapper(env_fn):
    """Create a wrapper to ensure consistent reset behavior with Monitor"""
    def _init():
        env = env_fn()
        env = Monitor(env)
        return env
    return _init

def train_stable_baselines_agent(env, algorithm="recurrent_ppo", total_timesteps=100000,
                           eval_freq=5000, n_eval_episodes=5, vecenv_type="dummy"):
    """Train an agent using Stable Baselines 3"""
    # Create directory for logs and models
    log_dir = f"./logs/{algorithm}_{int(time.time())}"
    model_dir = f"./models/{algorithm}_{int(time.time())}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Create vectorized environment(s)
    env_fn = lambda: Monitor(env)
    if vecenv_type == "subproc":
        vec_env = SubprocVecEnv([create_env_wrapper(lambda: env) for _ in range(4)])
    else:
        vec_env = DummyVecEnv([create_env_wrapper(lambda: env)])

    # Create evaluation environment
    eval_env = DummyVecEnv([create_env_wrapper(lambda: env)])

    # Create callbacks with more reasonable reward threshold
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=0.5, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        callback_after_eval=stop_callback,
        verbose=1
    )

    # Create and train the model with improved exploration
    if algorithm == "recurrent_ppo":
        model = RecurrentPPO(
            "MlpLstmPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Added entropy coefficient for more exploration
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)

    # Save the final model
    final_model_path = os.path.join(model_dir, "final_model")
    model.save(final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")
    
    # Extract rewards for visualization
    rewards = [ep_info['r'] for ep_info in model.ep_info_buffer]
    
    return model, rewards

def train_custom_agent(env, max_episodes=1000, max_steps_per_episode=1000, eval_freq=20,
                   early_stopping_patience=100, save_path="./models/custom_lstm_agent.pt"):
    """Train a custom LSTM agent using our implementation"""
    # Get environment dimensions
    input_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.n

    # Create agent with improved parameters
    agent = RL_Agent(
        input_dim=input_dim,
        action_dim=action_dim,
        lr=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.999,  # Changed from 0.995 to 0.999 for slower decay
        min_epsilon=0.05,     # Changed from 0.01 to 0.05 for more exploration
        hidden_dim=256,       # Increased from 128 to 256
        buffer_size=50000     # Increased buffer size
    )

    # Training metrics
    rewards = []
    mean_rewards = []
    best_reward = -float('inf')
    patience_counter = 0

    # Training loop
    for episode in range(max_episodes):
        # Train for one episode
        episode_reward = agent.train_episode(env, max_steps=max_steps_per_episode)
        rewards.append(episode_reward)

        # Evaluate performance
        if episode % eval_freq == 0 or episode == max_episodes - 1:
            last_rewards = rewards[-min(eval_freq, len(rewards)):]
            mean_reward = np.mean(last_rewards)
            mean_rewards.append(mean_reward)
            print(f"Episode {episode}/{max_episodes}, Mean Reward: {mean_reward:.4f}, Epsilon: {agent.epsilon:.4f}")

            # Check for early stopping
            if mean_reward > best_reward:
                best_reward = mean_reward
                patience_counter = 0
                # Save best model
                agent.save(save_path)
                print(f"New best model saved with mean reward: {best_reward:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {episode+1} episodes")
                    break

    # Create and return training curve plot
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title("Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True, alpha=0.3)
    plt.savefig("./training_curve.png")

    return agent, rewards

def curriculum_training(start_size=5, max_size=15, episodes_per_size=200, reward_type="dense", callback=None):
    """Train agent with curriculum learning on increasingly complex mazes"""
    save_path = f"./models/curriculum_agent.pt"
    best_overall_reward = -float('inf')
    all_rewards = []
    agent = None
    
    # Train on progressively larger mazes
    for size in range(start_size, max_size + 1):
        print(f"\n--- Training on maze size {size} ---")
        
        # Generate a new maze of the current size
        generator = MazeGenerator(size)
        maze = generator.generate_maze(algorithm="dfs")
        
        # Create environment with dense rewards
        env = MazeEnv(maze, reward_type=reward_type)
        
        # Initialize agent for the first maze size or update for new size
        if agent is None:
            input_dim = np.prod(env.observation_space.shape)
            action_dim = env.action_space.n
            
            agent = RL_Agent(
                input_dim=input_dim,
                action_dim=action_dim,
                lr=0.001,
                gamma=0.99,
                epsilon=1.0,
                epsilon_decay=0.999,
                min_epsilon=0.05,
                hidden_dim=256,
                buffer_size=50000
            )
        else:
            # For subsequent maze sizes, create a compatible state representation
            # by preserving the network structure but adapting input dimensions
            new_input_dim = np.prod(env.observation_space.shape)
            
            if new_input_dim != agent.input_dim:
                print(f"Updating input dimension from {agent.input_dim} to {new_input_dim}")
                
                # Create a new agent with the updated input dimension
                # We'll preserve exploration parameters from the previous agent
                old_epsilon = agent.epsilon
                
                agent = RL_Agent(
                    input_dim=new_input_dim,
                    action_dim=action_dim,
                    lr=0.001,
                    gamma=0.99,
                    epsilon=max(0.3, old_epsilon),  # Reset epsilon to ensure exploration on new maze sizes
                    epsilon_decay=0.999,
                    min_epsilon=0.05,
                    hidden_dim=256,
                    buffer_size=50000
                )
        
        # Train for specified number of episodes on this maze size
        size_rewards = []
        
        for episode in range(episodes_per_size):
            # Train for one episode
            episode_reward = agent.train_episode(env)
            size_rewards.append(episode_reward)
            all_rewards.append(episode_reward)
            
            # Print progress every 20 episodes
            if episode % 20 == 0 or episode == episodes_per_size - 1:
                mean_reward = np.mean(size_rewards[-min(20, len(size_rewards)):])
                print(f"Size {size}, Episode {episode}/{episodes_per_size}, Mean Reward: {mean_reward:.4f}, Epsilon: {agent.epsilon:.4f}")
                
                # Call callback if provided
                if callback:
                    callback(size, episode, mean_reward, agent.epsilon)
                
                # Save best model
                if mean_reward > best_overall_reward:
                    best_overall_reward = mean_reward
                    agent.save(save_path)
                    print(f"New best overall model saved with mean reward: {best_overall_reward:.4f}")
                    
        # Only proceed to the next maze size if performance on this size is good enough
        final_mean_reward = np.mean(size_rewards[-20:])
        if final_mean_reward < 0 and size < max_size:
            print(f"Performance not satisfactory on size {size} (reward: {final_mean_reward:.4f})")
            print(f"Will train for another {episodes_per_size} episodes at this size")
            
            # Train for additional episodes on this size
            for episode in range(episodes_per_size):
                episode_reward = agent.train_episode(env)
                size_rewards.append(episode_reward)
                all_rewards.append(episode_reward)
                
                if episode % 20 == 0 or episode == episodes_per_size - 1:
                    mean_reward = np.mean(size_rewards[-min(20, len(size_rewards)):])
                    print(f"Size {size} (Extended), Episode {episode}/{episodes_per_size}, Mean Reward: {mean_reward:.4f}, Epsilon: {agent.epsilon:.4f}")
                    
                    # Call callback if provided
                    if callback:
                        callback(size, episode + episodes_per_size, mean_reward, agent.epsilon)
                    
                    if mean_reward > best_overall_reward:
                        best_overall_reward = mean_reward
                        agent.save(save_path)
                        print(f"New best overall model saved with mean reward: {best_overall_reward:.4f}")
    
    print(f"\nCurriculum training complete! Final model saved to {save_path}")
    
    # Create and return training curve plot
    plt.figure(figsize=(10, 6))
    plt.plot(all_rewards)
    plt.title("Curriculum Training Rewards")
    plt.xlabel("Total Episodes")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.savefig("./curriculum_training_curve.png")
    
    return agent, all_rewards

def train_agent(env, algorithm="recurrent_ppo", total_timesteps=100000, curriculum=False, 
                start_size=5, max_size=15, reward_type="dense"):
    """Train an agent using the specified algorithm"""
    if curriculum:
        return curriculum_training(
            start_size=start_size, 
            max_size=max_size, 
            episodes_per_size=int(total_timesteps/(max_size-start_size+1)/200),
            reward_type=reward_type
        )
    elif algorithm == "recurrent_ppo" or algorithm == "sb3":
        return train_stable_baselines_agent(env, algorithm="recurrent_ppo", total_timesteps=total_timesteps)
    elif algorithm == "custom_lstm":
        return train_custom_agent(env, max_episodes=int(total_timesteps/200))
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
