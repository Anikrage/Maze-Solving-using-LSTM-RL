import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

def train_agent(env, episodes=1000, steps_per_episode=200):
    # Determine the device: use CUDA if available, otherwise CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st_device = f"Using device: {device}"
    print(st_device)  # or log it in your application

    # Wrap the environment for Stable-Baselines3
    env = DummyVecEnv([lambda: env])
    # Pass the device to the model; RecurrentPPO accepts a 'device' parameter
    model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, device=device)
    
    # Total timesteps for training: episodes * steps_per_episode
    total_timesteps = episodes * steps_per_episode
    model.learn(total_timesteps=total_timesteps)
    
    # Evaluate the trained model to log rewards per episode
    rewards = []
    for ep in range(episodes):
        obs = env.reset()
        total_reward = 0
        for _ in range(steps_per_episode):
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        rewards.append(total_reward)
        
    model.save("maze_solver_agent.zip")
    return model, rewards
