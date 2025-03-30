from sb3_contrib import RecurrentPPO  # Import from sb3_contrib
from stable_baselines3.common.vec_env import DummyVecEnv

def train_agent(env):
    env = DummyVecEnv([lambda: env])
    
    model = RecurrentPPO("MlpLstmPolicy", env, verbose=0)  # Use MlpLstmPolicy for LSTM

    rewards = []
    for episode in range(1000):
        obs = env.reset()
        total_reward = 0
        for _ in range(200):
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        rewards.append(total_reward)
    
    return model, rewards
