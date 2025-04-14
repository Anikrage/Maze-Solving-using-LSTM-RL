from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

def train_agent(env, episodes=1000, steps_per_episode=200):
    env = DummyVecEnv([lambda: env])
    model = RecurrentPPO("MlpLstmPolicy", env, verbose=0)
    rewards = []
    for episode in range(episodes):
        obs = env.reset()
        total_reward = 0
        for _ in range(steps_per_episode):
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            if done: break
        rewards.append(total_reward)
    model.save("maze_solver_agent.zip")
    return model, rewards
