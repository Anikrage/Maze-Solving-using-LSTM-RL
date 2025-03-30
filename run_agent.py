import gymnasium as gym
from stable_baselines3 import PPO
from maze_env import MazeEnv

def run_agent():
    env = MazeEnv(size=10)
    model = PPO.load("maze_solver_agent")

    state, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(state)
        state, _, done, _, _ = env.step(action)
        env.render()

if __name__ == "__main__":
    run_agent()
