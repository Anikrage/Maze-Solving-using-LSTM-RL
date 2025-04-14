import torch
from stable_baselines3 import PPO
from environment import MazeEnv

def run_agent():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = MazeEnv(maze)  # Create your maze appropriately
    model = PPO.load("maze_solver_agent.zip", device=device)

    state, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(state)
        state, _, done, _, _ = env.step(action)
        env.render()

if __name__ == "__main__":
    run_agent()
