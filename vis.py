import numpy as np
import matplotlib.pyplot as plt
from maze_generator import MazeGenerator
from environment import MazeEnv
from lstm_rl_agent import RL_Agent
import imageio.v2 as imageio
import io

# 1. Generate a maze
size = 10
generator = MazeGenerator(size)
maze = generator.generate_maze(algorithm="dfs")
env = MazeEnv(maze, reward_type="dense")

# 2. Initialize agent with the same architecture as during training
input_dim = np.prod(env.observation_space.shape)
action_dim = env.action_space.n
agent = RL_Agent(
    input_dim=input_dim,
    action_dim=action_dim,
    hidden_dim=256  # Match the hidden dimension used during training
)

# 3. Load the trained model
agent.load("./models/custom_lstm_agent.pt", eval_mode=True)
print("Model loaded successfully")

# 4. Solve the maze
state, _ = env.reset()
done = truncated = False
path = [env.state]
total_reward = 0
hidden = None

while not (done or truncated):
    # Get action from the model (minimal exploration in evaluation)
    action, hidden = agent.q_network.get_action(state, hidden, epsilon=0.01)
    
    # Execute action
    next_state, reward, done, truncated, _ = env.step(action)
    
    # Update state and accumulate reward
    state = next_state
    total_reward += reward
    path.append(env.state)

print(f"Maze solution found! Steps: {len(path)}, Total reward: {total_reward:.2f}")

# 5. Visualize the solution
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(env.maze, cmap="binary")
path_array = np.array(path)
ax.plot(path_array[:, 1], path_array[:, 0], 'r-', linewidth=2)
ax.scatter(path_array[0, 1], path_array[0, 0], c="green", marker="o", s=100, label="Start")
ax.scatter(path_array[-1, 1], path_array[-1, 0], c="blue", marker="x", s=100, label="End")
ax.set_title(f"Agent Solution (Reward: {total_reward:.2f})")
ax.legend()
plt.savefig("agent_solution.png")
plt.show()
