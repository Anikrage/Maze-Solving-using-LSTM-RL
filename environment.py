import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

class MazeEnv(gym.Env):
    def __init__(self, maze):
        super(MazeEnv, self).__init__()
        self.maze = maze
        self.start = (1, 0)  # Entrance
        self.goal = (maze.shape[0] - 2, maze.shape[1] - 1)  # Exit
        self.state = self.start

        # Observation: full maze grid with agent position marked as 2
        self.observation_space = spaces.Box(low=0, high=2, shape=maze.shape, dtype=np.uint8)
        self.action_space = spaces.Discrete(4)  # 0: Up, 1: Right, 2: Down, 3: Left

    def _get_obs(self):
        """Return a copy of the maze with the agent's position marked."""
        obs = self.maze.copy().astype(np.uint8)
        row, col = self.state
        obs[row, col] = 2
        return obs

    def reset(self, seed=None, options=None):
        """Reset environment state and return observation."""
        super().reset(seed=seed)
        self.state = self.start
        return self._get_obs(), {}

    def step(self, action):
        """Execute one step in the environment."""
        row, col = self.state
        new_row, new_col = row, col

        if action == 0:  # Up
            new_row -= 1
        elif action == 1:  # Right
            new_col += 1
        elif action == 2:  # Down
            new_row += 1
        elif action == 3:  # Left
            new_col -= 1

        # Check boundaries and wall collision
        if 0 <= new_row < self.maze.shape[0] and 0 <= new_col < self.maze.shape[1]:
            if self.maze[new_row, new_col] == 0:
                self.state = (new_row, new_col)

        done = self.state == self.goal
        reward = 1 if done else -0.1
        truncated = False
        return self._get_obs(), reward, done, truncated, {}

    def render(self):
        """Print the maze with the agent's current position (for debugging)."""
        obs = self._get_obs()
        print(obs)

    def solve_with_trained_agent(self):
        """
        Use BFS to find a solution path from start to goal.
        This is a placeholder for the trained agent's policy.
        Returns a list of coordinates representing the path.
        """
        start = self.start
        goal = self.goal
        queue = deque([start])
        came_from = {start: None}

        while queue:
            current = queue.popleft()
            if current == goal:
                break

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                # Ensure neighbor is within bounds and is a path (0)
                if (0 <= neighbor[0] < self.maze.shape[0] and
                    0 <= neighbor[1] < self.maze.shape[1] and
                    self.maze[neighbor] == 0 and
                    neighbor not in came_from):
                    queue.append(neighbor)
                    came_from[neighbor] = current

        # If the goal was not reached, return None
        if goal not in came_from:
            return None

        # Reconstruct path from goal to start
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path
