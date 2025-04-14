import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

class MazeEnv(gym.Env):
    def __init__(self, maze):
        super(MazeEnv, self).__init__()
        self.maze = maze
        self.start = (1, 0)
        self.goal = (maze.shape[0] - 2, maze.shape[1] - 1)
        self.state = self.start
        self.observation_space = spaces.Box(low=0, high=2, shape=maze.shape, dtype=np.uint8)
        self.action_space = spaces.Discrete(4)

    def _get_obs(self):
        obs = self.maze.copy().astype(np.uint8)
        row, col = self.state
        obs[row, col] = 2
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.start
        return self._get_obs(), {}

    def step(self, action):
        row, col = self.state
        new_row, new_col = row, col
        if action == 0: new_row -= 1
        elif action == 1: new_col += 1
        elif action == 2: new_row += 1
        elif action == 3: new_col -= 1

        if 0 <= new_row < self.maze.shape[0] and 0 <= new_col < self.maze.shape[1]:
            if self.maze[new_row, new_col] == 0:
                self.state = (new_row, new_col)

        done = self.state == self.goal
        reward = 1 if done else -0.1
        return self._get_obs(), reward, done, False, {}

    def solve_with_trained_agent(self, model=None):
        if model is not None:
            obs, _ = self.reset()
            path = [self.state]
            for _ in range(500):
                action, _ = model.predict(obs)
                obs, _, done, _, _ = self.step(action)
                path.append(self.state)
                if done:
                    return path
            return None
        start = self.start
        goal = self.goal
        queue = deque([start])
        came_from = {start: None}

        while queue:
            current = queue.popleft()
            if current == goal: break
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if (0 <= neighbor[0] < self.maze.shape[0] and
                    0 <= neighbor[1] < self.maze.shape[1] and
                    self.maze[neighbor] == 0 and
                    neighbor not in came_from):
                    queue.append(neighbor)
                    came_from[neighbor] = current

        if goal not in came_from: return None
        path, current = [], goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path