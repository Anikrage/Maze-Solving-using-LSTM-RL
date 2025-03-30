import numpy as np
import random
from collections import deque

class MazeGenerator:
    def __init__(self, size, seed=None):
        self.size = size
        self.maze = None
        self.rng = np.random.RandomState(seed)

    def generate_dfs_maze(self):
        size = self.size
        grid = np.ones((size * 2 + 1, size * 2 + 1))
        visited = np.zeros((size, size), dtype=bool)

        def dfs(row, col):
            visited[row][col] = True
            maze_row, maze_col = 2 * row + 1, 2 * col + 1
            grid[maze_row][maze_col] = 0

            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            self.rng.shuffle(directions)

            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                if (0 <= new_row < size and 0 <= new_col < size and not visited[new_row][new_col]):
                    grid[maze_row + dr][maze_col + dc] = 0
                    dfs(new_row, new_col)

        dfs(self.rng.randint(0, size), self.rng.randint(0, size))

        # Entrance & Exit
        grid[1][0] = 0
        grid[size * 2 - 1][size * 2] = 0

        self.maze = grid
        return grid
