import numpy as np
import random
from collections import deque

class MazeGenerator:
    def __init__(self, size, seed=None):
        self.size = size
        self.maze = None
        self.rng = np.random.RandomState(seed)
    
    def generate_dfs_maze(self):
        """Generate a maze using Depth-First Search algorithm"""
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
    
    def generate_prims_maze(self):
        """Generate a maze using Prim's algorithm"""
        size = self.size
        grid = np.ones((size * 2 + 1, size * 2 + 1))
        
        # Initialize all cells as walls
        # Then mark all cell centers as passages
        for i in range(size):
            for j in range(size):
                grid[2*i+1][2*j+1] = 0
        
        # Start with a random cell
        start_row, start_col = self.rng.randint(0, size), self.rng.randint(0, size)
        
        # Track cells in the maze and frontier cells
        maze_cells = {(start_row, start_col)}
        frontier = []
        
        # Add initial frontier cells
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = start_row + dr, start_col + dc
            if 0 <= nr < size and 0 <= nc < size:
                frontier.append((nr, nc, start_row, start_col))
        
        # Continue until no more frontier cells
        while frontier:
            # Pick a random frontier cell
            idx = self.rng.randint(0, len(frontier))
            cell_row, cell_col, parent_row, parent_col = frontier.pop(idx)
            
            # Skip if already in maze
            if (cell_row, cell_col) in maze_cells:
                continue
            
            # Add to maze
            maze_cells.add((cell_row, cell_col))
            
            # Connect to parent by removing wall
            wall_row = cell_row + parent_row + 1
            wall_col = cell_col + parent_col + 1
            grid[wall_row][wall_col] = 0
            
            # Add new frontiers
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = cell_row + dr, cell_col + dc
                if (0 <= nr < size and 0 <= nc < size and 
                    (nr, nc) not in maze_cells):
                    frontier.append((nr, nc, cell_row, cell_col))
        
        # Entrance & Exit
        grid[1][0] = 0
        grid[size * 2 - 1][size * 2] = 0
        
        self.maze = grid
        return grid
    
    def generate_recursive_division(self):
        """Generate a maze using Recursive Division algorithm"""
        size = self.size
        # Initialize grid with all paths (0s)
        grid = np.zeros((size * 2 + 1, size * 2 + 1))
        
        # Set the border walls
        grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 1
        
        def divide(x, y, width, height):
            # Base case - if space is too small to divide
            if width < 2 or height < 2:
                return
                
            # Decide horizontal or vertical wall
            horizontal = self.rng.randint(0, 2) == 0 if width == height else width < height
                
            if horizontal:
                # Ensure we have space for wall + passage
                if height <= 2:
                    return
                    
                # Create horizontal wall with passage
                wall_y = y + self.rng.randint(1, height - 1)
                passage_x = x + self.rng.randint(0, width)
                
                for i in range(x, x + width):
                    if i != passage_x:
                        grid[wall_y, i] = 1
                
                # Recursively divide spaces
                if wall_y - y >= 2:
                    divide(x, y, width, wall_y - y)
                if y + height - wall_y - 1 >= 2:
                    divide(x, wall_y + 1, width, y + height - wall_y - 1)
            else:
                # Ensure we have space for wall + passage
                if width <= 2:
                    return
                    
                # Create vertical wall with passage
                wall_x = x + self.rng.randint(1, width - 1)
                passage_y = y + self.rng.randint(0, height)
                
                for i in range(y, y + height):
                    if i != passage_y:
                        grid[i, wall_x] = 1
                
                # Recursively divide spaces
                if wall_x - x >= 2:
                    divide(x, y, wall_x - x, height)
                if x + width - wall_x - 1 >= 2:
                    divide(wall_x + 1, y, x + width - wall_x - 1, height)
        
        # Start the recursive division
        divide(1, 1, 2 * size - 1, 2 * size - 1)
        
        # Ensure entrance & exit
        grid[1, 0] = 0
        grid[size * 2 - 1, size * 2] = 0
        
        self.maze = grid
        return grid
    
    def generate_maze(self, algorithm="dfs"):
        """Generate a maze using the specified algorithm"""
        if algorithm == "dfs":
            return self.generate_dfs_maze()
        elif algorithm == "prims":
            return self.generate_prims_maze()
        elif algorithm == "recursive_division":
            return self.generate_recursive_division()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
