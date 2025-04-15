import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

class MazeEnv(gym.Env):
    def __init__(self, maze, reward_type="dense"):  # Changed default from "sparse" to "dense"
        super(MazeEnv, self).__init__()
        self.maze = maze
        self.start = (1, 0)  # Entrance
        self.goal = (maze.shape[0] - 2, maze.shape[1] - 1)  # Exit
        self.state = self.start
        self.previous_state = self.start  # Keep track of previous state for reward calculation
        self.reward_type = reward_type
        self.max_steps = maze.shape[0] * maze.shape[1] * 2  # Reasonable step limit
        self.steps_taken = 0
        self.visited_positions = set()  # Track visited cells
        
        # For distance-based rewards
        self.initial_distance = self._manhattan_distance(self.start, self.goal)
        
        # Observation: maze with agent position marked as 2
        self.observation_space = spaces.Box(low=0, high=2, shape=maze.shape, dtype=np.uint8)
        self.action_space = spaces.Discrete(4)  # 0: Up, 1: Right, 2: Down, 3: Left


    
    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _get_obs(self):
        """Return a copy of the maze with the agent's position marked."""
        obs = self.maze.copy().astype(np.uint8)
        row, col = self.state
        obs[row, col] = 2
        return obs
    
    def _get_info(self):
        """Get additional information about the environment state"""
        return {
            "distance": self._manhattan_distance(self.state, self.goal),
            "steps": self.steps_taken,
            "visited_before": self.state in self.visited_positions
        }
    
    def _calculate_reward(self):
        """Calculate reward based on the selected reward type"""
        if self.state == self.goal:
            return 1.0  # High reward for reaching goal
        
        if self.reward_type == "dense":
            # Distance-based reward
            current_distance = self._manhattan_distance(self.state, self.goal)
            previous_distance = self._manhattan_distance(self.previous_state, self.goal)
            distance_reward = previous_distance - current_distance
            
            # Penalty for revisiting cells
            revisit_penalty = -0.2 if self.state in self.visited_positions else 0.0
            
            # Small step penalty to encourage efficiency
            step_penalty = -0.01
            
            return distance_reward + revisit_penalty + step_penalty
        else:
            # Sparse reward
            return -0.1  # Small penalty for each step
    
    def reset(self, seed=None, options=None):
        """Reset environment state and return observation."""
        super().reset(seed=seed)
        self.state = self.start
        self.previous_state = self.start
        self.steps_taken = 0
        self.visited_positions = set()
        self.visited_positions.add(self.start)
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        """Execute one step in the environment."""
        self.previous_state = self.state
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
        if (0 <= new_row < self.maze.shape[0] and 0 <= new_col < self.maze.shape[1] and
            self.maze[new_row, new_col] == 0):
            self.state = (new_row, new_col)
        
        # Add to visited positions set
        self.visited_positions.add(self.state)
        
        # Update step count and check for termination
        self.steps_taken += 1
        done = self.state == self.goal
        truncated = self.steps_taken >= self.max_steps
        
        reward = self._calculate_reward()
        info = self._get_info()
        
        return self._get_obs(), reward, done, truncated, info
    
    def render(self):
        """Print the maze with the agent's current position (for debugging)."""
        obs = self._get_obs()
        print(obs)
    
    def solve_with_bfs(self):
        """
        Use BFS to find an optimal solution path from start to goal.
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
                
                # Check if neighbor is valid
                if (0 <= neighbor[0] < self.maze.shape[0] and
                    0 <= neighbor[1] < self.maze.shape[1] and
                    self.maze[neighbor] == 0 and
                    neighbor not in came_from):
                    queue.append(neighbor)
                    came_from[neighbor] = current
        
        # If goal wasn't reached, return None
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
    
    def solve_with_trained_agent(self, model, deterministic=True):
        """
        Use the trained agent to solve the maze.
        Returns the agent's path through the maze.
        """
        # Reset the environment
        obs, _ = self.reset()
        
        path = [self.state]
        done = truncated = False
        
        # Run until reaching goal or max steps
        while not (done or truncated):
            # Get action from model
            action, _ = model.predict(obs, deterministic=deterministic)
            
            # Execute action
            obs, _, done, truncated, _ = self.step(action)
            
            # Add new position to path
            path.append(self.state)
            
            # Avoid infinite loops
            if len(path) > self.max_steps:
                break
        
        return path
