import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class LSTM_Agent(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTM_Agent, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Feature extraction layers for spatial understanding
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU()
        )
        
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights with Xavier for better convergence
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        # Ensure input is properly shaped
        if len(x.shape) == 4:  # If input is (batch, seq, height, width)
            x = x.reshape(batch_size, x.size(1), -1)
        elif len(x.shape) == 2:  # If input is (batch, features)
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # Extract features
        features = self.feature_extractor(x.view(batch_size * x.size(1), -1))
        features = features.view(batch_size, x.size(1), -1)
        
        # Initialize hidden state if not provided
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
            hidden = (h0, c0)
        
        # Forward pass through LSTM
        lstm_out, hidden = self.lstm(features, hidden)
        
        # Get output for last time step
        output = self.fc(lstm_out[:, -1, :])
        
        return output, hidden

    def get_action(self, state, hidden=None, epsilon=0.0):
        """Get action with epsilon-greedy policy"""
        if random.random() < epsilon:
            return random.randint(0, 3), hidden
            
        # Convert state to tensor and get Q-values
        with torch.no_grad():
            # Flatten the state and add batch dimension
            if isinstance(state, np.ndarray) and len(state.shape) > 1:
                state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0).unsqueeze(0)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
                
            q_values, new_hidden = self.forward(state_tensor, hidden)
            action = torch.argmax(q_values).item()
            
        return action, new_hidden

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha  # Priority exponent
        self.max_priority = 1.0
        
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer with max priority"""
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(self.max_priority)
    
    def sample(self, batch_size):
        """Sample a batch of experiences based on priorities"""
        if len(self.buffer) < batch_size:
            return random.sample(self.buffer, len(self.buffer)), None, None
            
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample experiences
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        
        return samples, indices, np.ones(batch_size)
        
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        for idx, error in zip(indices, td_errors):
            # Add a small constant to avoid zero priority
            priority = np.abs(error) + 1e-5
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)

class RL_Agent:
    def __init__(self, input_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.999,
                min_epsilon=0.05, buffer_size=50000, batch_size=128, update_every=4, hidden_dim=256):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.update_every = update_every
        self.step_counter = 0
        self.input_dim = input_dim
        
        # Create Q networks (main and target)
        self.q_network = LSTM_Agent(input_dim, hidden_dim, action_dim)
        self.target_network = LSTM_Agent(input_dim, hidden_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer with gradient clipping
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Prioritized replay memory
        self.memory = PrioritizedReplayBuffer(buffer_size)
        
        # For storing training metrics
        self.loss_history = []
        self.reward_history = []
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network.to(self.device)
        self.target_network.to(self.device)
        
        print(f"Agent initialized with input_dim={input_dim}, hidden_dim={hidden_dim}, using {self.device}")
    
    def preprocess_state(self, state):
        """Preprocess state for LSTM input"""
        if isinstance(state, np.ndarray) and len(state.shape) > 1:
            return state.flatten()
        return state
    
    def act(self, state, hidden=None):
        """Select an action using epsilon-greedy policy"""
        return self.q_network.get_action(state, hidden, self.epsilon)
    
    def step(self, state, action, reward, next_state, done):
        """Store experience in replay memory and learn if needed"""
        # Add experience to memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every update_every time steps
        self.step_counter += 1
        if self.step_counter % self.update_every == 0:
            if len(self.memory) > self.batch_size:
                experiences, indices, weights = self.memory.sample(self.batch_size)
                self.learn(experiences, indices, weights)
    
    def learn(self, experiences, indices, weights):
        """Update Q-Network weights based on batch of experiences"""
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        # Preprocess states
        states = [self.preprocess_state(s) for s in states]
        next_states = [self.preprocess_state(s) for s in next_states]
        
        try:
            # Convert to tensors
            states = torch.FloatTensor(np.array(states)).to(self.device)
            next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
            weights = torch.FloatTensor(weights).to(self.device)
            
            # Add sequence dimension if missing
            if len(states.shape) == 2:
                states = states.unsqueeze(1)
            if len(next_states.shape) == 2:
                next_states = next_states.unsqueeze(1)
            
            # Get Q values for current states
            q_values, _ = self.q_network(states)
            q_values = q_values.gather(1, actions)
            
            # Get max Q values for next states from target network
            with torch.no_grad():
                next_q_values, _ = self.target_network(next_states)
                next_q_values = next_q_values.max(1)[0].unsqueeze(1)
            
            # Compute target Q values
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
            
            # Compute TD errors for updating priorities
            td_errors = (q_values - target_q_values).detach().cpu().numpy()
            
            # Compute loss
            loss = F.mse_loss(q_values, target_q_values)
            self.loss_history.append(loss.item())
            
            # Minimize the loss
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            self.optimizer.step()
            
            # Update priorities in replay buffer
            self.memory.update_priorities(indices, np.abs(td_errors.squeeze()))
            
            # Update target network
            self.soft_update(self.q_network, self.target_network, 0.001)
            
            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        except Exception as e:
            print(f"Error during learning: {e}")
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters: θ_target = τ*θ_local + (1 - τ)*θ_target"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def save(self, filename):
        """Save the model to disk"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_counter': self.step_counter,
            'loss_history': self.loss_history,
            'reward_history': self.reward_history,
            'input_dim': self.input_dim,
            'action_dim': self.action_dim,
            'hidden_dim': self.q_network.hidden_dim
        }, filename)
        print(f"Model saved to {filename}")
    
    def load(self, filename, eval_mode=False, weights_only=False):
        """Load the model from disk"""
        checkpoint = torch.load(filename, map_location=self.device, weights_only=weights_only)
        
        # Check if input dimensions match
        loaded_input_dim = checkpoint.get('input_dim', self.input_dim)
        if loaded_input_dim != self.input_dim:
            print(f"Warning: Loaded model input dimension ({loaded_input_dim}) doesn't match current ({self.input_dim})")
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon'] if not eval_mode else self.min_epsilon
        self.step_counter = checkpoint['step_counter']
        self.loss_history = checkpoint['loss_history']
        
        if 'reward_history' in checkpoint:
            self.reward_history = checkpoint['reward_history']
        
        if eval_mode:
            self.q_network.eval()
            self.target_network.eval()
        else:
            self.q_network.train()
            self.target_network.train()
        
        print(f"Model loaded from {filename}")
    
    def train_episode(self, env, max_steps=1000, render=False):
        """Train for one episode"""
        state, _ = env.reset()
        total_reward = 0
        hidden = None
        
        for step in range(max_steps):
            # Select action
            action, hidden = self.act(state, hidden)
            
            # Take action
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store experience and learn
            self.step(state, action, reward, next_state, done or truncated)
            
            # Update state and total reward
            state = next_state
            total_reward += reward
            
            # Render if required
            if render:
                env.render()
            
            # Break if episode ended
            if done or truncated:
                break
        
        self.reward_history.append(total_reward)
        return total_reward
