import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class LSTM_Agent(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTM_Agent, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h, _ = self.lstm(x)
        return self.fc(h[:, -1, :])

class RL_Agent:
    def __init__(self, state_dim, action_dim, lr=0.001):
        self.model = LSTM_Agent(state_dim, 128, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.memory = []

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 1000:
            self.memory.pop(0)

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        q_values = self.model(states)
        next_q_values = self.model(next_states).detach()

        target_q_values = rewards + (1 - dones) * 0.99 * next_q_values.max(1)[0]
        loss = self.criterion(q_values, target_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
