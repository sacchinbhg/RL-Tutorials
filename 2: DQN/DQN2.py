import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import pickle
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from time import sleep
from datetime import datetime
from math import sqrt

filename1 = "/home/sacchin/Desktop/dnt/RL_Tutorials/Grid Worlds/0.1grid_world.pkl"

# Check for GPU and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class GeneralGridWorld:
    def __init__(self, filename, size=100):
        self.size = size
        self.filename = filename
        self.reset()

    def reset(self):
        self.agent_pos = (0, 0)
        self.goal = (50, 50)
        self.obstacles = self.generate_obstacles()
        self.validate_positions()
        self.state = np.zeros((self.size, self.size))
        return self.get_state()

    def generate_obstacles(self):
        obstacles = set()
        with open(self.filename, 'rb') as file:
            temp_gw = pickle.load(file)

        for i, value1 in enumerate(temp_gw):
            for j, value2 in enumerate(value1):
                if value2==1:
                    obstacles.add((i,j))
        return obstacles

    def validate_positions(self):
        if self.agent_pos in self.obstacles or self.goal in self.obstacles:
            raise ValueError("Agent's initial position or goal cannot be on an obstacle.")

    def get_state(self):
        # Create a state representation, for example, a flattened grid
        self.state = np.zeros((self.size, self.size))
        self.state[self.agent_pos] = 1  # Represent the agent
        self.state[self.goal] = 2  # Represent the goal
        for obs in self.obstacles:
            self.state[obs] = -1  # Represent obstacles
        return self.state.flatten()

    def step(self, action):
        # Define actions: 0=Up, 1=Down, 2=Left, 3=Right
        next_pos = list(self.agent_pos)
        if action == 0 and self.agent_pos[0] > 0:
            next_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < self.size - 1:
            next_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0:
            next_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < self.size - 1:
            next_pos[1] += 1
        next_pos = tuple(next_pos)

        reward = (1-(sqrt((next_pos[0]-50)**2+(next_pos[1]-50)**2))/(sqrt(2)*50))*100
        done = False
        if next_pos in self.obstacles:
            reward = -100  # Penalty for hitting an obstacle
            done = True
        elif next_pos == self.goal:
            reward = 100000  # Reward for reaching the goal
            done = True
        if action == 0 and self.agent_pos[0] == 0:
            reward = -100
        if action == 1 and self.agent_pos[0] == self.size - 1:
            reward = -100
        if action == 2 and self.agent_pos[1] == 0:
            reward = -100
        if action == 3 and self.agent_pos[1] == self.size - 1:
            reward = -100

        self.agent_pos = next_pos
        return self.get_state(), reward, done

    def render(self):
        grid = self.state.reshape(self.size, self.size)
        # Scale up the grid for better visualization
        grid = cv2.resize(grid, (400, 400), interpolation=cv2.INTER_NEAREST)
        # Create a colormap for visualization
        cmap = plt.get_cmap('hot')
        # Normalize the grid values to the range [0, 1] for colormap
        normalized_grid = (grid - grid.min()) / (grid.max() - grid.min())
        # Apply the colormap
        heatmap = (cmap(normalized_grid) * 255).astype(np.uint8)
        # Display the heatmap using OpenCV
        cv2.imshow('Grid World', heatmap)
        cv2.waitKey(100)  # Wait for 100 milliseconds


class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.5  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.5
        self.epsilon_decay = 0.99999
        self.learning_rate = 0.001
        self.model = DQNNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model = self.model.to(device)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
            q_values = self.model(state)
            return torch.argmax(q_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for experience in minibatch:
            state, action, reward, next_state, done = experience

            # Convert numpy arrays to tensors and move them to the specified device
            state = torch.from_numpy(state).float().to(device)
            next_state = torch.from_numpy(next_state).float().to(device)
            reward = torch.tensor(reward).float().to(device)
            action = torch.tensor(action).long().to(device)  # Actions are typically long integers
            done = torch.tensor(done).float().to(device)  # done flag as float for computations

            # Compute the target Q-value
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state).detach())
            
            # Get the current Q-value predictions
            current_q = self.model(state)[action]

            # Calculate loss
            loss = nn.MSELoss()(current_q, target)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, file_name):
        torch.save(self.model.state_dict(), file_name)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))


class EpisodeData:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def add_step(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)


def render_state(state, size=100):
    grid = state.reshape(size, size)
    grid = cv2.resize(grid, (400, 400), interpolation=cv2.INTER_NEAREST)
    cmap = plt.get_cmap('hot')
    normalized_grid = (grid - grid.min()) / (grid.max() - grid.min())
    heatmap = (cmap(normalized_grid) * 255).astype(np.uint8)
    cv2.imshow('Replay Grid World', heatmap)
    cv2.waitKey(100)  # Adjust the wait time as needed for visualization


def replay_episode(episode_data, env, size=100):
    env.reset()
    for state, action, reward in zip(episode_data.states, episode_data.actions, episode_data.rewards):
        render_state(state, size)
        print(f"Action: {action}, Reward: {reward}")
        sleep(0.1)  # Delay for visualization purposes
    cv2.destroyAllWindows()  # Close the window after replay is done


# Main training loop with modifications to store and replay the best episode
env = GeneralGridWorld(filename1)
agent = DQNAgent(100*100, 4)
agent.load("/home/sacchin/Desktop/dnt/RL_Tutorials/2: DQN/models/dqn_model_nye.pth")
episodes = 1000
sequential_rewards = []

best_episode = None
highest_reward = -float('inf')

for e in tqdm(range(episodes)):
    state = env.reset()
    current_episode = EpisodeData()
    tot_reward = 0

    for time in range(500):
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        tot_reward += reward

        current_episode.add_step(state, action, reward)

        state = next_state
        if done:
            break

    if tot_reward > highest_reward:
        highest_reward = tot_reward
        best_episode = current_episode

    agent.replay(32)
    sequential_rewards.append(tot_reward)

plt.plot(sequential_rewards)
plt.show()

model_save_path = "/home/sacchin/Desktop/dnt/RL_Tutorials/2: DQN/models/dqn_model_nye.pth"
agent.save(model_save_path)
print(f"Model saved to {model_save_path}")

# Replay the best episode after training
print(f"Replaying best episode with total reward: {highest_reward}")
replay_episode(best_episode, env)