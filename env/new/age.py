import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

# Constants for replay memory and training
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001  # Learning rate for the Q-network

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # Exploration rate
        self.gamma = 0.9  # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # Experience replay memory
        self.model = Linear_QNet(11, 256, 3)  # State has 11 features; action space is 3
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Define state representation
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Current direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Relative food position
            game.food.x < game.head.x,  # Food to the left
            game.food.x > game.head.x,  # Food to the right
            game.food.y < game.head.y,  # Food above
            game.food.y > game.head.y  # Food below
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        # Store experience in memory for experience replay
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        # Sample a mini-batch from memory and train
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        # Train on a single experience step
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Epsilon-greedy action selection
        self.epsilon = max(30 - self.n_games, 10)  # Minimum epsilon of 10
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)  # Q-values for each action
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train():
    plot_scores = []
    plot_rewards = []
    total_score = 0
    total_reward = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # Get current state
        state_old = agent.get_state(game)

        # Decide action based on state
        final_move = agent.get_action(state_old)

        # Perform action and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Train short memory and store experience
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Train long memory on episode completion and reset game
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            # Record best score and save model
            if score > record:
                record = score
                agent.model.save()

            # Print and log training progress
            print(f'Game {agent.n_games}, Score: {score}, Reward: {reward}, Record: {record}, Epsilon: {agent.epsilon:.2f}')

            plot_scores.append(score)
            plot_rewards.append(reward)
            total_score += score
            total_reward += reward
            plot(plot_scores, plot_rewards)

if __name__ == '__main__':
    train()