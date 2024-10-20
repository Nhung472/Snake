import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

# Constants
WIDTH, HEIGHT = 600, 600
BLOCK_SIZE = 20
REWARD_EAT_FOOD = 10
REWARD_ELSE = -0.1
REWARD_NEAR_FOOD = 1
REWARD_FAR_FOOD = -1
REWARD_DIE = -10
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.9
EXPLORATION_RATE = 0.1
ACTIONS = ['up', 'down', 'left', 'right']

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Pygame Initialization
pygame.init()
wn = pygame.display.set_mode((WIDTH, HEIGHT))

class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.snake = [(200, 200), (220, 200), (240, 200)]
        self.food = self.generate_food()
        self.score = 0
        self.direction = 'right'

    def generate_food(self):
        while True:
            food = (random.randint(0, WIDTH - BLOCK_SIZE) // BLOCK_SIZE * BLOCK_SIZE,
                    random.randint(0, HEIGHT - BLOCK_SIZE) // BLOCK_SIZE * BLOCK_SIZE)
            if food not in self.snake:
                return food

    def step(self, action):
        reward = REWARD_ELSE
        if action == 'up' and self.direction != 'down':
            self.direction = 'up'
        elif action == 'down' and self.direction != 'up':
            self.direction = 'down'
        elif action == 'left' and self.direction != 'right':
            self.direction = 'left'
        elif action == 'right' and self.direction != 'left':
            self.direction = 'right'

        head = self.snake[-1]
        if self.direction == 'up':
            new_head = (head[0], head[1] - BLOCK_SIZE)
        elif self.direction == 'down':
            new_head = (head[0], head[1] + BLOCK_SIZE)
        elif self.direction == 'left':
            new_head = (head[0] - BLOCK_SIZE, head[1])
        elif self.direction == 'right':
            new_head = (head[0] + BLOCK_SIZE, head[1])

        self.snake.append(new_head)

        if self.snake[-1] == self.food:
            reward = REWARD_EAT_FOOD
            self.score += 1
            self.food = self.generate_food()
        else:
            self.snake.pop(0)

        if (self.snake[-1][0] < 0 or self.snake[-1][0] > WIDTH - BLOCK_SIZE or
            self.snake[-1][1] < 0 or self.snake[-1][1] > HEIGHT - BLOCK_SIZE or
            self.snake[-1] in self.snake[:-1]):
            reward = REWARD_DIE
            return None, reward

        return self.get_state(), reward

    def get_state(self):
        state = [0] * 11
        head = self.snake[-1]
        for i in range(1, 11):
            if head[0] - i * BLOCK_SIZE >= 0 and (head[0] - i * BLOCK_SIZE, head[1]) in self.snake:
                state[0] = 1
                break
            elif head[0] + i * BLOCK_SIZE < WIDTH and (head[0] + i * BLOCK_SIZE, head[1]) in self.snake:
                state[1] = 1
                break
            elif head[1] - i * BLOCK_SIZE >= 0 and (head[0], head[1] - i * BLOCK_SIZE) in self.snake:
                state[2] = 1
                break
            elif head[1] + i * BLOCK_SIZE < HEIGHT and (head[0], head[1] + i * BLOCK_SIZE) in self.snake :
                state[3] = 1
                break

        if self.food[0] < head[0]:
            state[4] = 1
        elif self.food[0] > head[0]:
            state[5] = 1
        if self.food[1] < head[1]:
            state[6] = 1
        elif self.food[1] > head[1]:
            state[7] = 1

        if self.direction == 'up':
            state[8] = 1
        elif self.direction == 'down':
            state[9] = 1
        elif self.direction == 'left':
            state[10] = 1

        return state

class Agent:
    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(11, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float)
        action_values = self.model(state)
        _, action = torch.max(action_values, 0)
        return ACTIONS[action.item()]

    def learn(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        action_values = self.model(state)
        target = action_values.clone()
        target[ACTIONS.index(action)] = reward
        self.optimizer.zero_grad()
        loss = nn.MSELoss()(action_values, target)
        loss.backward()
        self.optimizer.step()

def draw_game(snake_game):
    wn.fill(BLACK)
    for pos in snake_game.snake:
        pygame.draw.rect(wn, GREEN, (pos[0], pos[1], BLOCK_SIZE, BLOCK_SIZE))
    pygame.draw.rect(wn, RED, (snake_game.food[0], snake_game.food[1], BLOCK_SIZE, BLOCK_SIZE))
    pygame.display.update()

def main():
    try:
        scores = []
        rewards = []
        num_games = 1000
        for _ in range(num_games):
            snake_game = SnakeGame()
            agent = Agent()
            total_reward = 0
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()

                state = snake_game.get_state()
                action = agent.get_action(state)
                next_state, reward = snake_game.step(action)
                if next_state is None:
                    break
                agent.learn(state, ACTIONS.index(action), reward, next_state, False)
                draw_game(snake_game)
                pygame.time.delay(50)  # Adjust the delay as needed
                total_reward += reward

            scores.append(snake_game.score)
            rewards.append(total_reward)

        plt.plot(range(1, num_games + 1), scores, label='Score')
        plt.plot(range(1, num_games + 1), rewards, label='Reward')
        plt.xlabel('Number of Games')
        plt.ylabel('Score/Reward')
        plt.title('Score and Reward Over Time')
        plt.legend()
        plt.show()

    except Exception as e:
        print("Error running game:", e)

if __name__ == "__main__":
    main()