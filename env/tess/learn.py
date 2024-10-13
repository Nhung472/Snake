from learning_snake import Snake
from util import QLearningAlgorithm, featureExtractor, State
import pygame
import argparse
import numpy as np

import numpy as np
import pygame
import argparse

class State:
    def __init__(self, x, y, food_distance, obstacle_distance):
        self.x = x
        self.y = y
        self.food_distance = food_distance
        self.obstacle_distance = obstacle_distance

class QLearningAlgorithm:
    def __init__(self, actions, discount, featureExtractor, height, width, size, learning_rate):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.height = height
        self.width = width
        self.size = size
        self.learning_rate = learning_rate
        self.q_values = np.zeros((height, width, len(actions)))

    def getAction(self, state, debug_mode, trial):
        features = self.featureExtractor(state)
        q_values = self.q_values[state.x, state.y, :]
        action = np.argmax(q_values)
        if debug_mode:
            print(f"Trial: {trial}, State: ({state.x}, {state.y}), Q-values: {q_values}, Action: {self.actions[action]}")
        return action

    def incorporateFeedback(self, state, action, reward, new_state):
        features = self.featureExtractor(state)
        q_values = self.q_values[state.x, state.y, :]
        new_q_values = self.q_values[new_state.x, new_state.y, :]
        q_values[action] = (1 - self.learning_rate) * q_values[action] + self.learning_rate * (reward + self.discount * np.max(new_q_values))

class DoubleQLearningAlgorithm:
    def __init__(self, actions, discount, featureExtractor, height, width, size, learning_rate):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.height = height
        self.width = width
        self.size = size
        self.learning_rate = learning_rate
        self.q_values1 = np.zeros((height, width, len(actions)))
        self.q_values2 = np.zeros((height, width, len(actions)))

    def getAction(self, state, debug_mode, trial):
        features = self.featureExtractor(state)
        q_values1 = self.q_values1[state.x, state.y, :]
        q_values2 = self.q_values2[state.x, state.y, :]
        action = np.argmax(q_values1 + q_values2)
        if debug_mode:
            print(f"Trial: {trial}, State: ({state.x}, {state.y}), Q-values: {q_values1 + q_values2}, Action: {self.actions[action]}")
        return action

    def incorporateFeedback(self, state, action, reward, new_state):
        features = self.featureExtractor(state)
        q_values1 = self.q_values1[state.x, state.y, :]
        q_values2 = self.q_values2[state.x, state.y, :]
        new_q_values1 = self.q_values1[new_state.x, new_state.y, :]
        new_q_values2 = self.q_values2[new_state.x, new_state.y, :]
        q_values1[action] = (1 - self.learning_rate) * q_values1[action] + self.learning_rate * (reward + self.discount * np.max(new_q_values2))
        q_values2[action] = (1 - self.learning_rate) * q_values2[action] + self.learning_rate * (reward + self.discount * np.max(new_q_values1))

class Snake:
    def __init__(self, display, height, width, size, debug_mode):
        self.display = display
        self.height = height
        self.width = width
        self.size = size
        self.debug_mode = debug_mode
        self.x = width // 2
        self.y = height // 2
        self.food_distance = 0
        self.obstacle_distance = 0

    def reset(self):
        self.x = self.width // 2
        self.y = self.height // 2
        self.food_distance = 0
        self.obstacle_distance = 0
        return State(self.x, self.y, self.food_distance, self.obstacle_distance)

    def choices(self, state, action):
        if action == "up":
            self.y -= self.size
        elif action == "down":
            self.y += self.size
        elif action == "left":
            self.x -= self.size
        elif action == "right":
            self.x += self.size
        reward = -1
        if self.x < 0 or self.x >= self.width or self.y < 0 or self.y >= self.height:
            reward = -10
        return State(self.x, self.y, self.food_distance, self.obstacle_distance), reward

    def draw(self, state, trial):
        self.display.fill((0, 0, 0))
        pygame.draw.rect(self.display, (255, 255, 255), (state.x, state.y, self.size, self.size))
        pygame.display.update()

def featureExtractor(state):
    features = []
    features.append(state.x / 600)  # normalized x-coordinate
    features.append(state.y / 600)  # normalized y-coordinate
    features.append(state.food_distance / (600 + 600))  # normalized food distance
    features.append(state.obstacle_distance / (600 + 600))  # normalized obstacle distance
    return features

def simulate(display, height, width, debug_mode, numTrials=300, maxIterations=10000):
    actions = ["up","down","left","right"]
    discount = 0.9  # discount factor
    size = 30
    qlearn = DoubleQLearningAlgorithm(actions, discount, featureExtractor, height, width, size, 0.1)  # learning rate
    snake = Snake(display, height, width, size, debug_mode)
    total_rewards = []
    clock = pygame.time.Clock()
    timer = 20

    for trial in range(numTrials):
        state = snake.reset()
        totalDiscount = 1  # future
        trial_reward = 0
        skip = False
        for _ in range(maxIterations):
            action = qlearn.getAction(state, debug_mode, trial+1)
            new_state, reward = snake.choices(state, action)
            if reward < -10:
                qlearn.incorporateFeedback(state, action, reward, new_state)
                break
            qlearn.incorporateFeedback(state, action, reward, new_state)
            trial_reward += totalDiscount * reward
            totalDiscount *= discount
            state = new_state
            snake.draw(state, trial)
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == 1073741906:
                        timer += 100
                    if event.key == 1073741905:
                        timer -= 100
                    if event.key == 100:
                        if not debug_mode:
                            debug_mode = True
                            snake.debug_mode = True
                            skip = False
            clock.tick(timer)
        total_rewards.append(trial_reward)
    return total_rewards

def learn():
    parser = argparse.ArgumentParser(description='Run program in debug mode')
    parser.add_argument('-d', action='store_true', help='Run program in debug mode')
    args = parser.parse_args()
    height = 600
    width = 600
    pygame.init()
    dis=pygame.display.set_mode((height,width))
    pygame.display.set_caption('Reinforcement Learning Snake')
    debug_mode = False
    if args.d:
        debug_mode = True
    total_reward = simulate(dis,height,width, debug_mode)
    print(total_reward)
    pygame.quit()
    quit()

learn()