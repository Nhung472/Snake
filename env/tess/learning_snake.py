import pygame
import time
import random
import math
from util import State
import numpy as np
import argparse

CONTROLS = False

class Snake:
    def __init__(self, dis, height, width, size, debug_mode) -> None:
        self.size = size
        self.dis = dis
        self.height = height
        self.width = width
        self.debug_mode = debug_mode

    def reset(self) -> State:
        foodx = 270
        foody = 300
        self.dis.fill("black")
        state = State(300, 300, 1, foodx, foody, [])
        return state

    def collide(self, head, state):
        #add tail check later
        if head in state.snake_list[:-1]:
            if self.debug_mode:
                print("Hit snake body")
            return True

        if state.x < 0 or state.x > self.width-self.size or state.y<0 or state.y > self.height-self.size:
            # print("crashing")
            if self.debug_mode:
                print("Crashed into wall")
            return True 
        
        return False
    
    def reward_distance_to_fruit(self, state, new_state):
        return 0
        new_distance = abs(new_state.x - new_state.foodx) + abs(new_state.y - new_state.foody)
        old_distance = abs(state.x - state.foodx) + abs(state.y - state.foody)
        if new_distance - old_distance < 0:
            return 1
        
        return -1

    def choices(self, state, action):
        moves = {"up":[0,-self.size], "down":[0,self.size], "left":[-self.size,0], "right":[self.size,0]} #up, down left, right
        if action not in moves:
            print("Invalid action")
            exit(0)
        move = moves[action]
        temp_x = state.x + move[0]
        temp_y = state.y + move[1]
        new_state = State(temp_x, temp_y, state.length, state.foodx, state.foody, state.snake_list)
        reward = self.reward_distance_to_fruit(state, new_state)
        
        if temp_x == state.foodx and temp_y == state.foody:
            reward = 10
            new_state.length += 1
            new_state.foodx = round(random.randrange(0, self.width-self.size) / self.size) * self.size
            new_state.foody = round(random.randrange(0, self.width-self.size) / self.size) * self.size
            while [new_state.foodx, new_state.foody] in new_state.snake_list:
                new_state.foodx = round(random.randrange(0, self.width-self.size) / self.size) * self.size
                new_state.foody = round(random.randrange(0, self.width-self.size) / self.size) * self.size

        if temp_x < 0 or temp_x > self.width-self.size or temp_y<0 or temp_y > self.height-self.size or [temp_x, temp_y] in new_state.snake_list[:-1]:
            reward = -1000

        return new_state, reward              

    def draw(self, state, trial):

        blue=(0,0,255)
        red=(255,0,0)  
        white = (255, 255, 255) 
        score_font = pygame.font.SysFont("comicsansms", 35)
        mesg = score_font.render(f"Trial: {trial}", True, white)

        snake_Head = []
        snake_Head.append(state.x)
        snake_Head.append(state.y)
        state.snake_list.append(snake_Head)
        if len(state.snake_list) > state.length:
            del state.snake_list[0]

        self.dis.fill("black")
        self.dis.blit(mesg, [0, 0])
        for x,y in state.snake_list:
            pygame.draw.rect(self.dis,blue,[x,y,self.size,self.size])
        pygame.draw.rect(self.dis,red,[state.foodx,state.foody,self.size,self.size])
        pygame.display.update()

class QLearningAlgorithm:
    def __init__(self, actions, discount, learning_rate, epsilon):
        self.actions = actions
        self.discount = discount
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.q_values = {}

    def getAction(self, state, trial):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            q_values = [self.q_values.get((state.x, state.y, action), 0) for action in self.actions]
            return self.actions[np.argmax(q_values)]

    def incorporateFeedback(self, state, action, reward, new_state):
        q_value = self.q_values.get((state.x, state.y, action), 0)
        new_q_values = [self.q_values.get((new_state.x, new_state.y, new_action), 0) for new_action in self.actions]
        new_q_value = max(new_q_values)
        self.q_values[(state.x, state.y, action)] = (1 - self.learning_rate) * q_value + self.learning_rate * (reward + self.discount * new_q_value)

def simulate(dis, height, width, debug_mode, numTrials=300, maxIterations=10000):
    actions = ["up","down","left","right"]
    discount = 0.9  # discount factor
    size = 30
    learning_rate = 0.1
    epsilon = 0.1
    qlearn = QLearningAlgorithm(actions, discount, learning_rate, epsilon)
    snake = Snake(dis, height, width, size, debug_mode)
    total_rewards = []
    clock = pygame.time.Clock()
    timer = 20

    for trial in range(numTrials):
        state = snake.reset()
        totalDiscount = 1  # future
        trial_reward = 0
        skip = False
        for _ in range(maxIterations):
            action = qlearn.getAction(state, trial+1)
            new_state, reward = snake.choices(state, action)
            if reward < -1000:
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

def start_display():
    height = 600
    width = 600
    pygame.init()
    dis = pygame.display.set_mode((height, width))
    pygame.display.set_caption('Reinforcement Learning Snake')

    # Initialize game board
    board = [[0 for _ in range(width // 30)] for _ in range(height // 30)]

    # Run the learning process
    total_reward = learn(dis, height, width, False)
    print(total_reward)

    # Display the game board
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Draw the game board
        dis.fill((255, 255, 255))  # White background
        for i in range(height // 30):
            for j in range(width // 30):
                if board[i][j] == 1:  # Snake body
                    pygame.draw.rect(dis, (0, 0, 0), (j * 30, i * 30, 30, 30))
                elif board[i][j] == 2:  # Food
                    pygame.draw.rect(dis, (255, 0, 0), (j * 30, i * 30, 30, 30))

        pygame.display.update()

# Add the rest of your game logic here

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