import pygame
import time
import random
import math
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional, NamedTuple, Callable

# Feature named tuple for storing features and their values
Feature = NamedTuple('Feature', [('featureKey', Tuple), ('featureValue', int)])

# State class to represent the game state
class State:
    def __init__(self, x, y, length, foodx, foody, snake_list) -> None:
        self.x = x
        self.y = y
        self.length = length
        self.foodx = foodx
        self.foody = foody
        self.snake_list = snake_list

# QLearningAlgorithm class implementing the Q-learning algorithm
class QLearningAlgorithm:
    def __init__(self, actions: List, discount: float, featureExtractor: Callable, height: int, width: int, size: int, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0
        self.height = height
        self.width = width
        self.size = size

    def getQ(self, state: State, action: Any) -> float:
        score = 0
        featureValues, _ = self.featureExtractor(state, action, self.height, self.width, self.size)
        for f, v in featureValues:
            score += self.weights[f] * v
        return score

    def getAction(self, state: State, debug_mode: Optional[bool] = False, trial: Optional[int] = 0) -> Tuple[Any, Optional[List], Optional[List]]:
        self.numIters += 1
        moves = {"up": [0, -self.size], "down": [0, self.size], "left": [-self.size, 0], "right": [self.size, 0]}
        actions = []

        for key in moves:
            delta_vals = moves[key]
            if len(state.snake_list) > 1 and [state.x + delta_vals[0], state.y + delta_vals[1]] == state.snake_list[-2]:
                continue
            actions.append(key)

        if random.random() < self.explorationProb:
            return random.choice(actions), None, None  # Return action with placeholders for debug info
        else:
            action = max(actions, key=lambda action: self.getQ(state, action))
            return action, None, None  # Return action with placeholders for debug info

    def getStepSize(self) -> float:
        return 1.0 / math.sqrt(self.numIters)

    def incorporateFeedback(self, state: State, action: Any, reward: int, newState: State) -> None:
        featureValues, _ = self.featureExtractor(state, action, self.height, self.width, self.size)
        max_action_value = float('-inf')
        for new_action in self.actions:
            temp_value = self.getQ(newState, new_action)
            if temp_value > max_action_value:
                max_action_value = temp_value

        value = self.getQ(state, action)
        difference = reward + self.discount * max_action_value - value

        alpha = self.getStepSize()
        for f, v in featureValues:
            self.weights[f] += alpha * difference * v

# Feature extraction function
def featureExtractor(state: State, action: Any, height: int, width: int, size: int):
    features = []
    
    # Check relative positions of the snake and food
    features.append(Feature(featureKey=('foodRight', action), featureValue=int(state.x > state.foodx)))
    features.append(Feature(featureKey=('foodLeft', action), featureValue=int(state.x < state.foodx)))
    features.append(Feature(featureKey=('foodUp', action), featureValue=int(state.y < state.foody)))
    features.append(Feature(featureKey=('foodDown', action), featureValue=int(state.y > state.foody)))

    # Danger checks
    features.append(Feature(featureKey=('dangerUp', action), featureValue=int(state.y == 0 or [state.x, state.y - size] in state.snake_list[:-1])))
    features.append(Feature(featureKey=('dangerDown', action), featureValue=int(state.y == height - size or [state.x, state.y + size] in state.snake_list[:-1])))
    features.append(Feature(featureKey=('dangerLeft', action), featureValue=int(state.x == 0 or [state.x - size, state.y] in state.snake_list[:-1])))
    features.append(Feature(featureKey=('dangerRight', action), featureValue=int(state.x == width - size or [state.x + size, state.y] in state.snake_list[:-1])))

    return features, None  # Ensure you return two values

# Snake game class
class Snake:
    def __init__(self, dis, height, width) -> None:
        self.size = 30
        self.dis = dis
        self.height = height
        self.width = width
        self.x = 300
        self.y = 300
        self.snake_length = 1
        self.foodx = round(random.randrange(0, self.width - self.size) / self.size) * self.size
        self.foody = round(random.randrange(0, self.height - self.size) / self.size) * self.size
        self.snake_list = []
        self.delta_x = 0
        self.delta_y = 0
        self.q_learning = QLearningAlgorithm(actions=['up', 'down', 'left', 'right'], discount=0.9, featureExtractor=featureExtractor, height=height, width=width, size=self.size)

    def collide(self, head):
        if head in self.snake_list[:-1] or self.x < 0 or self.x > self.width - self.size or self.y < 0 or self.y > self.height - self.size:
            return True
        return False

    def play(self):
        clock = pygame.time.Clock()
        blue = (0, 0, 255)
        red = (255, 0, 0)
        game_over = False
        total_score = 0
        scores = []
        rewards = []
        
        # Initialize plotting
        plt.ion()  # Enable interactive mode
        fig, ax = plt.subplots()
        ax.set_title('Real-time Score and Rewards')
        ax.set_xlabel('Games')
        ax.set_ylabel('Scores/Rewards')
        score_line, = ax.plot([], [], label='Score', color='blue')
        reward_line, = ax.plot([], [], label='Total Rewards', color='red')
        ax.legend()
        ax.set_xlim(0, 10)  # Initial x-axis limit
        ax.set_ylim(-10, 10)  # Initial y-axis limit
        plt.show()

        while not game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_over = True
            
            # AI decides the action
            state = State(self.x, self.y, self.snake_length, self.foodx, self.foody, self.snake_list)
            action, _, _ = self.q_learning.getAction(state)  # Unpack the three values

            if action == 'up':
                self.delta_x = 0
                self.delta_y = -self.size
            elif action == 'down':
                self.delta_x = 0
                self.delta_y = self.size
            elif action == 'left':
                self.delta_x = -self.size
                self.delta_y = 0
            elif action == 'right':
                self.delta_x = self.size
                self.delta_y = 0

            self.x += self.delta_x
            self.y += self.delta_y
            head = [self.x, self.y]
            self.snake_list.append(head)

            if self.collide(head):
                print("Collision! Game Over")
                game_over = True
                continue
            
            if self.x == self.foodx and self.y == self.foody:
                self.foodx = round(random.randrange(0, self.width - self.size) / self.size) * self.size
                self.foody = round(random.randrange(0, self.height - self.size) / self.size) * self.size
                self.snake_length += 1
                total_score += 1
                reward = 1  # Reward for eating food
            else:
                reward = -1  # Penalty for each move
            
            self.snake_list = self.snake_list[-self.snake_length:]  # Maintain snake length

            # Update Q-learning with the reward
            new_state = State(self.x, self.y, self.snake_length, self.foodx, self.foody, self.snake_list)
            self.q_learning.incorporateFeedback(state, action, reward, new_state)

            # Plotting scores and rewards
            scores.append(total_score)
            rewards.append(reward)

            # Update plot data
            score_line.set_xdata(range(len(scores)))
            score_line.set_ydata(scores)
            reward_line.set_xdata(range(len(rewards)))
            reward_line.set_ydata(rewards)
            ax.set_xlim(0, len(scores) + 1)
            ax.set_ylim(min(rewards) - 1, max(scores) + 1)
            plt.draw()
            plt.pause(0.001)  # Allow plot to update

        # Wait for a moment before quitting
        time.sleep(2)
        pygame.quit()
        plt.ioff()  # Disable interactive mode
        plt.show()  # Show final plot
        plt.savefig("snake_game_results.png")  # Save final plot to file

# Main function to initialize Pygame and start the game
def main():
    pygame.init()
    width = 600
    height = 600
    dis = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Snake with Q-learning")
    snake_game = Snake(dis, height, width)
    snake_game.play()

if __name__ == "__main__":
    main()
