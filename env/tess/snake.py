import pygame
import random
from util import State, Feature

CONTROLS = False

class Snake:
    def __init__(self, dis, height, width) -> None:
        self.size = 30
        self.dis = dis
        self.height = height
        self.width = width
        self.x = 300
        self.y = 300
        self.snake_length = 1
        self.foodx = self._generate_food_position()[0]
        self.foody = self._generate_food_position()[1]
        self.snake_list = []
        self.delta_x = 0
        self.delta_y = 0

    def _generate_food_position(self):
        # Generate food position ensuring it does not spawn on the snake
        foodx = round(random.randrange(0, self.width - self.size) / self.size) * self.size
        foody = round(random.randrange(0, self.height - self.size) / self.size) * self.size
        return foodx, foody

    def collide(self, head):
        # Check collision with snake body
        if head in self.snake_list[:-1]:
            return True
        # Check wall collision
        if self.x < 0 or self.x >= self.width - self.size or self.y < 0 or self.y >= self.height - self.size:
            return True 
        return False

    def choices(self):
        choices_list = []
        moves = [[0, -self.size], [0, self.size], [-self.size, 0], [self.size, 0]]  # Up, down, left, right

        for move in moves:
            new_state = State(self.x + move[0], self.y + move[1], 0)  # State should reflect new position
            # Check for food
            if new_state.x == self.foodx and new_state.y == self.foody:
                new_state.reward = 1
            # Check for collisions
            elif (new_state.x < 0 or new_state.x >= self.width - self.size or 
                  new_state.y < 0 or new_state.y >= self.height - self.size or 
                  [new_state.x, new_state.y] in self.snake_list[:-1]):
                new_state.reward = -1
            else:
                new_state.reward = 0  # Neutral reward
            choices_list.append(new_state)

        random.shuffle(choices_list)
        return choices_list                

    def ai_decide(self):
        choices_list = self.choices()
        max_reward = -2
        chosen = None
        for choice in choices_list:
            if choice.reward > max_reward:
                chosen = choice
                max_reward = choice.reward
        return chosen

    def play(self):
        clock = pygame.time.Clock()
        blue = (0, 0, 255)
        red = (255, 0, 0)
        game_over = False

        while not game_over:
            if CONTROLS:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        game_over = True
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_LEFT:
                            self.delta_x = -self.size
                            self.delta_y = 0
                        elif event.key == pygame.K_RIGHT:
                            self.delta_x = self.size
                            self.delta_y = 0
                        elif event.key == pygame.K_UP:
                            self.delta_y = -self.size
                            self.delta_x = 0
                        elif event.key == pygame.K_DOWN:
                            self.delta_y = self.size
                            self.delta_x = 0
            else:  # Algorithm determines
                chosen = self.ai_decide()
                self.delta_x, self.delta_y = chosen.delta_x, chosen.delta_y

            self.x += self.delta_x
            self.y += self.delta_y    

            snake_head = [self.x, self.y]
            self.snake_list.append(snake_head)
            if len(self.snake_list) > self.snake_length:
                del self.snake_list[0]
            
            if self.collide(snake_head):
                break

            self.dis.fill("black")
            for x, y in self.snake_list:
                pygame.draw.rect(self.dis, blue, [x, y, self.size, self.size])
            pygame.draw.rect(self.dis, red, [self.foodx, self.foody, self.size, self.size])
            pygame.display.update()

            # Check if the snake has eaten the food
            if self.x == self.foodx and self.y == self.foody:
                self.snake_length += 1
                self.foodx, self.foody = self._generate_food_position()
                # Ensure food doesn't spawn on the snake
                while [self.foodx, self.foody] in self.snake_list:
                    self.foodx, self.foody = self._generate_food_position()
            
            clock.tick(15)  # Adjusted game speed for better playability

        return self.snake_length - 1

    def feature_extractor(self, action):
        features = []
        temp_x = self.x + self.delta_x
        temp_y = self.y + self.delta_y

        # Distance to left wall
        features.append(Feature(featureKey=('leftWall', temp_x, action), featureValue=1))
        # Distance to right wall
        features.append(Feature(featureKey=('rightWall', self.width - self.size - temp_x, action), featureValue=1))
        # Distance to top wall
        features.append(Feature(featureKey=('topWall', temp_y, action), featureValue=1))
        # Distance to bottom wall
        features.append(Feature(featureKey=('bottomWall', self.height - self.size - temp_y, action), featureValue=1))
        # Distance to fruit
        features.append(Feature(featureKey=('fruitDistance', abs(temp_y - self.foody) + abs(temp_x - self.foodx), action), featureValue=1))

        return features

def learn(dis, height, width):
    snake = Snake(dis, height, width)
    score = snake.play()
    return score

def start_display():
    height = 600
    width = 600
    pygame.init()
    dis = pygame.display.set_mode((height, width))
    pygame.display.set_caption('Reinforcement Learning Snake')
    
    print(learn(dis, height, width))
    pygame.quit()
    quit()

start_display()
