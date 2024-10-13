import pygame
import random
from util import State

class Snake:
    def __init__(self, dis, height, width, size, debug_mode) -> None:
        self.size = size
        self.dis = dis
        self.height = height
        self.width = width
        self.debug_mode = debug_mode
        # Define colors
        self.blue = (0, 0, 255)
        self.red = (255, 0, 0)
        self.white = (255, 255, 255)

    def reset(self) -> State:
        # Set initial position of the food
        foodx, foody = self._generate_food()
        self.dis.fill("black")
        state = State(300, 300, 1, foodx, foody, [])
        return state

    def _generate_food(self):
        # Generate food position ensuring it's not on the snake
        foodx = round(random.randrange(0, self.width - self.size) / self.size) * self.size 
        foody = round(random.randrange(0, self.height - self.size) / self.size) * self.size
        return foodx, foody

    def collide(self, head, state):
        # Check collision with the snake body
        if head in state.snake_list[:-1]:
            if self.debug_mode:
                print("Hit snake body")
            return True

        # Check wall collision
        if state.x < 0 or state.x >= self.width - self.size or state.y < 0 or state.y >= self.height - self.size:
            if self.debug_mode:
                print("Crashed into wall")
            return True 
        
        return False
    
    def reward_distance_to_fruit(self, state, new_state):
        # Calculate the distance reward
        new_distance = abs(new_state.x - new_state.foodx) + abs(new_state.y - new_state.foody)
        old_distance = abs(state.x - state.foodx) + abs(state.y - state.foody)
        
        if new_distance < old_distance:
            return 1  # Closer to food
        elif new_distance > old_distance:
            return -1  # Further from food
        
        return 0  # Same distance

    def choices(self, state, action):
        moves = {"up": [0, -self.size], "down": [0, self.size], "left": [-self.size, 0], "right": [self.size, 0]}  # Up, down, left, right
        if action not in moves:
            print("Invalid action")
            exit(0)
        
        move = moves[action]
        temp_x = state.x + move[0]
        temp_y = state.y + move[1]
        
        new_state = State(temp_x, temp_y, state.length, state.foodx, state.foody, state.snake_list.copy())
        reward = self.reward_distance_to_fruit(state, new_state)
        
        if temp_x == state.foodx and temp_y == state.foody:
            reward = 10
            new_state.length += 1
            new_state.foodx, new_state.foody = self._generate_food()
            # Regenerate food if it is on the snake
            while [new_state.foodx, new_state.foody] in new_state.snake_list:
                new_state.foodx, new_state.foody = self._generate_food()

        # Check for collisions with walls or snake body
        if (temp_x < 0 or temp_x >= self.width or temp_y < 0 or temp_y >= self.height or
                [temp_x, temp_y] in new_state.snake_list[:-1]):
            reward = -1000  # Heavy penalty for collision

        return new_state, reward              

    def draw(self, state, trial):
        score_font = pygame.font.SysFont("comicsansms", 35)
        mesg = score_font.render(f"Trial: {trial}", True, self.white)

        snake_head = [state.x, state.y]
        state.snake_list.append(snake_head)
        if len(state.snake_list) > state.length:
            del state.snake_list[0]

        self.dis.fill("black")
        self.dis.blit(mesg, [0, 0])
        
        for x, y in state.snake_list:
            pygame.draw.rect(self.dis, self.blue, [x, y, self.size, self.size])
        
        pygame.draw.rect(self.dis, self.red, [state.foodx, state.foody, self.size, self.size])
        pygame.display.update()
