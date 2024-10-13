import pygame
import time
import random
import math
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
        self.foodx = round(random.randrange(0, self.width-self.size) / self.size) * self.size # 30 is snake block
        self.foody = round(random.randrange(0, self.width-self.size) / self.size) * self.size
        self.snake_list = []
        self.delta_x = 0
        self.delta_y = 0

    
    def collide(self, head):
        #add tail check later
        if head in self.snake_list[:-1]:
            return True

        if self.x < 0 or self.x > self.width-self.size or self.y<0 or self.y > self.height-self.size:
            # print("crashing")
            return True 
        
        return False

    def new_location(self):
        # call rl methode here
        return -1, 0
    
    def collide(self, head):
        #add tail check later
        if head in self.snake_list[:-1]:
            return True

        if self.x < 0 or self.x > self.width-self.size or self.y<0 or self.y > self.height-self.size:
            # print("crashing")
            return True 
        
        return False
    

    
    def choices(self):
        choices_list = []
        moves = [[0,-self.size], [0,self.size], [-self.size,0], [self.size,0]] #up, down left, right

        for move in moves:
            new_state = State(move[0], move[1], 0)
            temp_x = self.x + new_state.delta_x
            temp_y = self.y + new_state.delta_y
            if temp_x == self.foodx and temp_y == self.foody:
                new_state.reward = 1
            if temp_x < 0 or temp_x > self.width-self.size or temp_y<0 or temp_y > self.height-self.size or [temp_x, temp_y] in self.snake_list[:-1]:
                new_state.reward = -1
            choices_list.append(new_state)

        random.shuffle(choices_list)
        return choices_list                
    

    def ai_decide(self):
        choices_list = self.choices()
        max = -2
        chosen = None
        for choice in choices_list:
            if choice.reward > max:
                chosen = choice
                max = choice.reward
        return chosen


    def play(self):
        clock = pygame.time.Clock()
        blue=(0,0,255)
        red=(255,0,0)
        game_over=False
        while not game_over:
            if CONTROLS:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        game_over = True
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_LEFT:
                            self.delta_x = -self.size
                            self.delta_y  = 0
                        elif event.key == pygame.K_RIGHT:
                            self.delta_x = self.size
                            self.delta_y  = 0
                        elif event.key == pygame.K_UP:
                            self.delta_y  = -self.size
                            self.delta_x = 0
                        elif event.key == pygame.K_DOWN:
                            self.delta_y  = self.size
                            self.delta_x = 0
            else: #algorithm determines
                chosen = self.ai_decide()
                self.delta_x, self.delta_y = chosen.delta_x, chosen.delta_y
            self.x += self.delta_x
            self.y += self.delta_y    

            snake_Head = []
            snake_Head.append(self.x)
            snake_Head.append(self.y)
            self.snake_list.append(snake_Head)
            if len(self.snake_list) > self.snake_length:
                del self.snake_list[0]
            
            if self.collide(snake_Head):
                break



            self.dis.fill("black")
            for x,y in self.snake_list:
                pygame.draw.rect(self.dis,blue,[x,y,self.size,self.size])
            pygame.draw.rect(self.dis,red,[self.foodx,self.foody,self.size,self.size])
            pygame.display.update()
            if self.x == self.foodx and self.y == self.foody:
                self.snake_length +=1
                self.foodx = round(random.randrange(0, self.width-self.size) / self.size) * self.size
                self.foody = round(random.randrange(0, self.width-self.size) / self.size) * self.size
            clock.tick(600)


        return self.snake_length - 1


def featureExtractor(self, state: State, action):
    features = []
    temp_x = self.x + state.delta_x
    temp_y = self.y + state.delta_y

    #distance to left wall, how big the tail is, action maybe you need location?
    features.append(Feature(featureKey=('leftWall', temp_x, action), featureValue=1))

    #distance to right wall, how big the tail is, action
    features.append(Feature(featureKey=('rightWall', self.width-self.size-temp_x, action), featureValue=1))

    #distance to top wall, how big the tail is, action
    features.append(Feature(featureKey=('topWall', temp_y, action), featureValue=1))

    #distance to bottom wall, how big the tail is, action
    features.append(Feature(featureKey=('bottomWall', self.height-self.size-temp_y, action), featureValue=1))

    #maybe how close the head is to the body?

    #distance to fruit, action
    features.append(Feature(featureKey=('bottomWall', abs(temp_y-self.foody) + abs(temp_x-self.foodx), action), featureValue=1))

    return features




def learn(dis, height, width):
    snake = Snake(dis, height, width)
    score = snake.play()
    return score



def start_display():
    height = 600
    width = 600
    pygame.init()
    dis=pygame.display.set_mode((height,width))
    pygame.display.set_caption('Reinforcement Learning Snake')
    

    print(learn(dis, height, width))
    pygame.quit()
    quit()

start_display()