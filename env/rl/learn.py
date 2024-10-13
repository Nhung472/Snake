from learning_snake import Snake
from util import QLearningAlgorithm, featureExtractor, State
import pygame
import argparse




def simulate(display, height, width, debug_mode, numTrials=300, maxIterations=10000):
    actions = ["up","down","left","right"]
    discount = 1 # what is this
    size = 30
    qlearn = QLearningAlgorithm(actions, discount, featureExtractor, height, width, size)
    snake = Snake(display, height, width, size, debug_mode)
    total_rewards = []
    clock = pygame.time.Clock()
    timer = 20

    for trial in range(numTrials):
        state = snake.reset()
        totalDiscount = 1 #future
        trial_reward = 0
        skip = False
        for _ in range(maxIterations):
            action, debug_list, debug_bits = qlearn.getAction(state, debug_mode, trial+1)


            if debug_mode and debug_bits:
                debug_bit_list = ['foodLeft','foodRight','foodDown','foodUp','dangerUp','dangerDown','dangerLeft','dangerRight']
                for i in range(len(debug_bits)):
                    if debug_bits[i] != 0:
                        print(debug_bit_list[i])
                print("="*20)
            #For debugging purposes
            if debug_mode and not skip:
                debug_qValues = []
                # draw = [[0,-size], [0,size], [-size,0],[size,0]]
                draw ={"up":[0,-size], "down":[0,size], "left":[-size,0], "right":[size,0]}
                x, y = state.x, state.y
                font = pygame.font.SysFont('arial', 15)
                for v, v_action in debug_list:
                    chosen_action = draw[v_action]
                    text = font.render(str(round(v, 2)), True, (255, 255, 255))
                    if v_action == action:
                        text = font.render(str(round(v, 2)), True, (255, 0, 0))
                    display.blit(text, [x+chosen_action[0]+size/2, y+chosen_action[1]+size/2])
                pygame.display.update()
                out = False
                while not out:
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            if event.key == 32:
                                out = True
                            if event.key == 115:
                                skip = True
                                out = True
                            if event.key == 100:
                                debug_mode = False
                                snake.debug_mode = False
                                out = True
                            

            new_state, reward = snake.choices(state, action)
            if reward < -10:
                qlearn.incorporateFeedback(state, action, reward, new_state)
                break
            
            qlearn.incorporateFeedback(state, action, reward, new_state)
            trial_reward += totalDiscount * reward
            totalDiscount *= discount
            state = new_state
            snake.draw(new_state, trial)
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