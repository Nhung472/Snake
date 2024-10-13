from learning_snake import Snake
from util import QLearningAlgorithm, featureExtractor, State
import pygame
import argparse

def simulate(display, height, width, debug_mode, numTrials=300, maxIterations=10000):
    actions = ["up", "down", "left", "right"]
    discount = 1  # Future reward discount factor
    size = 30
    qlearn = QLearningAlgorithm(actions, discount, featureExtractor, height, width, size)
    snake = Snake(display, height, width, size, debug_mode)
    total_rewards = []
    clock = pygame.time.Clock()
    timer = 20

    for trial in range(numTrials):
        state = snake.reset()
        totalDiscount = 1  # Initialize for future rewards
        trial_reward = 0
        skip = False
        
        for _ in range(maxIterations):
            action, debug_list, debug_bits = qlearn.getAction(state, debug_mode, trial + 1)

            if debug_mode and debug_bits:
                debug_bit_list = ['foodLeft', 'foodRight', 'foodDown', 'foodUp', 'dangerUp', 'dangerDown', 'dangerLeft', 'dangerRight']
                for i in range(len(debug_bits)):
                    if debug_bits[i] != 0:
                        print(debug_bit_list[i])
                print("=" * 20)

            # Debugging information display
            if debug_mode and not skip:
                debug_qValues = []
                draw = {"up": [0, -size], "down": [0, size], "left": [-size, 0], "right": [size, 0]}
                x, y = state.x, state.y
                font = pygame.font.SysFont('arial', 15)

                for v, v_action in debug_list:
                    chosen_action = draw[v_action]
                    text = font.render(str(round(v, 2)), True, (255, 255, 255))
                    if v_action == action:
                        text = font.render(str(round(v, 2)), True, (255, 0, 0))
                    display.blit(text, [x + chosen_action[0] + size / 2, y + chosen_action[1] + size / 2])
                pygame.display.update()

                # Event handling for debugging
                out = False
                while not out:
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:  # Space key
                                out = True
                            elif event.key == pygame.K_s:  # S key to skip
                                skip = True
                                out = True
                            elif event.key == pygame.K_d:  # D key to toggle debug mode
                                debug_mode = False
                                snake.debug_mode = False
                                out = True

            new_state, reward = snake.choices(state, action)
            
            # If the snake receives a heavy penalty, break the loop
            if reward < -10:
                qlearn.incorporateFeedback(state, action, reward, new_state)
                break
            
            # Incorporate feedback into the Q-learning algorithm
            qlearn.incorporateFeedback(state, action, reward, new_state)
            trial_reward += totalDiscount * reward
            totalDiscount *= discount  # Update total discount for future rewards
            state = new_state
            snake.draw(new_state, trial)

            # Event handling for control keys
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:  # Increase timer
                        timer += 100
                    elif event.key == pygame.K_DOWN:  # Decrease timer
                        timer -= 100
                    elif event.key == pygame.K_d:  # Toggle debug mode
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
    dis = pygame.display.set_mode((width, height))  # Corrected dimensions to (width, height)
    pygame.display.set_caption('Reinforcement Learning Snake')
    debug_mode = args.d  # Set debug_mode based on argument
    total_reward = simulate(dis, height, width, debug_mode)
    print(total_reward)
    pygame.quit()
    quit()

learn()
