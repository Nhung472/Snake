from typing import List, Tuple, Dict, Any, Optional, NamedTuple, Callable
from typing import List, Callable, Tuple, Any
from collections import defaultdict
import random
import math


Feature = NamedTuple('Feature', [('featureKey', Tuple), ('featureValue', int)])

class State:
    def __init__(self, x, y, length, foodx, foody, snake_list) -> None:
        self.x = x
        self.y = y
        self.length = length
        self.foodx = foodx
        self.foody = foody
        self.snake_list = snake_list


############################################################
# Performs Q-learning. 
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm():
    def __init__(self, actions: List, discount: float, featureExtractor: Callable,  height, width, size, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0
        self.height = height
        self.width = width
        self.size = size

    # Return the Q function associated with the weights and features
    def getQ(self, state: State, action: Any) -> float:
        score = 0
        featureValues, debug_values = self.featureExtractor(state, action, self.height, self.width, self.size)
        for f, v in featureValues:
            score += self.weights[f] * v
        return score, debug_values

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state: State, debug, iterations) -> Any:
        self.numIters += 1
        moves = {"up":[0,-self.size], "down":[0,self.size], "left":[-self.size,0], "right":[self.size,0]}
        actions = []
        debug_list = []
        explorationProb = self.explorationProb / iterations #decreasing randomness as we go
        for key in moves:
            delta_vals = moves[key]
            # if state.snake_list:
                # print([state.x + delta_vals[0],state.y + delta_vals[1]], state.snake_list)
            if len(state.snake_list) > 1 and [state.x + delta_vals[0],state.y + delta_vals[1]] == state.snake_list[-2]:
                continue
            actions.append(key)
        # print(actions)
        chosen_action = None
        debug_bit = None
        if random.random() < explorationProb:
            chosen_action = random.choice(actions)
        else:
            # random.shuffle(actions)
            if debug:
                debug_list = [(self.getQ(state, action)[0], action) for action in actions]
            chosen_action_list = max((self.getQ(state, action)[0], action, self.getQ(state, action)[1]) for action in actions)
            debug_bit = chosen_action_list[2]
            chosen_action = chosen_action_list[1]
        return chosen_action, debug_list, debug_bit

    # Call this function to get the step size to update the weights.
    def getStepSize(self) -> float:
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state: State, action: Any, reward: int, newState: State) -> None:

        # if newState == None:
        #     # s is terminal
        #     return

        featureValues, debug_values = self.featureExtractor(state, action, self.height, self.width, self.size)
        max_action_value = float('-inf')
        for new_action in self.actions:
            temp_value, debug_values = self.getQ(newState, new_action)
            if temp_value > max_action_value:
                max_action_value = temp_value

        value, debug_values = self.getQ(state, action)
        difference = reward + self.discount*max_action_value - value

        alpha = self.getStepSize()
        for f, v in featureValues:
            self.weights[f] += alpha*difference*v


def featureExtractor(state: State, action, height, width, size):
    features = []
    state_features = False
    direction_and_danger = True
    wall_tracker = True
    '''
    Some ideas
    if the snake is in the wall, put it at 0? Since it doesnt want to crash
    '''

    bit = None
    debug_bit = []
    if state_features:
        #if snake is right of food
        if state.x > state.foodx:
            bit = 1
        else:
            bit = 0
        features.append(Feature(featureKey=('foodRight', action), featureValue=bit))
        debug_bit.append(bit)
        #if snake is left of food
        if state.x < state.foodx:
            bit = 1
        else:
            bit = 0
        features.append(Feature(featureKey=('foodLeft', action), featureValue=bit))
        debug_bit.append(bit)
        #if snake is up of food
        if state.y < state.foody:
            bit = 1
        else:
            bit = 0
        features.append(Feature(featureKey=('foodUp', action), featureValue=bit))
        debug_bit.append(bit)
        #if snake is down of food
        if state.y > state.foody:
            bit = 1
        else:
            bit = 0
        features.append(Feature(featureKey=('foodDown', action), featureValue=bit))
        debug_bit.append(bit)
        

        #danger up
        if state.y == 0 or [state.x, state.y - size] in state.snake_list[:-1]:
            bit = -1
        else:
            bit = 0
        features.append(Feature(featureKey=('dangerUp', action), featureValue=bit))
        debug_bit.append(bit)
        #danger down
        if state.y == height-size or [state.x, state.y + size] in state.snake_list[:-1]:
            bit = -1
        else:
            bit = 0
        features.append(Feature(featureKey=('dangerDown', action), featureValue=bit))
        debug_bit.append(bit)
        #danger left
        if state.x == 0 or [state.x - size, state.y] in state.snake_list[:-1]:
            bit = -1
        else:
            bit = 0
        features.append(Feature(featureKey=('dangerLeft', action), featureValue=bit))
        debug_bit.append(bit)
        #danger right
        if state.x == width-size or [state.x + size, state.y] in state.snake_list[:-1]:
            bit = -1
        else:
            bit = 0
        features.append(Feature(featureKey=('dangerRight', action), featureValue=bit))
        debug_bit.append(bit)
    

    if direction_and_danger:
        #if snake is right of food
        if state.x > state.foodx:
            bit = 1
        else:
            bit = 0
        features.append(Feature(featureKey=('foodRight', action), featureValue=bit))
        debug_bit.append(bit)
        #if snake is left of food
        if state.x < state.foodx:
            bit = 1
        else:
            bit = 0
        features.append(Feature(featureKey=('foodLeft', action), featureValue=bit))
        debug_bit.append(bit)
        #if snake is up of food
        if state.y < state.foody:
            bit = 1
        else:
            bit = 0
        features.append(Feature(featureKey=('foodUp', action), featureValue=bit))
        debug_bit.append(bit)
        #if snake is down of food
        if state.y > state.foody:
            bit = 1
        else:
            bit = 0
        features.append(Feature(featureKey=('foodDown', action), featureValue=bit))
        debug_bit.append(bit)
        

        #danger up
        if (state.y == 0 or [state.x, state.y - size] in state.snake_list[:-1]) and action == "up":
            bit = -1
        else:
            bit = 0
        features.append(Feature(featureKey=('dangerUp', action), featureValue=bit))
        debug_bit.append(bit)
        #danger down
        if (state.y == height-size or [state.x, state.y + size] in state.snake_list[:-1]) and action == "down":
            bit = -1
        else:
            bit = 0
        features.append(Feature(featureKey=('dangerDown', action), featureValue=bit))
        debug_bit.append(bit)
        #danger left
        if (state.x == 0 or [state.x - size, state.y] in state.snake_list[:-1]) and action == "left":
            bit = -1
        else:
            bit = 0
        features.append(Feature(featureKey=('dangerLeft', action), featureValue=bit))
        debug_bit.append(bit)
        #danger right
        if (state.x == width-size or [state.x + size, state.y] in state.snake_list[:-1]) and action == "right":
            bit = -1
        else:
            bit = 0
        features.append(Feature(featureKey=('dangerRight', action), featureValue=bit))
        debug_bit.append(bit)
    

    # if wall_tracker:



    return features, debug_bit
