---
D key: Enter/Exit Debug Mode
S key: Skip Trial
Space key: Next Frame
Up Arrow Key: Speed up frame by 100
Down Arrow Key: Slow down frame by 100
---

## Check out my Website
https://www.aidenwchang.com/


## How to run the code

Clone the github repository, install and activate python venv. 

Install a python virtual environment (Optional):
```
python3 -m venv [path to virtual environment]
```

To start virtual environment and install required packages:
```
cd [path to virtual environment]
source bin/activate
pip install -r requirements.txt
```


Run the code typing in:
```
python3 learn.py
```
or
```
python3 learn.py -d 
```
to start the code in debug mode.

Once the snake program is launched, you can adjust the frame rate of the snake with the up and down arrow keys.

### Debug Mode
If you press the 'D' key, you will enter debug mode. This allows you to see graphically what the weight values of each possible moves are. The red colored value will indicate the direction the snake chose for the next move.

Pressing the 'D' key one more time will exit debug mode.

Press the 'S' key to skip one trial. A trial is defined to be 10,000 frames or until the snake dies.

Press the space bar to allow the snake to move one frame. 

On the terminal, the current feature values will be displayed followed by a dashed line.
| Command | Description |
| --- | --- |
| foodLeft | Food is on the left side of snake |
| foodRight | Food is on the right side of the snake |
| foodDown | Food is on the down side of the snake |
| foodUp | Food is on the up side of the snake |
| dangerDown | the snake will die if it travels one block below |
| dangerUp | the snake will die if it travels one block above |
| dangerLeft | the snake will die if it travels one block to the left |
| dangerRight | the snake will die if it travels one block to the right |


## About the Project
The snake AI uses Reinforcement Learning to solve the most optimal way to play snake.

The AI uses Q algorithm to solve the problem. 

There are 4 actions that the AI can take. "up", "down", "left", or "right". Depending on if the body size > 1, this could be limited to 3 actions.

A "state" in this problem is defined to be: 
- x,y coordinates of the head of the snake
- Length of the snake
- x,y coordinates of the food
- The current snake body

Given that the food is randomly generated across the screen (besides the first iteration so the snake can learn quickly that eating the food gives a good reward). There are too many possible states to visit. Therefore instead of saving and trying to visit all possible states, approximate Q learning will be used. 

For actual rewards, two methodes were tested. 

1. Reward for every closer block to the fruit.
    - Reward for eating fruit: 10
    - Reward for being in a new state closer to the fruit compared to the previous state: 1
    - Reward for being in a new state further to the fruit compared to the previous state: -1
    - Reward for dying: -1000

The idea was to try to get the snake always moving towards the fruit, but it ended up determening that dying might be worth more than going around its long tail in later stages.

2. Reward for only the fruit.
    - Reward for eating fruit: 10
    - Reward for dying: -1000
    - Any other reward: 0

The idea was to try to get the snake to prioritize staying alive, giving it no negative feedback when it takes the long way around.

### Features
For each given state and action, the featureExtractor will generate a bitlist. There are 8 Features with the following 
| index | Description |
| --- | --- |
| 0 | (Bit is 1 if food is on the left side, 0 otherwise. , action)|
| 1 | (Bit is 1 if food is on the right side, 0 otherwise. , action)|
| 2 | (Bit is 1 if food is on the bottom side, 0 otherwise. , action)|
| 3 | (Bit is 1 if food is on the top side, 0 otherwise. , action)|
| 4 | (Bit is 1 if the snake will die if it travels one block below, action) |
| 5 | (Bit is 1 if the snake will die if it travels one block above, action) |
| 6 | (Bit is 1 if the snake will die if it travels one block to the left, action) |
| 7 | (Bit is 1 if the snake will die if it travels one block to the rightm, action) |

For Example, if the snake was in the top left corner without a tail and the food is in the middle of the screen, the result bitlist will be this:
- (0, 1, 1, 0, 0, 1, 1, 0)

### The algorithm:
With a probability of $\epsilon$ (default 0.2), the snake will take a random legal action. This value will decrease as the number of trials increase.

With a probability of $1-\epsilon$, the snake will choose its next move with the highest reward. This is given from the function getQ(s, a), where s is the current state and $f$ is the feature values.

Next Action = $max_a$($getQ(s,a)= w_1 * f_1 (s,a)+w_2 * f_2 (s,a)$+  ... $+ w_n*f_n(s,a) $) for a in legal actions

Now the actual learning part. To update the respective weights for these specific feature values, for a given weight $w_i$ which weights the importance of its respective feature $f_i$, we update $w_i$ to be:

$w_i = w_i + \alpha * difference * f_i(s,a)$

Where $\alpha$ is the step size, or learning rate. Our $\alpha$ is currently fixed to an integer < 1.

When we take this action a in state s we land in state s'. The difference variable mentioned when calculating $w_i$ is calculated by subtracting the total reward for being in state s' - what we predicted it would be. This can also be written as:
- [reward observed in s'+ discount value*max(future predicted rewards)] - predicted reward
- Mathematically this can be written as $difference$ = $[r(s') + \gamma * max_{a'}Q(s',a')] - Q(s,a)$
    - $\gamma$ = discount value


Issues resolved:
- The code has an $\epsilon$ value that acts as a probability of the snake to make a random move with the purpose of exploring new states. This causes an issue in later stages, so an $\epsilon$ value of each trial is determined to be $\epsilon$ = (max $\epsilon$)/trial. This decreases randomness as our trials increase
- The fruit spawned in the snake body causing the snake to crash into itself

Future Implementations:
- Adjust $\alpha$ on the fly.

