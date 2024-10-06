# Snake
Train an AI to Play Snake using Python + PyTorch + Pygame Reinforcement Learning

## model.py
Use PyTorch, Linear_QNet (DQN) to predict the Action

## Training:
    State = get_state(game)
    Action = get_move(state):
    Reward, game_over, score = game.play_step(action)
    New_state = get_state(game)
    Remember
    Model.train()

## Reward
    Eat food: +10
    Game over: -10
    Else: 0

## Action
    [1, 0, 0]: Straight
    [0, 1, 0]: right turn
    [0, 0, 1]: left turn

## State: tell the snake information about the game that it knows about the environment
Have 11 values:
    Danger straight, danger right, danger left
    Direction left, direction right, direction up, direction down
    Food left, food right, food up, food down

## Deep Q Learning
    1. Init Q value
    2. Choose action
    3. Measeure reward
    4. Update Q value

## Bellman Equation
            NewQ(s, a) = Q(s, a) +a[R(s,a) + ymaxQ' (s',a' )-Q(s,a)]

when:
    - NewQ(s,a): new Q value for that state and that action
    - Q(s,a): current Q value
    - a: learning rate
    - R(s,a): reward for taking that action at that state
    - y: discount rate
    - maxQ'(s',a'): max expected future reward given the new s' and all possible actions at that new state

## Q Update rule simplified
        Q = model.predict(state_0)
        Q_new = R+y*max(Q(state_1 ))

## Loss function
        loss=(Q_new-Q)^2

## To run program:
        Conda create -n snake_game python=3.10
        Conda activate snake_game

        python agent.py
