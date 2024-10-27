import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, rewards):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score and Reward')
    plt.plot(scores)
    plt.plot(rewards)
    plt.ylim(ymin=-20)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(rewards)-1, rewards[-1], str(rewards[-1]))
    plt.show(block=False)
    plt.pause(.1)