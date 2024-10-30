import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, rewards, mean_rewards):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    
    # Scores plot
    plt.subplot(2, 1, 1)
    plt.title('Training Progress')
    plt.plot(scores, label="Scores", color='blue')
    plt.xlabel('Games')
    plt.ylabel('Score')
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]), ha='center')

    # Rewards and Mean Rewards plot
    plt.subplot(2, 1, 2)
    plt.plot(rewards, label="Total Rewards", color='red')
    plt.plot(mean_rewards, label="Mean Rewards", color='green')
    plt.xlabel('Games')
    plt.ylabel('Rewards')
    plt.ylim(ymin=-30)
    plt.text(len(rewards)-1, rewards[-1], str(rewards[-1]), ha='center')
    plt.text(len(mean_rewards)-1, mean_rewards[-1], str(mean_rewards[-1]), ha='center')
    plt.legend()

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(.1)
