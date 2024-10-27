import matplotlib.pyplot as plt

def plot(scores, mean_scores, rewards):
    plt.figure(figsize=(12, 6))

    # Create subplots
    plt.subplot(3, 1, 1)
    plt.plot(scores, label='Score', color='blue')
    plt.ylabel('Score')
    plt.title('Score over Games')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(mean_scores, label='Mean Score', color='orange')
    plt.ylabel('Mean Score')
    plt.title('Mean Score over Games')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(rewards, label='Rewards', color='green')
    plt.ylabel('Rewards')
    plt.title('Rewards over Games')
    plt.xlabel('Games')
    plt.legend()

    plt.tight_layout()
    plt.show()