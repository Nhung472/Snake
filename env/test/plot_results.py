import matplotlib.pyplot as plt
import csv

def plot_scores_and_rewards(csv_file):
    scores = []
    rewards = []

    try:
        with open(csv_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                scores.append(int(row['Score']))
                rewards.append(float(row['Total Reward']))
    except FileNotFoundError:
        print(f"Error: The file {csv_file} was not found.")
        return

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(scores, label='Scores', color='blue')
    plt.plot(rewards, label='Total Rewards', color='orange')
    plt.title('Scores and Total Rewards Over Games')
    plt.xlabel('Game Number')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.show()