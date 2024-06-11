import json
import matplotlib.pyplot as plt

#The plot_evalution funtion creates a chart of visual representation to validate the trained model based on accuracy metrics
def plot_evaluation(train_losses, train_hamming_scores, val_losses, val_hamming_scores, precision, recall, f_score):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot Losses
    axes[0].plot(train_losses, label='Train Losses', marker='o')
    axes[0].plot(val_losses, label='Validation Losses', marker='o')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Train and Validation Losses vs. Epochs')
    axes[0].legend()
    axes[0].grid(True)

    # Plot Hamming Scores
    axes[1].plot(train_hamming_scores, label='Train Hamming Scores', marker='o')
    axes[1].plot(val_hamming_scores, label='Validation Hamming Scores', marker='o')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Hamming Score')
    axes[1].set_title('Train and Validation Hamming Scores')
    axes[1].legend()
    axes[1].grid(True)

    # Plot Precision, Recall, F-Score
    metrics = ['Precision', 'Recall', 'F-Score']
    values = [precision, recall, f_score]
    axes[2].bar(metrics, values, color=['blue', 'green', 'red'])
    axes[2].set_title('Score')
    axes[2].set_ylabel('Score')
    axes[2].grid(axis='y')
    axes[2].set_ylim(0, 1) 

    


with open('data.json', 'r') as file:
    data = json.load(file)
    
name_models = list(data.keys())

for name_ in name_models:
    parts = name_.split('_')
    batch_size = parts[1]
    learning_rate = parts[2]
    num_features = parts[3]
    dropout = parts[4].replace('.pth','')
    print(f'batch Size = {batch_size} | learning Rate = {learning_rate} | num_features = {num_features} | drop out = {dropout}')
    train_losses, train_hamming_scores, val_losses, val_hamming_scores = data[name_][0]
    precision, recall, f_score = data[name_][1]
    plot_evaluation(train_losses, train_hamming_scores, val_losses, val_hamming_scores, precision, recall, f_score)
    plt.savefig(f'{name_}_evaluation.png')