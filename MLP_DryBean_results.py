import pandas as pd
import matplotlib.pyplot as plt

def plot_accuracy_evolution():
    """
    Reads the MLP accuracy history from a CSV and plots the learning curve.
    """
    try:
        df = pd.read_csv("mlp_accuracy_beans.csv")
    except FileNotFoundError:
        print("Error: mlp_accuracy_beans.csv not found.")
        print("Please run the C program first to generate the file.")
        return

    # Create the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(df['Epoch'], df['Accuracy'], label='Validation Accuracy', color='dodgerblue', lw=2.5)

    # Find and mark the best accuracy
    best_epoch = df['Accuracy'].idxmax()
    best_score = df['Accuracy'].max()
    ax.axvline(x=df['Epoch'][best_epoch], color='red', linestyle=':', 
                label=f'Best Accuracy: {best_score:.2f}% at Epoch {df["Epoch"][best_epoch]}')

    # Add titles and labels
    ax.set_title('MLP Validation Accuracy on Dry Bean Dataset', fontsize=16)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True)
    plt.ylim(0, 105) # Set y-axis limits

    # Save the plot and show it
    plt.savefig("mlp_accuracy_plot_beans.png")
    print("Accuracy plot saved to mlp_accuracy_plot_beans.png")
    plt.show()

if __name__ == '__main__':
    plot_accuracy_evolution()