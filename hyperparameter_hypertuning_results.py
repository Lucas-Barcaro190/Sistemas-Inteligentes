import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_advanced_results():
    try:
        df = pd.read_csv("advanced_hyperparameter_results.csv")
    except FileNotFoundError:
        print("Error: advanced_hyperparameter_results.csv not found.")
        print("Please run the C tuner program first to generate the file.")
        return

    # Get the list of unique split ratios tested
    split_ratios = sorted(df['SplitRatio'].unique(), reverse=True)
    num_ratios = len(split_ratios)

    # Create a subplot for each split ratio
    fig, axes = plt.subplots(1, num_ratios, figsize=(num_ratios * 6, 5), sharey=True)
    fig.suptitle('Perceptron Performance by Train/Test Split Ratio', fontsize=16)

    # Find the global min and max accuracy for a consistent color scale
    vmin = df['Accuracy'].min()
    vmax = df['Accuracy'].max()

    for i, ratio in enumerate(split_ratios):
        ax = axes[i]
        
        # Filter data for the current ratio
        subset = df[df['SplitRatio'] == ratio]
        
        pivot = subset.pivot_table(
            values='Accuracy', 
            index='LearningRate', 
            columns='Epochs', 
            aggfunc='mean'
        )
        
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="viridis", ax=ax, vmin=vmin, vmax=vmax, cbar=False)
        ax.set_title(f'{int(ratio*100)}% / {100-int(ratio*100)}% Split')
        ax.set_ylabel('') # Hide y-label on all but the first plot

    axes[0].set_ylabel('Learning Rate')

    # Add a single, shared color bar
    fig.colorbar(axes[-1].collections[0], ax=axes, orientation='vertical', fraction=.1, pad=0.02, label='Average Accuracy (%)')

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make room for suptitle
    plt.savefig("advanced_hyperparameter_plot.png")
    print("Advanced hyperparameter plot saved to advanced_hyperparameter_plot.png")
    plt.show()

if __name__ == '__main__':
    visualize_advanced_results()