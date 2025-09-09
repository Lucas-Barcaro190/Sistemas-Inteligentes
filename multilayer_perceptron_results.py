import pandas as pd
import matplotlib.pyplot as plt

def plot_mlp_results():
    """
    Reads the MLP training results and plots the accuracy evolution for each scenario.
    """
    try:
        df = pd.read_csv("multilayer_perceptron_results.csv")
    except FileNotFoundError:
        print("Error: multilayer_perceptron_results.csv not found.")
        print("Please run the C program first to generate the file.")
        return

    # Create a plot with a better style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 9))

    # Get the unique training scenarios
    scenarios = df['TrainingScenario'].unique()

    # Plot each scenario's accuracy curve
    for scenario in scenarios:
        # Filter the dataframe for the current scenario
        scenario_df = df[df['TrainingScenario'] == scenario]
        
        # Use a thicker line for the important 'Multi-Class' and 'Versicolor vs Virginica'
        linewidth = 4.0 if "Multi-Class" in scenario or "Versicolor vs Virginica" in scenario else 2.0
        linestyle = '--' if "vs Rest" in scenario else '-'
        
        ax.plot(scenario_df['Epoch'], scenario_df['Accuracy'], label=scenario, 
                linewidth=linewidth, linestyle=linestyle)

    # Add titles, labels, and a legend
    ax.set_title('MLP Training Accuracy Evolution on Iris Dataset', fontsize=18)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Test Accuracy (%)', fontsize=14)
    ax.legend(title='Training Scenario', fontsize=11)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(0, 105) # Set y-axis limits

    # Save the plot and show it
    plt.savefig("multilayer_perceptron_plot.png")
    print("Plot saved to multilayer_perceptron_plot.png")
    plt.show()

if __name__ == '__main__':
    plot_mlp_results()