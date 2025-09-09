import pandas as pd
import matplotlib.pyplot as plt

try:
    df = pd.read_csv("overfitting_results.csv")
except FileNotFoundError:
    print("Error: overfitting_results.csv not found. Run the C tracker program first.")
    exit()

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 8))

# Plot both training and validation accuracy
ax.plot(df['Epoch'], df['TrainingAccuracy'], label='Training Accuracy', color='blue', lw=2.5)
ax.plot(df['Epoch'], df['ValidationAccuracy'], label='Validation Accuracy', color='red', linestyle='--', lw=2.5)

# Find the best validation score to mark on the plot
best_epoch = df['ValidationAccuracy'].idxmax()
best_score = df['ValidationAccuracy'].max()
ax.axvline(x=df['Epoch'][best_epoch], color='green', linestyle=':', label=f'Best Validation Score: {best_score:.2f}% at Epoch {df["Epoch"][best_epoch]}')

ax.set_title('Training vs. Validation Accuracy Over Time', fontsize=16)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True)
plt.ylim(40, 105)

plt.savefig("overfitting_plot.png")
print("Overfitting analysis plot saved to overfitting_plot.png")
plt.show()