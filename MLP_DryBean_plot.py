import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from tqdm import tqdm

def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))

def parse_weights_file(filepath):
    # ... (Same parse_weights_file function as before) ...

def predict_mlp_vectorized(all_points, epoch_weights):
    # ... (Same vectorized predict function as before) ...

def generate_bean_frames():
    if not os.path.exists('frames_beans'):
        os.makedirs('frames_beans')

    print("Parsing bean weights file...")
    epochs_data = parse_weights_file('mlp_weights_beans.txt')
    if not epochs_data:
        print("Error: mlp_weights_beans.txt not found. Run the C program first.")
        return

    print("Loading Dry Bean data...")
    # Load the full dataset to get original (unnormalized) values for plotting
    df = pd.read_csv('dry_bean_dataset.csv')
    
    # These are the 4 features we will visualize
    features_to_plot = ['Perimeter', 'MajorAxisLength', 'ShapeFactor1', 'roundness']
    feature_indices = {name: df.columns.get_loc(name) for name in features_to_plot}
    
    # Normalize the entire dataset for prediction logic
    X_full = df.drop('Class', axis=1).values
    X_norm = (X_full - X_full.min(axis=0)) / (X_full.max(axis=0) - X_full.min(axis=0))
    feature_means_norm = X_norm.mean(axis=0)
    
    class_map = {name: i for i, name in enumerate(df['Class'].unique())}
    y = df['Class'].map(class_map).values

    print(f"Generating {len(epochs_data)} frames...")
    for epoch, weights in tqdm(epochs_data.items(), desc="Generating Frames"):
        fig, axes = plt.subplots(4, 4, figsize=(15, 15))
        fig.suptitle(f'MLP Decision Boundary Slices (Dry Bean) - Epoch: {epoch}', fontsize=20)

        for i, y_feature in enumerate(features_to_plot):
            for j, x_feature in enumerate(features_to_plot):
                ax = axes[i, j]
                x_idx, y_idx = feature_indices[x_feature], feature_indices[y_feature]
                
                if i == j: # Diagonal plots
                    sns.kdeplot(data=df, x=x_feature, hue='Class', fill=True, ax=ax, legend=False)
                else: # Scatter plots with decision boundaries
                    x_min, x_max = df[x_feature].min(), df[x_feature].max()
                    y_min, y_max = df[y_feature].min(), df[y_feature].max()
                    
                    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
                    
                    # Create a grid of normalized points for prediction
                    mesh_points_norm = np.full((xx.ravel().shape[0], 16), feature_means_norm)
                    # Normalize the grid points for the two active features
                    mesh_points_norm[:, x_idx] = (xx.ravel() - X_full[:, x_idx].min()) / (X_full[:, x_idx].max() - X_full[:, x_idx].min())
                    mesh_points_norm[:, y_idx] = (yy.ravel() - X_full[:, y_idx].min()) / (X_full[:, y_idx].max() - X_full[:, y_idx].min())

                    Z = predict_mlp_vectorized(mesh_points_norm, weights)
                    Z = Z.reshape(xx.shape)
                    
                    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.viridis, levels=6)
                    sns.scatterplot(data=df, x=x_feature, y=y_feature, hue='Class', ax=ax, legend=False, s=10)

                if i == 3: ax.set_xlabel(x_feature)
                if j == 0: ax.set_ylabel(y_feature)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'frames_beans/frame_{epoch:04d}.png')
        plt.close(fig)

    print("All bean frames generated successfully.")

if __name__ == '__main__':
    # Make sure to copy parse_weights_file and predict_mlp_vectorized here
    generate_bean_frames()