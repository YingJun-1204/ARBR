import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_etth1_first_512():
    # 1. Read the dataset
    csv_path = os.path.join("data", "ETTh1.csv")
    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Parse date and take first 512 points
    df['date'] = pd.to_datetime(df['date'])
    df_subset = df.iloc[:512]
    
    # 2. Setup the plot style (sleek, professional look)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(16, 12), sharex=True)
    axes = axes.flatten()
    
    features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    # Plot each feature
    for i, feature in enumerate(features):
        ax = axes[i]
        ax.plot(df_subset['date'], df_subset[feature], color=colors[i], label=feature, linewidth=1.5)
        ax.set_title(f"{feature} Curve", fontsize=12, fontweight='bold', pad=8)
        ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
    
    # Disable the 8th subplot (since we have 7 features)
    axes[-1].axis('off')
    
    # Global titles and formatting
    plt.suptitle("ETTh1 Dataset - First 512 Time Points (7 Features)", fontsize=16, fontweight='bold', y=0.98)
    fig.text(0.5, 0.02, 'Date/Time (Hourly)', ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the output image
    output_filename = "etth1_first_512.png"
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully generated and saved plot to '{output_filename}'")

if __name__ == "__main__":
    plot_etth1_first_512()
