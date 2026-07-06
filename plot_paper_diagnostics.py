import os
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_diagnostics():
    datasets = ["electricity", "weather", "etth1", "etth2", "ettm1", "ettm2"]
    display_names = {
        "electricity": "Electricity",
        "weather": "Weather",
        "etth1": "ETTh1",
        "etth2": "ETTh2",
        "ettm1": "ETTm1",
        "ettm2": "ETTm2"
    }
    
    data = []
    
    for ds in datasets:
        file_path = os.path.join("loss_mdagV9", f"diagnostics_{ds}_best.txt")
        if not os.path.exists(file_path):
            print(f"Warning: file {file_path} not found.")
            continue
            
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Extract EGC
        egc_mean = float(re.search(r"Mean EGC\s*:\s*([\d\.]+)", content).group(1))
        egc_min = float(re.search(r"Min EGC\s*:\s*([\d\.]+)", content).group(1))
        egc_max = float(re.search(r"Max EGC\s*:\s*([\d\.]+)", content).group(1))
        
        # Extract LGA Means
        lga_neutral = float(re.search(r"LGA Mean \(Routed to Neutral\):\s*(-?[\d\.]+)", content).group(1))
        lga_forward = float(re.search(r"LGA Mean \(Routed to Forward\):\s*(-?[\d\.]+)", content).group(1))
        lga_reverse = float(re.search(r"LGA Mean \(Routed to Reverse\):\s*(-?[\d\.]+)", content).group(1))
        
        # Extract Correlation
        corr = float(re.search(r"Correlation\(LGA,\s*Energy\)\s*->\s*mean:\s*([\d\.]+)", content).group(1))
        
        # Extract Argmax Ratios
        neutral_ratio = float(re.search(r"Neutral:\s*([\d\.]+)%", content).group(1)) / 100.0
        forward_ratio = float(re.search(r"Forward:\s*([\d\.]+)%", content).group(1)) / 100.0
        reverse_ratio = float(re.search(r"Reverse:\s*([\d\.]+)%", content).group(1)) / 100.0
        
        data.append({
            "name": display_names[ds],
            "egc_mean": egc_mean,
            "egc_min": egc_min,
            "egc_max": egc_max,
            "lga_neutral": lga_neutral,
            "lga_forward": lga_forward,
            "lga_reverse": lga_reverse,
            "corr": corr,
            "neutral_ratio": neutral_ratio,
            "forward_ratio": forward_ratio,
            "reverse_ratio": reverse_ratio
        })
        
    return data

def generate_plots(data):
    # Set professional plotting style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    
    names = [d["name"] for d in data]
    x = np.arange(len(names))
    
    # ----------------------------------------------------
    # Left Panel: EGC (Effective Gaussian Count) vs Complexity
    # ----------------------------------------------------
    egc_means = [d["egc_mean"] for d in data]
    egc_mins = [d["egc_mean"] - d["egc_min"] for d in data]
    egc_maxs = [d["egc_max"] - d["egc_mean"] for d in data]
    yerr = np.array([egc_mins, egc_maxs])
    
    bars = ax1.bar(x, egc_means, yerr=yerr, color='#2c3e50', alpha=0.85, capsize=5, edgecolor='none', label='Mean EGC')
    ax1.set_ylabel('Effective Gaussian Count (EGC)', fontsize=12, fontweight='bold', labelpad=8)
    ax1.set_title('A: CAS Capacity Selection vs. Signal Complexity', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 8.5)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
                    
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # ----------------------------------------------------
    # Right Panel: LGA Mean (Z-score) by Gating Decision
    # ----------------------------------------------------
    width = 0.25
    lga_n = [d["lga_neutral"] for d in data]
    lga_f = [d["lga_forward"] for d in data]
    lga_r = [d["lga_reverse"] for d in data]
    
    rects1 = ax2.bar(x - width, lga_n, width, label='Neutral (Align)', color='#95a5a6', alpha=0.85)
    rects2 = ax2.bar(x, lga_f, width, label='Forward (Reinforce)', color='#2ecc71', alpha=0.85)
    rects3 = ax2.bar(x + width, lga_r, width, label='Reverse (Filter)', color='#e74c3c', alpha=0.85)
    
    ax2.set_ylabel('LGA Z-Score Anomaly Mean', fontsize=12, fontweight='bold', labelpad=8)
    ax2.set_title('B: Local Geometry Alignment vs. Gating Decisions', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=11, fontweight='bold')
    ax2.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none')
    
    # Drawing horizontal line at 0 for Z-score reference
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save file
    output_path = "paper_diagnostics_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved publication-quality figure to {output_path}")

if __name__ == "__main__":
    data = parse_diagnostics()
    generate_plots(data)
