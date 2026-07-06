import os
import json

def main():
    dirs = "loss日志"
    out_file = os.path.join("scratch", "tune_results.txt")
    
    if not os.path.exists(dirs):
        print(f"Directory {dirs} not found")
        return
        
    files = sorted(os.listdir(dirs))
    
    results = {}
    for f in files:
        if f.endswith('.json'):
            path = os.path.join(dirs, f)
            try:
                with open(path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    results[f] = {
                        "mse": data.get("mse"),
                        "mae": data.get("mae"),
                        "gs_residual_weight": data.get("gs_residual_weight"),
                        "num_gaussians": data.get("num_gaussians"),
                        "stride": data.get("stride"),
                        "gs_dropout": data.get("gs_dropout"),
                        "learning_rate": data.get("learning_rate"),
                        "gate_type": data.get("gate_type", "none")
                    }
            except Exception as e:
                print(f"Error reading {f}: {e}")
                
    # Format and write to output file
    with open(out_file, 'w', encoding='utf-8') as out:
        out.write("=== Ablation Tuning Results Summary ===\n\n")
        out.write(f"{'File Name':<50} | {'Gating':<8} | {'MSE':<10} | {'MAE':<10} | {'Res Weight':<12} | {'Gaussians':<10} | {'LR':<10}\n")
        out.write("-" * 115 + "\n")
        for name, info in sorted(results.items()):
            out.write(f"{name:<50} | {info['gate_type']:<8} | {info['mse']:<10.6f} | {info['mae']:<10.6f} | {info['gs_residual_weight']:<12.6f} | {info['num_gaussians']:<10} | {info['learning_rate']:<10.6e}\n")
            
    print("Results written to scratch/tune_results.txt")

if __name__ == "__main__":
    main()
