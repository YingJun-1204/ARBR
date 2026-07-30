import sys
import os
import re
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Summarize forecasting benchmark results into a formatted table.")
    parser.add_argument("--file", type=str, default="result_long_term_forecast.txt", help="Path to result file")
    parser.add_argument("--start_line", type=int, default=0, help="Line index to start reading from")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Result file '{args.file}' does not exist.")
        return

    datasets = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity", "Traffic"]
    horizons = [96, 192, 336, 720]
    
    results = {ds: {} for ds in datasets}

    with open(args.file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    lines = lines[args.start_line:]

    current_setting = None
    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
        if line_str.startswith("mse:"):
            if current_setting is not None:
                m = re.search(r"mse:\s*([0-9\.]+),\s*mae:\s*([0-9\.]+)", line_str)
                if m:
                    mse = float(m.group(1))
                    mae = float(m.group(2))
                    
                    setting_lower = current_setting.lower()
                    matched_ds = None
                    for ds in datasets:
                        if ds.lower() in setting_lower:
                            matched_ds = ds
                            break
                    
                    matched_pl = None
                    for pl in horizons:
                        if f"_{pl}_" in setting_lower or setting_lower.endswith(f"_{pl}"):
                            matched_pl = pl
                            break
                    
                    if matched_ds and matched_pl:
                        results[matched_ds][matched_pl] = (mse, mae)
            current_setting = None
        else:
            current_setting = line_str

    header = f"| {'Dataset':<12} | {'96 (MSE/MAE)':<17} | {'192 (MSE/MAE)':<17} | {'336 (MSE/MAE)':<17} | {'720 (MSE/MAE)':<17} | {'Avg (MSE/MAE)':<17} |"
    divider = "+" + "-"*14 + "+" + "-"*19 + "+" + "-"*19 + "+" + "-"*19 + "+" + "-"*19 + "+" + "-"*19 + "+"
    
    print("\n" + "="*115)
    print("                      BENCHMARK EXPERIMENT RESULTS SUMMARY TABLE")
    print("="*115)
    print(divider)
    print(header)
    print(divider)

    pl_totals = {pl: {"mse": [], "mae": []} for pl in horizons}

    for ds in datasets:
        row_str = f"| {ds:<12} |"
        ds_mses = []
        ds_maes = []
        for pl in horizons:
            if pl in results[ds]:
                mse, mae = results[ds][pl]
                cell = f"{mse:.4f} / {mae:.4f}"
                ds_mses.append(mse)
                ds_maes.append(mae)
                pl_totals[pl]["mse"].append(mse)
                pl_totals[pl]["mae"].append(mae)
            else:
                cell = "N/A"
            row_str += f" {cell:<17} |"
        
        if ds_mses:
            avg_mse = float(np.mean(ds_mses))
            avg_mae = float(np.mean(ds_maes))
            avg_cell = f"{avg_mse:.4f} / {avg_mae:.4f}"
        else:
            avg_cell = "N/A"
        row_str += f" {avg_cell:<17} |"
        print(row_str)

    print(divider)
    
    avg_row = f"| {'Average':<12} |"
    all_mses = []
    all_maes = []
    for pl in horizons:
        if pl_totals[pl]["mse"]:
            p_mse = float(np.mean(pl_totals[pl]["mse"]))
            p_mae = float(np.mean(pl_totals[pl]["mae"]))
            cell = f"{p_mse:.4f} / {p_mae:.4f}"
            all_mses.extend(pl_totals[pl]["mse"])
            all_maes.extend(pl_totals[pl]["mae"])
        else:
            cell = "N/A"
        avg_row += f" {cell:<17} |"
    
    if all_mses:
        overall_cell = f"{float(np.mean(all_mses)):.4f} / {float(np.mean(all_maes)):.4f}"
    else:
        overall_cell = "N/A"
    avg_row += f" {overall_cell:<17} |"
    
    print(avg_row)
    print(divider)
    print("="*115 + "\n")

if __name__ == "__main__":
    main()
