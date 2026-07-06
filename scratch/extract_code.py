with open("exp/exp_long_term_forecasting.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

found_idx = -1
for i, line in enumerate(lines):
    if "Adaptive Gating Direction Diagnostics" in line:
        found_idx = i
        break

if found_idx != -1:
    start = max(0, found_idx - 50)
    end = min(len(lines), found_idx + 50)
    for idx in range(start, end):
        print(f"{idx+1}: {lines[idx]}", end="")
else:
    print("Not found")
