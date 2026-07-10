import os
import sys

log_path = "scratch/ettm2_run_demo.log"
if not os.path.exists(log_path):
    print("Log file does not exist yet.")
    sys.exit(0)

# Try reading as UTF-16 first (often used by Powershell redirection)
try:
    with open(log_path, "r", encoding="utf-16") as f:
        lines = f.readlines()
except Exception:
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Failed to read with utf-8/utf-16: {e}")
        sys.exit(1)

# Print last 30 lines
print(f"--- LAST 30 LINES OF LOG (Total lines: {len(lines)}) ---")
for line in lines[-30:]:
    print(line, end="")
