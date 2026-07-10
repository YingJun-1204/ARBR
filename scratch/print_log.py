import sys

log_path = "scratch/ettm2_test2.log"
# Try reading with utf-16 or utf-8
try:
    with open(log_path, "r", encoding="utf-16") as f:
        lines = f.readlines()
except Exception:
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

print(f"Total lines in log: {len(lines)}")
for l in lines[-50:]:
    print(l.strip())
