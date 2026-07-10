import pandas as pd
import os

for f in ["ETTm1.csv", "ETTm2.csv"]:
    path = os.path.join("data", f)
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"{f} shape: {df.shape}")
    else:
        print(f"{f} not found")
