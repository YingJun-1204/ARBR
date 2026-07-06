import pandas as pd
df = pd.read_csv('data/weather.csv')
print("Shape:", df.shape)
print("Columns:", df.columns.tolist()[:5], "... total", len(df.columns))
