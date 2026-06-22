import pandas as pd
import os

df = pd.read_csv("data/raw/spam.csv", encoding="latin-1")

# Keep only needed columns
df = df[["v1", "v2"]]
df.columns = ["label", "text"]

# Remove duplicates
df = df.drop_duplicates()

os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/spam.csv", index=False)

print(f"Processed {len(df)} rows → data/processed/spam.csv")