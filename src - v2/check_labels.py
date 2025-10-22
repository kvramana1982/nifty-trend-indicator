import pandas as pd

df = pd.read_parquet("data/processed/labeled_daily.parquet")

print("\n=== First 5 rows ===")
print(df.head())

# If "date" column not present, use "timestamp"
date_col = "date" if "date" in df.columns else "timestamp"

print("\nDates sorted:", df[date_col].is_monotonic_increasing)

print("\nTrend label distribution:")
print(df['label_class'].value_counts())

print("\nStrength bin distribution:")
print(df['strength_bin'].value_counts())
