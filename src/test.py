import pandas as pd
df = pd.read_parquet("data/processed/labeled_daily.parquet")
print(df.columns.tolist())
print(df[['date','label_class','label_class_bin']].head())
