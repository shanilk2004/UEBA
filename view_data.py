import os
import pandas as pd

rel_path=os.path.join("data","raw","UNSW_NB15_testing-set.parquet")
path=os.path.abspath(rel_path)

df=pd.read_parquet(path)
attack_rows = df[df['label'] == 1]

# Print the index and our 5 MVP features for the first 3 attacks
print("🎯 GUARANTEED ATTACK ROWS FOUND IN DATASET:")
features = ['dur', 'sbytes', 'dbytes', 'sload', 'dload']
print(attack_rows[features].head(3))