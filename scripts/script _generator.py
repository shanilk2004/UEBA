import os
import time
import requests
import pandas as pd

og_directory=os.path.dirname(__file__)
rel_model_path=os.path.join(og_directory,"..","data","raw","UNSW_NB15_testing-set.parquet")
fn_data_path=os.path.abspath(rel_model_path)

train_data=pd.read_parquet(fn_data_path)
features=['dur','sbytes','dbytes','sload','dload']
final_train_data=train_data[features].dropna()

for index,row in final_train_data.iterrows():
    payload={
        "dur": float(row['dur']),
        "sbytes": int(row['sbytes']),
        "dbytes": int(row['dbytes']),
        "sload": float(row['sload']),
        "dload": float(row['dload'])
    }
    x=requests.post(url="http://127.0.0.1:8000/api/v1/score",json=payload)
    resp=x.json()
    desc=resp.get("prediction")
    raw=resp.get("raw_score")
    print(f"Row {index} | Model Decision: {desc} (Raw: {raw})")
    time.sleep(1.0)
