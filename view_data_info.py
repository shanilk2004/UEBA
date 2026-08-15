import os
import pandas as pd

data_dir="data/archive/r4.2"

filetoinspect={
    "log":os.path.join(data_dir,"logon.csv"),
    "device":os.path.join(data_dir,"device.csv"),
    "HTTP":os.path.join(data_dir,"http.csv")
}

for name,path in filetoinspect.items():
    print(f"\n---first 5 rows of {name} ({path})---")
    try:
        df=pd.read_csv(path,nrows=5)
        print(df.columns.tolist())
        print(df.head(2))
    except Exception as e:
        print(f"error reading {name}:{e}")    