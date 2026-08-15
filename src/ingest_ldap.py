import glob
import os
import sqlite3
import pandas as pd

def discovercsv(pt):
    csv_files=sorted(glob.glob(pt))
    return csv_files

def procces_list(files):
    basket=[]
    for file_path in files:
        file_name=os.path.basename(file_path)
        name_split=os.path.splitext(file_name)
        df=pd.read_csv(file_path)
        df['snapshot_month']=name_split[0]
        basket.append(df)
    master_df=pd.concat(basket,ignore_index=True)
    return master_df    

def save(df,db_path):
    conn=sqlite3.connect(db_path)
    df.to_sql("ldap_records",conn,if_exists="replace",index=False)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_ldap_user ON ldap_records (employee_name);")

    conn.close()

abs_path=os.path.dirname(os.path.abspath(__file__))
pattern=os.path.join(abs_path,"..","data","archive","r4.2","LDAP","*.csv")
db_path=os.path.join(abs_path,"..","ueba_records.db")


csvlist=discovercsv(pattern)
master_dataframe=procces_list(csvlist)
save(master_dataframe, db_path)
print("Ingestion complete! Data successfully saved to SQLite.")


