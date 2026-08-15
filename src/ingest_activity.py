import os
import sqlite3
import pandas as pd

def ingestlargecsv(filepath,dbpath,tablename,chunksize=100000):
    conn=sqlite3.connect(dbpath)

    fisrtchunk=True

    for chunk in pd.read_csv(filepath,chunksize=chunksize):
        mode = "replace" if fisrtchunk else "append"
        chunk.to_sql(tablename,conn,if_exists=mode,index=False)
        fisrtchunk=False
        print("chunk loaded..")

    try:
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{tablename}_user ON {tablename} (user)")
    except sqlite3.OperationalError:
        try:
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{tablename}_user ON {tablename} (user_id)")
        except sqlite3.OperationalError:
            pass    
    conn.close()
    print(f"sucesfully ingested {tablename}\n")


if __name__=="__main__":
    abspath=os.path.dirname(os.path.abspath(__file__))
    dbpath=os.path.join(abspath,"..","ueba_records.db")

    devpath=os.path.join(abspath,"..","data","archive","r4.2","device.csv")  
    emailpath=os.path.join(abspath,"..","data","archive","r4.2","email.csv")
    filepath=os.path.join(abspath,"..","data","archive","r4.2","file.csv")
    httppath=os.path.join(abspath,"..","data","archive","r4.2","http.csv")
    logpath=os.path.join(abspath,"..","data","archive","r4.2","logon.csv")
    psychpath=os.path.join(abspath,"..","data","archive","r4.2","psychometric.csv")

    ingestlargecsv(devpath,dbpath,"device")
    ingestlargecsv(emailpath,dbpath,"email")
    ingestlargecsv(filepath,dbpath,"files")
    ingestlargecsv(httppath,dbpath,"https")
    ingestlargecsv(logpath,dbpath,"logs")
    ingestlargecsv(psychpath,dbpath,"physchs")


