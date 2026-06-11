from contextlib import asynccontextmanager
import pickle
from fastapi import FastAPI,Request
import pandas as pd
from pydantic import BaseModel

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("loading the isolation tree model")
    with open("./models/baseline_isolation_forest.pkl","rb")as f:   
        app.state.model=pickle.load(f)
    print("model opened")
    yield
    print("model closed")    

app=FastAPI(lifespan=lifespan)

class networklog(BaseModel):
    dur: float
    sbytes: int
    dbytes: int
    sload: float
    dload: float

@app.get("/")
def read_root():
    return{"status:activated"}

@app.post("/api/v1/score")
def analyse(packet:networklog,request:Request):
    data_dic=packet.model_dump()
    df=pd.DataFrame([data_dic])
    model=request.app.state.model

    pred= model.predict(df)[0]
    status= "anomaly" if pred == -1 else "normal"

    return{
        "prediction":status,
        "raw_score":int(pred)
    }
