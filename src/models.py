from sqlalchemy import Column, Integer,Float,String,DateTime
from datetime import datetime
from src.database import Base

class NetworkLog(Base):
    __tablename__="network_logs"

    id = Column(Integer, primary_key=True, index=True)
    dur = Column(Float, nullable=False)
    sbytes = Column(Integer, nullable=False)
    dbytes = Column(Integer, nullable=False)
    sload = Column(Float, nullable=False)
    dload = Column(Float, nullable=False)
    
    prediction = Column(String, nullable=False)  
    raw_score = Column(Float, nullable=False)   
    timestamp = Column(DateTime, default=datetime.utcnow)