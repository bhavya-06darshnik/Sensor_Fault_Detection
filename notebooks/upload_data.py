
from pymongo import MongoClient
import pandas as pd
import json

url="mongodb+srv://psinghyyyy:bhavya06@cluster1.mxmaa.mongodb.net/?retryWrites=true&w=majority&appName=Cluster1"

#crreate a new client
client=MongoClient(url)

DATABASE_NAME="Sensor"
COLLECTION_NAME="WaferFaults"


df=pd.read_csv(r"D:\Sensor fault\Sensor_Fault_Detection\notebooks\wafer_23012020_041211.csv")

df.head()

df=df.drop("Unnamed: 0", axis=1)


json_record=list(json.loads(df.T.to_json()).values())


type(json_record)

client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)