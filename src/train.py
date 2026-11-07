import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sqlalchemy import create_engine
from utils.logger import get_logger

logger=get_logger(__name__)
def read_dataset(path):
    logger.info("Carregando o dataset...")
    engine=create_engine(f'sqlite:///{path}')
    df=pd.read_sql_query(
        "SELECT timestamp, heat_efficiency FROM heat_exchanger ORDER BY timestamp",
        engine
    )
    df['timestamp']=pd.to_datetime(df['timestamp'])
    df['day_index']=(df['timestamp']-df['timestamp'].min()).dt.days
    return df


def train(x:np.ndarray,y:np.ndarray) -> LinearRegression:
    logger.info("treinando o modelo..")
    model=LinearRegression()
    model.fit(x,y)
    return model

def save_model(model: LinearRegression, path:str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model,f)


def evaluate(model:LinearRegression, y:np.ndarray, x:np.ndarray):
    pred=model.predict(x)
    r2=r2_score(y,pred)
    logger.info(f"R² Score: {r2:.4f}")
    return r2

if __name__ == "__main__":

    df=read_dataset(path="data/heat_exchanger.db")

    x = np.asarray(df['day_index']).reshape(-1, 1)
    y = np.asarray(df['heat_efficiency'])
    model= train(x, y)

    evaluate(model,y,x)

    save_model(model, path="artifacts/heat_efficiency_model.pkl")
