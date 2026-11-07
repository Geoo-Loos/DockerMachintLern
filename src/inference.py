import pickle
from utils.logger import get_logger
import numpy as np
import argparse



logger=get_logger(__name__)
def load_model(model_path):
    logger.info("Carregando o modelo...")
    with open(model_path, 'rb') as f:
        model=pickle.load(f)
    return model

def predict_eff(model, x):

    x_array=np.array(x).reshape(1,-1)
    logger.info("Realizando predição de eficiencia..")
    res=model.predict(x_array)
    logger.info(f"É {res[0]}, no dia {x}")

def predict_data(model,y):
    coef=model.coef_
    intercept=model.intercept_
    dia=(y-intercept)/coef[0]
    logger.info(f"É {dia[0]}, no dia {y}")

if __name__=="__main__":
    model=load_model('artifacts/heat_efficiency_model.pkl')
    parser=argparse.ArgumentParser(description="")
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--efficiency',type=float)
    group.add_argument('--data',type=float)

    args=parser.parse_args()

    if args.efficiency is not None:
        predict_eff(model, args.efficiency)
    elif args.data is not None:
        predict_data(model, args.data)