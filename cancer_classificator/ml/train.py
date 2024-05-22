import numpy as np
from glob import glob
from PIL import Image
from model import LungsModel, MlModel
from utils import preprocess_data

def load_data(file_path = "ml_data/lungs"):
    data = glob(f'{file_path}/*/*.jpg')
    X, y = preprocess_data(data)
    return X, y

def train(model_type: MlModel = LungsModel(), model_name='lungs',path='ml/models',version='0.1'):
    X, y = load_data()
    model = model_type
    model.train(X, y)
    model.save_model(f'{path}/{model_name}.{version}.h5 ')

def main():
    train()
    
if __name__ == '__main__':
    main()