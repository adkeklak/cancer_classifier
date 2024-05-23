import numpy as np
import tensorflow as tf
from glob import glob
import yaml
from cancer_classificator import ml
from cancer_classificator.ml import model,utils
from cancer_classificator.ml.model import LungsModel, MlModel
from cancer_classificator.ml.utils import preprocess_data

def load_data(file_path = "ml_data/lungs"):
    data = glob(f'{file_path}/*/*.jpg')
    X, y = preprocess_data(data)
    return X, y

def train(model_type: MlModel = LungsModel(), model_name='lungs', path='cancer_classificator/ml/models', version='0.1', epochs=10, verbose=1, from_file=False):
    if from_file:
        with open('cancer_classificator/ml/config.yaml', 'r') as config_file:
            config = yaml.safe_load(config_file)

        model_type = config['model_type']
        model_name = config['model_name']
        path = config['path']
        version = config['version']
        epochs = config['epochs']
        verbose = config['verbose']

        if model_type == 'LungsModel':
            model = LungsModel()
    else:
        model = model_type 

    X, y = load_data()
    model.train(X, y, epochs=epochs, verbose=verbose)
    model.save_model(f'{path}/{model_name}.{version}.h5')
 
if __name__ == '__main__':
    train(from_file=True)