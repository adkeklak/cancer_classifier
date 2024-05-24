import numpy as np
import tensorflow as tf
import yaml
from cancer_classificator import ml
from cancer_classificator.ml import model,utils
from cancer_classificator.ml.model import LungsModel, MlModel
from cancer_classificator.ml.utils import preprocess_data, preprocess_image, load_data

def train(model_type: MlModel = LungsModel(), model_name='lungs2', path='cancer_classificator/ml/models', version='1.0', epochs=5, verbose=2, from_file=False):
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
    model.save_model(f'{path}/{model_name}.{version}.keras')
 
def model_test():
    model2 = LungsModel() 
    model2.load_model(path="cancer_classificator/ml/models/lungs.0.1.2.keras")
    img = preprocess_image(file_path="ml_data/lungs/Bengin cases/Bengin case (1).jpg")
    out = model2.predict(img)
    print(out)

if __name__ == '__main__':
    train(from_file=True)