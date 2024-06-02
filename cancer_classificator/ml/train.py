import numpy as np
import tensorflow as tf
from cancer_classificator import ml
from cancer_classificator.ml import model,utils
from cancer_classificator.ml.model import LungsModel, MlModel
from cancer_classificator.ml.utils import load_yaml, preprocess_image, load_data, decode

def train(model_type: MlModel = LungsModel(), model_name='lungs2', path='cancer_classificator/ml/models', version='1.0', epochs=5, verbose=2, from_file=False):
    if from_file:
        model_type, model_name, path, version , epochs, verbose = load_yaml(file_path='cancer_classificator/ml/config.yaml')

        if model_type == 'LungsModel':
            model = LungsModel()
        else:
            model = MlModel()
    else:
        model = model_type 

    X, y = load_data()

    try:
        model.train(X, y, epochs=epochs, verbose=verbose)
        model.save_model(f'{path}/{model_name}.{version}.keras')
        return "no error"
    except:
        return "error"

 
def model_test():
    model2 = LungsModel() 
    model2.load_model(path="cancer_classificator/ml/models/lungs.1.0.keras")
    img = preprocess_image(file_path="ml_data/lungs/Bengin cases/Bengin case (2).jpg")
    out = model2.predict(img)
    print(out)

if __name__ == '__main__':
    train(from_file=True)