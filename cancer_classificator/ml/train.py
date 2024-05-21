import numpy as np
from ml.model import MLModel

def load_training_data():
    X = 0
    y = 0
    return X, y

def train():
    X, y = load_training_data()
    model = MLModel()
    model.train(X, y)
    model.save_model('ml/model.pkl')

if __name__ == '__main__':
    train()