from unittest import TestCase
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from PIL import Image
from glob import glob
from keras.preprocessing import image
from unittest.mock import patch, mock_open

class test_cancer_utils(TestCase):

    @patch("builtins.open", new_callable=mock_open, read_data="""
    model_type: lung
    model_name: lungs_cancer_classificator
    path: /models/lungs
    version: 1.1
    epochs: 89
    verbose: 2
    """)
    def test_load_yaml(self, mock_yaml):
        from cancer_classificator.ml.utils import load_yaml
        yaml_data = load_yaml("some/place.yaml")
        excepted = ("lung", "lungs_cancer_classificator", "/models/lungs", 1.1, 89, 2)
        self.assertEqual(yaml_data, excepted)

    def test_decode(self):
        from cancer_classificator.ml.utils import decode
        self.assertEqual(decode([[1, 0, 0]]), 'normal')
        self.assertEqual(decode([[0.4, 0.3, 0.3]]), 'normal')
        self.assertEqual(decode([[0, 1, 0]]), 'bengin')
        self.assertEqual(decode([[0.1, 0.8, 0.1]]), 'bengin')
        self.assertEqual(decode([[0, 0, 1]]), 'malignant')

    @patch("cancer_classificator.ml.utils.glob", return_value=[
        'ml_data/lungs/Normal cases/Normal cases (1).jpg',
        'ml_data/lungs/Bengin cases/Bengin cases (2).jpg',
        'ml_data/lungs/Malignant cases/Malignant cases (3).jpg',
        'ml_data/lungs/Malignant cases/Malignant cases (4).jpg'
    ])
    def test_load_data(self, mock_data):
        from cancer_classificator.ml.utils import load_data

        X, y = load_data()
        self.assertEqual(len(X), 4)
        self.assertEqual(len(y), 4)

    
    def test_preprocess_data(self):
        from cancer_classificator.ml.utils import preprocess_data
        data = [
        'ml_data/lungs/Normal cases/Normal cases (1).jpg',
        'ml_data/lungs/Malignant cases/Malignant cases (4).jpg',
        'ml_data/lungs/Bengin cases/Bengin cases (2).jpg',
        'ml_data/lungs/Malignant cases/Malignant cases (3).jpg'
        ]
        X, y = preprocess_data(data)
        self.assertEqual(len(X), 4)
        self.assertEqual(len(y), 4)
        self.assertEqual(y.iloc[0], 0)
        self.assertEqual(y.iloc[1], 2)
        self.assertEqual(y.iloc[2], 1)
        self.assertEqual(y.iloc[3], 2)