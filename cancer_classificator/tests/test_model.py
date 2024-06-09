from unittest import TestCase
import numpy as np
import pandas as pd
import tensorflow as tf
from unittest.mock import patch, mock_open, MagicMock
from keras.utils import to_categorical
from cancer_classificator.ml.utils import decode_image
from cancer_classificator.ml.model import LungsModel

class TestModel(TestCase):

    def setUp(self):
        self.lungs_model = LungsModel()
        self.X = np.array(['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg'])
        self.y = np.array([0, 2, 1, 2])
        self.model_path = 'path/to/model.keras'

    @patch('cancer_classificator.ml.model.load_model')
    def test_load_model(self, mock_load_model):
        mock_load_model.return_value = MagicMock(name='MockModel')

        self.lungs_model.load_model(self.model_path)

        mock_load_model.assert_called_once_with(self.model_path)
        self.assertIsInstance(self.lungs_model.model, MagicMock)
 
    @patch('tensorflow.keras.models.Sequential.save')
    def test_save_model(self, mock_save_model):
        self.lungs_model.save_model(self.model_path)
        
        mock_save_model.assert_called_once_with(self.model_path)

    @patch('tensorflow.data.Dataset.from_tensor_slices')
    @patch('tensorflow.keras.Sequential.fit')
    @patch('cancer_classificator.ml.utils.decode_image', side_effect=lambda x, y, augment: (tf.random.uniform((512, 512, 3)), y))
    def test_train(self, mock_decode_image, mock_fit, mock_from_tensor_slices):
        self.lungs_model.train(self.X, self.y, epochs=1, verbose=1)

        self.assertTrue(mock_fit.called)