from django.core.management import call_command
from django.test import TestCase
from unittest.mock import patch, MagicMock
from cancer_classificator.ml.model import LungsModel

class TestComamnd(TestCase):

    def setUp(self):
        self.mock_train = patch('cancer_classificator.ml.train.train').start()
        self.mock_lungs_model = patch('cancer_classificator.ml.model.LungsModel').start()

    def tearDown(self):
        patch.stopall()

    def test_handel(self):
        lungs_model_instance = self.mock_lungs_model.return_value
        
        call_command(
            'train_model',
            '--model_type=lungs',
            '--model_name=testmodel',
            '--path=path_to_model',
            '--epochs=5'
        )

        self.mock_train.assert_called_once_with(
            model_type=lungs_model_instance,
            model_name='testmodel',
            path='path_to_model',
            version='0.1',
            epochs=5,
            verbose=1,
            from_file=False
        )


