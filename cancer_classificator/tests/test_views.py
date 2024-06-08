from django.test import TestCase, Client
from cancer_classificator.forms import ImageUploadForm
from cancer_classificator.ml.utils import preprocess_image, decode
from unittest.mock import Mock, patch
from PIL import Image

class ImageUploadViewTest(TestCase):

    def setUp(self):
        self.client = Client()

    @patch('cancer_classificator.views.preprocess_image')
    @patch('cancer_classificator.views.LungsModel')
    @patch('cancer_classificator.views.ImageUploadForm')
    @patch('cancer_classificator.views.decode')
    def test_valid_upload(self, mock_decode, mock_form, mock_lungs_model, mock_preprocess_image):
        mock_image = Mock(spec=Image)
        mock_form.return_value.is_valid.return_value = True
        mock_form.cleaned_data = {'image': mock_image}
        mock_lungs_model.return_value.load_data.return_value = None
        mock_lungs_model.return_value.predict.return_value = [[0.1, 0.2, 0.7]]
        mock_preprocess_image.return_value = [0, 1, 2]
        mock_decode.return_value = 'normal'

        request = self.client.post('/', {'image': mock_image})

        self.assertTrue(mock_form.called)
        self.assertTrue(mock_form.return_value.is_valid.called)
        mock_lungs_model.return_value.load_model.assert_called_once_with(path="cancer_classificator/ml/models/lungs.1.0.keras")
        mock_lungs_model.return_value.predict.assert_called_once()
        mock_preprocess_image.assert_called_once()
        mock_decode.assert_called_once()
        self.assertTemplateUsed(request, 'result.html')
        self.assertEqual(request.context['predicted_class'], 'normal')

    
    @patch('cancer_classificator.views.ImageUploadForm')
    def test_invalid_upload(self, mock_form):
        mock_form.return_value.is_valid.return_value = False
        mock_form.cleaned_data = {}

        request = self.client.post('/')

        self.assertTrue(mock_form.called)
        self.assertTrue(mock_form.return_value.is_valid.called)
        self.assertTemplateUsed(request, 'upload.html')
        self.assertIn('form', request.context)
        self.assertTrue(request.context['form'], mock_form )
