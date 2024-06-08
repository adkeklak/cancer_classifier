from django.test import TestCase
from django.core.files.uploadedfile import SimpleUploadedFile
from cancer_classificator.forms import ImageUploadForm

class ImageUploadFormTest(TestCase):
    def test_valid_image(self):
        image_content = b'\x47\x49\x46\x38\x39\x61\x01\x00\x01\x00\x80\xff\x00\xff\xff\xff\x00\x00\x00\x21\xf9\x04\x01\x00\x00\x00\x00\x2c\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02\x4c\x01\x00\x3b'
        image = SimpleUploadedFile("ml_data/lungs/Bengin cases/Bengin case (2).jpg", image_content, content_type="image/jpeg")

        form_data = {'image': image}
        
        form = ImageUploadForm(files=form_data)

        self.assertTrue(form.is_valid())

    def test_invalid_image(self):
        form_data = {}

        form = ImageUploadForm(files=form_data)

        self.assertFalse(form.is_valid())

    def test_error_image(self):
        form_data = {}

        form = ImageUploadForm(files=form_data)

        self.assertIn('image', form.errors)
        