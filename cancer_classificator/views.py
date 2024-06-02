from django.shortcuts import render
from .forms import ImageUploadForm
from cancer_classificator.ml.model import MlModel, LungsModel
from cancer_classificator.ml.utils import preprocess_image, decode
from PIL import Image
import numpy as np

def image_upload_view(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            lungs_model = LungsModel()
            lungs_model.load_model(path="cancer_classificator/ml/models/lungs.1.0.keras")
            img_array = preprocess_image(image)
            predictions = lungs_model.predict(img_array)
            predicted = decode(predictions)
            predicted.capitalize()
            return render(request, 'result.html', {'predicted_class': predicted})
    else:
        form = ImageUploadForm()
    return render(request, 'upload.html', {'form': form})