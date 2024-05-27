from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
from .utils import process_image  # Correct the import path

def detect_plate(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_image = request.FILES['image']
        fs = FileSystemStorage(location=settings.MEDIA_ROOT)
        filename = fs.save(uploaded_image.name, uploaded_image)
        uploaded_file_url = fs.url(filename)

        # Process the uploaded image
        input_image_path = os.path.join(settings.MEDIA_ROOT, filename)
        processed_image_path, ocr_text = process_image(input_image_path, filename)
        
        # Here we ensure that processed_image_path is a relative path
        processed_image_url = settings.MEDIA_URL + os.path.basename(processed_image_path)

        # Render the result page with the processed image and OCR text
        return render(request, 'plate_detector/result.html', {
            'processed_image_url': processed_image_url,
            'ocr_text': ocr_text,
        })

    return render(request, 'plate_detector/detect_plate.html')
