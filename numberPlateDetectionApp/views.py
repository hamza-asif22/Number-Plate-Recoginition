# plate_detector/views.py
from django.shortcuts import render
import cv2
from django.core.files.storage import FileSystemStorage

from numberPlateDetectionApp.utils import process_image

def detect_plate(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_image = request.FILES['image']
        fs = FileSystemStorage(location='media/')
        filename = fs.save(uploaded_image.name, uploaded_image)
        uploaded_file_url = fs.url(filename)

        # Read uploaded image
        I = cv2.imread('media/' + filename)

        processed_image_path = process_image('media/' + filename, filename)
        # Perform number plate detection and processing...
        # (Add your image processing code here)
        
        
        return render(request, 'plate_detector/result.html', {'processed_image_url': processed_image_path})

    return render(request, 'plate_detector/detect_plate.html')
