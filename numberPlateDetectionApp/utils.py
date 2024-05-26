import pytesseract
import PIL.Image
import cv2
import numpy as np
import os

def process_image(image_path, filename):
    # Read uploaded image
    I = cv2.imread(image_path)
    if I is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Perform image processing operations
    Igray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    rows, cols = Igray.shape

    # Dilate and Erode Image in order to remove noise
    kernel = np.ones((3, 3), np.uint8)
    Idilate = cv2.dilate(Igray, kernel, iterations=1)
    I = Idilate

    difference = 0
    total_sum = 0

    # PROCESS EDGES IN HORIZONTAL DIRECTION
    print('Processing Edges Horizontally...')
    max_horz = 0
    maximum = 0
    horz1 = np.zeros(cols)

    for i in range(1, cols):
        total_sum = 0
        for j in range(1, rows):
            difference = np.abs(np.int32(I[j, i]) - np.int32(I[j - 1, i]))
            if difference > 20:
                total_sum += difference
        horz1[i] = total_sum
        if total_sum > maximum:
            max_horz = i
            maximum = total_sum

    average = np.mean(horz1)

    # Smoothen the Horizontal Histogram by applying Low Pass Filter
    horz = np.convolve(horz1, np.ones(41) / 41, mode='same')

    # Filter out Horizontal Histogram Values by applying Dynamic Threshold
    print('Filter out Horizontal Histogram...')
    horz[horz < average] = 0
    I[:, horz == 0] = 0

    # PROCESS EDGES IN VERTICAL DIRECTION
    print('Processing Edges Vertically...')
    maximum = 0
    max_vert = 0
    vert1 = np.zeros(rows)

    for i in range(1, rows):
        total_sum = 0
        for j in range(1, cols):
            difference = np.abs(np.int32(I[i, j]) - np.int32(I[i, j - 1]))
            if difference > 20:
                total_sum += difference
        vert1[i] = total_sum
        if total_sum > maximum:
            max_vert = i
            maximum = total_sum

    average = np.mean(vert1)

    # Smoothen the Vertical Histogram by applying Low Pass Filter
    vert = np.convolve(vert1, np.ones(41) / 41, mode='same')

    # Filter out Vertical Histogram Values by applying Dynamic Threshold
    vert[vert < average] = 0
    I[vert == 0, :] = 0

    # Find Probable candidates for Number Plate
    column = []
    for i in range(1, cols - 2):
        if horz[i] != 0 and horz[i - 1] == 0 and horz[i + 1] == 0:
            column.extend([i, i])
        elif (horz[i] != 0 and horz[i - 1] == 0) or (horz[i] != 0 and horz[i + 1] == 0):
            column.append(i)

    row = []
    for i in range(1, rows - 2):
        if vert[i] != 0 and vert[i - 1] == 0 and vert[i + 1] == 0:
            row.extend([i, i])
        elif (vert[i] != 0 and vert[i - 1] == 0) or (vert[i] != 0 and vert[i + 1] == 0):
            row.append(i)

    if len(column) % 2:
        column.append(cols)

    if len(row) % 2:
        row.append(rows)

    # Region of Interest Extraction
    for i in range(0, len(row), 2):
        for j in range(0, len(column), 2):
            if not ((max_horz >= column[j] and max_horz <= column[j + 1]) and
                    (max_vert >= row[i] and max_vert <= row[i + 1])):
                I[row[i]:row[i + 1], column[j]:column[j + 1]] = 0

    processed_image_path = os.path.join('media', 'processed_' + filename)
    print(f"Saving processed image to {processed_image_path}")
    cv2.imwrite(processed_image_path, I)

    # Ensure the file is saved correctly
    if not os.path.isfile(processed_image_path):
        raise FileNotFoundError(f"Failed to save the processed image at {processed_image_path}")

    # OCR using Tesseract on the original image
    my_config = r"--psm 7 --oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(PIL.Image.open(processed_image_path), config=my_config)

    return processed_image_path, text
