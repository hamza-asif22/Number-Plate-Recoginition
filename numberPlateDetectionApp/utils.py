# plate_detector/utils.py
import pytesseract
import PIL.Image
import cv2
import numpy as np

def process_image(image_path, filename):
    # Read uploaded image
    I = cv2.imread(image_path)
    


    # Perform image processing operations
    Igray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    rows, cols = Igray.shape

    # Dilate and Erode Image in order to remove noise
    Idilate = np.copy(Igray)
    for i in range(rows):
        for j in range(1, cols - 1):
            temp = max(Igray[i, j - 1], Igray[i, j])
            Idilate[i, j] = max(temp, Igray[i, j + 1])
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
            if I[j, i] > I[j - 1, i]:
                difference = np.uint32(I[j, i] - I[j - 1, i])
            else:
                difference = np.uint32(I[j - 1, i] - I[j, i])
            if difference > 20:
                total_sum += difference
        horz1[i] = total_sum
        if total_sum > maximum:
            max_horz = i
            maximum = total_sum

    average = np.sum(horz1) / cols

    # Smoothen the Horizontal Histogram by applying Low Pass Filter
    horz = np.copy(horz1)
    for i in range(20, cols - 21):
        horz[i] = np.sum(horz1[i - 20:i + 21]) / 41

    # Filter out Horizontal Histogram Values by applying Dynamic Threshold
    print('Filter out Horizontal Histogram...')
    for i in range(cols):
        if horz[i] < average:
            horz[i] = 0
            I[:, i] = 0

    # PROCESS EDGES IN VERTICAL DIRECTION
    print('Processing Edges Vertically...')
    maximum = 0
    max_vert = 0
    vert1 = np.zeros(rows)

    for i in range(1, rows):
        total_sum = 0
        for j in range(1, cols):
            if I[i, j] > I[i, j - 1]:
                difference = np.uint32(I[i, j] - I[i, j - 1])
            else:
                difference = np.uint32(I[i, j - 1] - I[i, j])
            if difference > 20:
                total_sum += difference
        vert1[i] = total_sum
        if total_sum > maximum:
            max_vert = i
            maximum = total_sum

    average = np.sum(vert1) / rows

    # Smoothen the Vertical Histogram by applying Low Pass Filter
    vert = np.copy(vert1)
    for i in range(20, rows - 21):
        vert[i] = np.sum(vert1[i - 20:i + 21]) / 41

    # Filter out Vertical Histogram Values by applying Dynamic Threshold
    for i in range(rows):
        if vert[i] < average:
            vert[i] = 0
            I[i, :] = 0

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

    processed_image_path = 'media/processed_' + filename
    cv2.imwrite(processed_image_path, I)
   
    # OCR using Tesseract on the original image
    my_config = r"--psm 7 --oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(PIL.Image.open(processed_image_path), config=my_config)

    return processed_image_path, text
