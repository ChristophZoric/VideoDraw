import numpy as np
import matplotlib.pyplot as plt
import ast
import re
import cv2  # OpenCV für Linien
from test import Predictor

file_path = 'classification/annotations_data.txt'
with open(file_path, 'r') as file:
    content = file.read()

content = re.sub(r'\b0+(\d)', r'\1', content)
content = content.replace('\n', '')
content = content.replace('), (', '),(')

coordinates = ast.literal_eval(content)

width, height = 36, 36
image = np.zeros((height, width), dtype=np.uint8)

for segment in coordinates:
    for i in range(len(segment) - 1):
        x1 = min(width - 1, int(segment[i][0] / max([pt[0]
                 for pts in coordinates for pt in pts]) * width))
        y1 = min(height - 1, int(segment[i][1] / max([pt[1]
                 for pts in coordinates for pt in pts]) * height))
        x2 = min(width - 1, int(segment[i + 1][0] / max([pt[0]
                 for pts in coordinates for pt in pts]) * width))
        y2 = min(height - 1, int(segment[i + 1][1] / max([pt[1]
                 for pts in coordinates for pt in pts]) * height))

        # Linie zwischen Punkten zeichnen
        cv2.line(image, (x1, y1), (x2, y2), 255, 1)

np.save('classification/converted_data.npy', image)

Predictor.load_and_test_coordinates('classification/converted_data.npy')

# Optional: Bild anzeigen (zur Überprüfung)
plt.imshow(image, cmap='gray')
plt.title("Transformiertes Bild mit Linien")
print("converter done")
plt.show()
