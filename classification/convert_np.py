import numpy as np
import matplotlib.pyplot as plt
import ast
import re
from test import Predictor
from tensorflow.keras.models import load_model

file_path = 'classification/annotations_data.txt'
with open(file_path, 'r') as file:
    content = file.read()

content = re.sub(r'\b0+(\d)', r'\1', content)

content = content.replace('\n', '') 
content = content.replace('), (', '),(')  

coordinates = ast.literal_eval(content)

width, height = 28, 28
image = np.zeros((height, width))

for segment in coordinates:
    for (x, y) in segment:
        x_scaled = min(width - 1, int(x / max([pt[0] for pts in coordinates for pt in pts]) * width))
        y_scaled = min(height - 1, int(y / max([pt[1] for pts in coordinates for pt in pts]) * height))
        image[y_scaled, x_scaled] = 255 

flattened_image = image.flatten()
np.save('classification/converted_data.npy', image)
Predictor.load_and_test_coordinates('classification/converted_data.npy')

# Optional: Bild anzeigen (zur ÃœberprÃ¼fung)
plt.imshow(image, cmap='gray')
plt.title("Transformiertes Bild")
plt.show()

# Ausgabe des eindimensionalen Arrays
print(flattened_image)