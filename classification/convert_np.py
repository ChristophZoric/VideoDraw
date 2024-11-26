import numpy as np
import matplotlib.pyplot as plt
import ast
import re
import cv2  # OpenCV für Linien


def convert_annotations(file_path, output_path='classification/converted_data.npy', image_size=(36, 36)):
    file_path = 'classification/annotations_data.txt'
    with open(file_path, 'r') as file:
        content = file.read()
    # process annotation input data
    content = re.sub(r'\b0+(\d)', r'\1', content)
    content = content.replace('\n', '').replace('), (', '),(')
    coordinates = ast.literal_eval(content)
    # create empty image
    width, height = image_size
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

    # Optional: Bild anzeigen (zur Überprüfung)
    # plt.ion()  # prevents app from hanging after plt.show()
    # plt.imshow(image, cmap='gray')
    # plt.title("Transformiertes Bild mit Linien")
    # plt.show()

    np.save('classification/converted_data.npy', image)
    return output_path
