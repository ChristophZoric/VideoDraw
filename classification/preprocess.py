import os
import numpy as np
from tensorflow.keras.utils import to_categorical
import cv2  # Für Resizing

def load_and_preprocess_data(data_dir, num_classes=5, sample_fraction=0.1):
    data = []
    labels = []

    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
    
    for idx, file in enumerate(files):
        class_data = np.load(os.path.join(data_dir, file))
        
        # Reshape auf 28x28, falls flach
        if class_data.shape[1] == 784:  # Flaches Array erkennen
            class_data = class_data.reshape((-1, 28, 28))
        
        # Sampling: Nur einen Teil der Daten verwenden
        sample_size = int(len(class_data) * sample_fraction)
        sampled_data = class_data[np.random.choice(len(class_data), size=sample_size, replace=False)]
        
        # Resize auf 64x64
        resized_data = []
        for image in class_data:
            resized_image = cv2.resize(image, (36, 36), interpolation=cv2.INTER_AREA)
            resized_data.append(resized_image)
        
        resized_data = np.array(resized_data) / 255.0  # Normalisierung
        resized_data = resized_data.reshape((-1, 36, 36, 1))  # Hinzufügen des Kanals

        class_labels = np.array([idx] * len(resized_data))
        class_labels = to_categorical(class_labels, num_classes)

        data.append(resized_data)
        labels.append(class_labels)

    data = np.vstack(data)
    labels = np.vstack(labels)

    indices = np.random.permutation(len(data))
    data, labels = data[indices], labels[indices]

    split = int(0.8 * len(data))
    train_data, val_data = data[:split], data[split:]
    train_labels, val_labels = labels[:split], labels[split:]

    return (train_data, train_labels), (val_data, val_labels)
