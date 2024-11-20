import os
import numpy as np
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_data(data_dir, num_classes=3):
    data = []
    labels = []

    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
    
    for idx, file in enumerate(files):
        class_data = np.load(os.path.join(data_dir, file))

        class_data = class_data / 255.0  # Werte zwischen 0 und 1

        class_data = class_data.reshape((-1, 28, 28, 1))

        class_labels = np.array([idx] * len(class_data))
        class_labels = to_categorical(class_labels, num_classes)

        data.append(class_data)
        labels.append(class_labels)

    data = np.vstack(data)
    labels = np.vstack(labels)

    indices = np.random.permutation(len(data))
    data, labels = data[indices], labels[indices]

    split = int(0.8 * len(data))
    train_data, val_data = data[:split], data[split:]
    train_labels, val_labels = labels[:split], labels[split:]

    return (train_data, train_labels), (val_data, val_labels)