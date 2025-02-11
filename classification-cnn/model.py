import os
import json
import numpy as np
import cv2
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def build_model(input_shape=(36, 36, 1), num_classes=5, dropout_rate=0.3):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def rasterize_sequence(sequence, img_size=36):
    if not sequence:
        return np.zeros((img_size, img_size), dtype=np.uint8)
    xs = [p[0] for p in sequence]
    ys = [p[1] for p in sequence]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    range_x = max_x - min_x if (max_x - min_x) != 0 else 1
    range_y = max_y - min_y if (max_y - min_y) != 0 else 1

    scaled_points = []
    for (x, y) in sequence:
        new_x = int(((x - min_x) / range_x) * (img_size - 1))
        new_y = int(((y - min_y) / range_y) * (img_size - 1))
        scaled_points.append((new_x, new_y))

    img = np.zeros((img_size, img_size), dtype=np.uint8)
    for i in range(len(scaled_points)-1):
        x1, y1 = scaled_points[i]
        x2, y2 = scaled_points[i+1]
        cv2.line(img, (x1, y1), (x2, y2), 255, 1)
    return img

def load_ndjson_data(file_paths, max_samples_per_class=5000):
    sequences = []
    labels = []
    for file_path in file_paths:
        class_name = os.path.splitext(os.path.basename(file_path))[0]
        class_count = 0
        with open(file_path, 'r') as f:
            for line in f:
                if class_count >= max_samples_per_class:
                    break
                data = json.loads(line)
                if data['recognized']:
                    flat_sequence = []
                    half_strokes = data['drawing'][:len(data['drawing']) // 2]
                    for stroke in half_strokes:
                        xs = stroke[0]
                        ys = stroke[1]
                        for x, y in zip(xs, ys):
                            flat_sequence.append((x, y))
                    sequences.append((flat_sequence, class_name))
                    class_count += 1
    return sequences

def show_samples(data, labels, class_names, num_samples=5):
    unique_classes = np.unique(np.argmax(labels, axis=1))  # Finde vorhandene Klassen
    num_classes = len(unique_classes)

    fig, axes = plt.subplots(1, num_classes, figsize=(10, 5))
    if num_classes == 1:
        axes = [axes]  # Falls nur eine Klasse vorhanden ist, in eine Liste packen

    for i, class_idx in enumerate(unique_classes):
        class_samples = np.where(np.argmax(labels, axis=1) == class_idx)[0]
        if len(class_samples) == 0:
            continue  # Falls keine Samples für eine Klasse existieren, überspringen
        sample_idx = np.random.choice(class_samples)
        img = data[sample_idx].squeeze()
        label = class_names[class_idx]

        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(label)
        axes[i].axis('off')

    plt.show()


def load_and_preprocess_data_from_ndjson(file_paths, num_classes=5, max_samples_per_class=5000, test_size=0.2, random_state=42):
    sequences = load_ndjson_data(file_paths, max_samples_per_class)
    class_names = sorted(list(set([c for _, c in sequences])))
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    data = []
    labels = []

    for seq, class_name in sequences:
        img = rasterize_sequence(seq, img_size=36)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=-1)  # (36,36) -> (36,36,1)
        data.append(img)
        labels.append(class_to_idx[class_name])

    data = np.array(data)
    labels = to_categorical(labels, num_classes=num_classes)

    train_data, val_data, train_labels, val_labels = train_test_split(
        data, labels, test_size=test_size, random_state=random_state
    )
    show_samples(train_data, train_labels, class_names)

    # Durchmischen
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    split_idx = int(len(data) * (1 - test_size))
    train_data, val_data = data[:split_idx], data[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]

    print("Klassen:", class_names)
    print("Klassenverteilung:", Counter([class_names[np.argmax(l)] for l in labels]))

    return (train_data, train_labels), (val_data, val_labels), class_names
