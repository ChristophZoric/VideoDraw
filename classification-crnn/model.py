import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


# 1. Laden der Sequenzdaten aus .ndjson-Dateien
def load_quickdraw_data(file_paths, max_samples_per_class=1000):
    sequences = []
    labels = []

    for file_path in file_paths:
        class_name = file_path.split('/')[-1].split('.')[0]  # Klassenname aus Dateiname
        with open(file_path, 'r') as f:
            for line in f:
                if len(sequences) >= max_samples_per_class * len(file_paths):
                    break
                data = json.loads(line)
                if data['recognized']:  # Nur erkannte Zeichnungen verwenden
                    sequence = data['drawing']
                    # Flache Sequenz aus x, y-Koordinaten erstellen
                    flat_sequence = []
                    for stroke in sequence:
                        flat_sequence.extend(list(zip(stroke[0], stroke[1])))  # x, y-Paare
                    sequences.append(flat_sequence)
                    labels.append(class_name)

    return sequences, labels


# 2. Vorverarbeitung der Sequenzdaten
def preprocess_sequences(sequences, max_length=128):
    # Normalisierung der Sequenzen (x, y-Koordinaten auf 0-1)
    normalized_sequences = [np.array(seq) / 255.0 for seq in sequences]

    # Padding/Truncation, um alle Sequenzen auf die gleiche Länge zu bringen
    padded_sequences = pad_sequences(normalized_sequences, maxlen=max_length, dtype='float32', padding='post', truncating='post')
    return padded_sequences


# 3. CRNN-Modell erstellen
def build_crnn_model(input_shape=(128, 2), num_classes=5):
    model = Sequential([
        # Masking Layer, um Padding zu ignorieren
        Masking(mask_value=0.0, input_shape=input_shape),

        # Bidirektionale LSTM-Schicht
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.5),

        # Weitere LSTM-Schicht
        LSTM(64),
        Dropout(0.5),

        # Vollständig verbundene Schichten
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
