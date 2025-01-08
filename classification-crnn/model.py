import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def load_quickdraw_data(file_paths, max_samples_per_class=5000):
    sequences = []
    labels = []

    for file_path in file_paths:
        class_name = file_path.split('/')[-1].split('.')[0]
        class_sequences_count = 0 

        with open(file_path, 'r') as f:
            for line in f:
                if class_sequences_count >= max_samples_per_class:
                    break

                data = json.loads(line)
                if data['recognized']:
                    sequence = data['drawing']
                    flat_sequence = []
                    for stroke in sequence:
                        flat_sequence.extend(list(zip(stroke[0], stroke[1]))) 

                    sequences.append(flat_sequence)
                    labels.append(class_name)
                    class_sequences_count += 1

    return sequences, labels

def preprocess_sequences(sequences, max_length=128):
    normalized_sequences = [np.array(seq) / 255.0 for seq in sequences]

    padded_sequences = pad_sequences(normalized_sequences, maxlen=max_length, dtype='float32', padding='post', truncating='post')
    return padded_sequences



def build_crnn_model(input_shape=(128, 2), num_classes=5, lstm_units=64, dropout_rate=0.3):
    model = Sequential([
        Masking(mask_value=0.0, input_shape=input_shape),
        Bidirectional(LSTM(lstm_units, return_sequences=True)),
        Dropout(dropout_rate),
        LSTM(lstm_units),
        Dropout(dropout_rate),
        Dense(128, activation='relu'),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')
    ])
    return model

