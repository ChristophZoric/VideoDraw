from model import build_crnn_model
from model import load_quickdraw_data, preprocess_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from collections import Counter
import numpy as np
import os

if __name__ == "__main__":
    file_paths = [
        'data-ndjsons/basketball.ndjson',
        'data-ndjsons/car.ndjson',
        'data-ndjsons/cloud.ndjson',
        'data-ndjsons/duck.ndjson',
        'data-ndjsons/plane.ndjson'
    ]

    sequences, labels = load_quickdraw_data(file_paths, max_samples_per_class=5000)
    print("Klassenverteilung nach dem Laden:", Counter(labels))

    max_length = 128
    processed_sequences = preprocess_sequences(sequences, max_length=max_length)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    one_hot_labels = to_categorical(encoded_labels)

    print("Anzahl Klassen:", len(label_encoder.classes_))
    print("Klassen:", label_encoder.classes_)

    train_data, val_data, train_labels, val_labels = train_test_split(
        processed_sequences, 
        one_hot_labels, 
        test_size=0.2, 
        random_state=42
    )

    model = build_crnn_model(
        input_shape=(max_length, 2),
        num_classes=len(label_encoder.classes_),
        lstm_units=64,
        dropout_rate=0.3
    )
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history_crnn = model.fit(
        train_data, train_labels,
        validation_data=(val_data, val_labels),
        epochs=5,
        batch_size=32
    )

    save_dir = 'classification-crnn'
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'crnn_quickdraw_model.h5')
    model.save(model_path)
    np.save(os.path.join(save_dir, 'label_classes.npy'), label_encoder.classes_)

    np.save(os.path.join(save_dir, 'history_crnn.npy'), history_crnn.history)

    np.save(os.path.join(save_dir, 'val_data.npy'), val_data)
    np.save(os.path.join(save_dir, 'val_labels.npy'), val_labels)

    print("CRNN Modell, Klassen, Trainings-Historie und Validierungsdaten erfolgreich gespeichert!")
