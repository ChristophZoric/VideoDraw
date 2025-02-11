from model import build_crnn_model, load_quickdraw_data, preprocess_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
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

    save_dir = 'classification-crnn'
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'crnn_quickdraw_model.h5')
    label_path = os.path.join(save_dir, 'label_classes.npy')

    # Daten laden und vorbereiten
    sequences, labels = load_quickdraw_data(file_paths, max_samples_per_class=5000)
    print("Klassenverteilung nach dem Laden:", Counter(labels))

    max_length = 128
    processed_sequences = preprocess_sequences(sequences, max_length=max_length)

    label_encoder = LabelEncoder()
    
    if os.path.exists(label_path):
        print(f"Lade bestehende Labels aus {label_path}...")
        label_encoder.classes_ = np.load(label_path, allow_pickle=True)
    else:
        label_encoder.fit(labels)
    
    encoded_labels = label_encoder.transform(labels)
    one_hot_labels = to_categorical(encoded_labels, num_classes=len(label_encoder.classes_))

    print("Anzahl Klassen:", len(label_encoder.classes_))
    print("Klassen:", label_encoder.classes_)

    train_data, val_data, train_labels, val_labels = train_test_split(
        processed_sequences, one_hot_labels, test_size=0.2, random_state=42)

    # Prüfen, ob ein gespeichertes Modell existiert
    if os.path.exists(model_path):
        print(f"Lade bestehendes Modell aus {model_path}...")
        model = load_model(model_path)
    else:
        print("Kein bestehendes Modell gefunden. Erstelle neues Modell...")
        model = build_crnn_model(input_shape=(max_length, 2), num_classes=len(label_encoder.classes_),
                                 lstm_units=64, dropout_rate=0.3)

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    # Weitertrainieren oder neu trainieren
    model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=35, batch_size=32)

    # Speichern des trainierten Modells und der Klassen
    model.save(model_path)
    np.save(label_path, label_encoder.classes_)
    
    print("Modell und Klassen erfolgreich gespeichert!")
