from model import build_crnn_model
from model import load_quickdraw_data, preprocess_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from collections import Counter
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Funktion zur Visualisierung der Sequenzen
def visualize_all_classes(sequences, labels, class_names):
    # Gruppiere Sequenzen nach Klassen
    class_to_sequences = {class_name: [] for class_name in class_names}
    for seq, label in zip(sequences, labels):
        class_name = class_names[np.argmax(label)]
        if seq:  # Nur nicht-leere Sequenzen hinzufügen
            class_to_sequences[class_name].append(seq)

    plt.figure(figsize=(15, 10))
    for i, (class_name, seq_list) in enumerate(class_to_sequences.items()):
        if seq_list:  # Falls es Sequenzen für diese Klasse gibt
            sequence = seq_list[0]  # Nimm die erste Sequenz dieser Klasse
            
            # **Sicherstellen, dass die Sequenz nicht leer ist**
            if len(sequence) > 0:
                # Extrahiere die Punkte
                xs, ys = zip(*[(p[0], p[1]) for p in sequence if p[0] != 0 or p[1] != 0])

                # Plot
                plt.subplot(1, len(class_names), i + 1)
                plt.plot(xs, ys, marker='o', markersize=2, linewidth=1)
                plt.gca().invert_yaxis()  # Invertiere die Y-Achse
                plt.title(class_name)
                plt.axis('off')

    plt.show()



if __name__ == "__main__":
    file_paths = [
        'data-ndjsons/basketball.ndjson',
        'data-ndjsons/car.ndjson',
        'data-ndjsons/cloud.ndjson',
        'data-ndjsons/duck.ndjson',
        'data-ndjsons/plane.ndjson'
    ]

    model_path = os.path.join('classification-crnn/crnn_quickdraw_model.h5')

    sequences, labels = load_quickdraw_data(file_paths, max_samples_per_class=5000)
    print("Klassenverteilung nach dem Laden:", Counter(labels))

    max_length = 128
    processed_sequences = preprocess_sequences(sequences, max_length=max_length)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    one_hot_labels = to_categorical(encoded_labels)

    print("Anzahl Klassen:", len(label_encoder.classes_))
    print("Klassen:", label_encoder.classes_)

    visualize_all_classes(sequences, one_hot_labels, label_encoder.classes_)

    train_data, val_data, train_labels, val_labels = train_test_split(
        processed_sequences, 
        one_hot_labels, 
        test_size=0.2, 
        random_state=42
    )

    # Prüfen, ob ein gespeichertes Modell existiert
    if os.path.exists(model_path):
        print(f"Lade bestehendes Modell aus {model_path}...")
        model = load_model(model_path)
    else:
        print("Kein bestehendes Modell gefunden. Erstelle neues Modell...")
        model = build_crnn_model(input_shape=(max_length, 2), num_classes=len(label_encoder.classes_),
                                 lstm_units=64, dropout_rate=0.3)
    
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
