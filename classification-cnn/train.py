from model import build_model, load_and_preprocess_data_from_ndjson
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

if __name__ == "__main__":
    file_paths = [
        'data-ndjsons/basketball.ndjson',
        'data-ndjsons/car.ndjson',
        'data-ndjsons/cloud.ndjson',
        'data-ndjsons/duck.ndjson',
        'data-ndjsons/plane.ndjson'
    ]

    (train_data, train_labels), (val_data, val_labels), class_names = load_and_preprocess_data_from_ndjson(
        file_paths, 
        num_classes=5, 
        max_samples_per_class=5000,
        test_size=0.2,
        random_state=42
    )

    model = build_model(input_shape=(36,36,1), num_classes=5, dropout_rate=0.3)

    history_cnn = model.fit(
        train_data, train_labels,
        validation_data=(val_data, val_labels),
        epochs=5,
        batch_size=32
    )

    save_dir = 'classification-cnn'
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'cnn_quickdraw_model.h5')
    model.save(model_path)
    np.save(os.path.join(save_dir, 'label_classes.npy'), class_names)
    
    np.save(os.path.join(save_dir, 'history_cnn.npy'), history_cnn.history)
    
    np.save(os.path.join(save_dir, 'val_data.npy'), val_data)
    np.save(os.path.join(save_dir, 'val_labels.npy'), val_labels)
    
    print("CNN Modell, Klassen, Trainings-Historie und Validierungsdaten erfolgreich gespeichert!")
