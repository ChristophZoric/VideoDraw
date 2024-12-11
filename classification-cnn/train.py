from model import build_model, load_and_preprocess_data_from_ndjson
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

    (train_data, train_labels), (val_data, val_labels), class_names = load_and_preprocess_data_from_ndjson(
        file_paths, num_classes=5, max_samples_per_class=5000)

    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1)
    datagen.fit(train_data)

    model = build_model(input_shape=(36,36,1), num_classes=5)
    model.fit(datagen.flow(train_data, train_labels, batch_size=32),
              validation_data=(val_data, val_labels),
              epochs=5)

    # Speicher das Modell und die Klassen
    save_dir = 'classification-cnn'
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'cnn_quickdraw_model.h5')
    model.save(model_path)
    np.save(os.path.join(save_dir, 'label_classes.npy'), class_names)
    print("Modell und Klassen erfolgreich gespeichert!")
