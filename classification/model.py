from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from preprocess import load_and_preprocess_data

def build_model(input_shape=(28, 28, 1), num_classes=3):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5), 
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    data_dir = 'data'  # Der Ordner mit den .npy-Dateien
    (train_data, train_labels), (val_data, val_labels) = load_and_preprocess_data(data_dir)

    
    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1)
    datagen.fit(train_data)

    model = build_model()
    model.fit(datagen.flow(train_data, train_labels, batch_size=32),
              validation_data=(val_data, val_labels),
              epochs=1)
    model.save('classification/saved_model/coordinate_model.h5')