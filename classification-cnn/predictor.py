from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import numpy as np
from model import rasterize_sequence

class Predictor:
    def __init__(self, model_path='classification-cnn/cnn_quickdraw_model.h5', label_path='classification-cnn/label_classes.npy'):
        self.model = load_model(model_path)
        self.class_names = np.load(label_path, allow_pickle=True)
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = self.class_names

    def predict(self, drawing_sequence):
        # drawing_sequence ist eine Liste von (x, y)-Paaren
        # Rasterize
        img = rasterize_sequence(drawing_sequence, img_size=36)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=-1)  # (36,36) -> (36,36,1)
        img = np.expand_dims(img, axis=0)   # (36,36,1) -> (1,36,36,1)

        prediction = self.model.predict(img)
        predicted_class = np.argmax(prediction[0])
        return self.label_encoder.inverse_transform([predicted_class])[0]

# Testcode
if __name__ == "__main__":
    predictor = Predictor(
        model_path='classification-cnn/cnn_quickdraw_model.h5',
        label_path='classification-cnn/label_classes.npy'
    )

    # Beispiel-Zeichnung (ähnlich wie bei CRNN, nur hier als (x,y)-Paare)
    new_drawing_sequence = [
        ([10, 30, 50, 70], [70, 50, 50, 70]),
        ([20, 30, 30, 20], [80, 80, 90, 90]),
        ([50, 60, 60, 50], [80, 80, 90, 90]),
        ([15, 25, 25, 15], [60, 60, 70, 70]),
        ([45, 55, 55, 45], [60, 60, 70, 70])
    ]

    # In flache (x,y)-Paare umwandeln
    flat_sequence = []
    for stroke in new_drawing_sequence:
        xs, ys = stroke
        for x, y in zip(xs, ys):
            flat_sequence.append((x,y))

    predicted_class = predictor.predict(flat_sequence)
    print(f"Vorhergesagte Klasse: {predicted_class}")
