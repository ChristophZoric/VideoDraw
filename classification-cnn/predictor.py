from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import numpy as np
import ast
from .model import rasterize_sequence


class Predictor:
    def __init__(self, model_path='classification-cnn/cnn_quickdraw_model.h5', label_path='classification-cnn/label_classes.npy'):
        self.model = load_model(model_path)
        self.class_names = np.load(label_path, allow_pickle=True)
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = self.class_names

    def predict(self, drawing_sequence):
        # Rasterize
        img = rasterize_sequence(drawing_sequence, img_size=36)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        prediction = self.model.predict(img)
        predicted_class = np.argmax(prediction[0])
        return self.label_encoder.inverse_transform([predicted_class])[0]


def load_annotations_from_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read().strip()
    annotations = ast.literal_eval(content)
    return annotations


if __name__ == "__main__":
    file_path = 'classification/annotations_data.txt'

    annotations = load_annotations_from_file(file_path)

    predictor = Predictor(
        model_path='classification-cnn/cnn_quickdraw_model.h5',
        label_path='classification-cnn/label_classes.npy'
    )

    flat_sequence = []
    for stroke in annotations:
        flat_sequence.extend(stroke)

    result = predictor.predict(flat_sequence)
    print("CNN Vorhersage:", result)
