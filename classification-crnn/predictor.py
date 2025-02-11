from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import ast
import numpy as np


def preprocess_sequences(sequences, max_length=128):
    normalized_sequences = [np.array(seq) / 255.0 for seq in sequences]
    padded_sequences = pad_sequences(
        normalized_sequences, maxlen=max_length, dtype='float32', padding='post', truncating='post')
    return padded_sequences


class Predictor:
    def __init__(self, model_path='classification-crnn/crnn_quickdraw_model.h5', label_path='classification-crnn/label_classes.npy'):
        self.model = load_model(model_path)
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.load(label_path)

    def predict(self, drawing_sequence):
        processed_sequence = preprocess_sequences(
            [drawing_sequence], max_length=128)
        prediction = self.model.predict(processed_sequence)
        predicted_class = np.argmax(prediction[0])
        return self.label_encoder.inverse_transform([predicted_class])[0]


def load_annotations_from_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read().strip()
    annotations = ast.literal_eval(content)
    return annotations


if __name__ == "__main__":
    file_path = 'annotations_data.txt'

    annotations = load_annotations_from_file(file_path)

    predictor = Predictor(
        model_path='classification-crnn/crnn_quickdraw_model.h5',
        label_path='classification-crnn/label_classes.npy'
    )

    result = predictor.predict(annotations[0])
    flat_sequence = []
    for stroke in annotations:
        flat_sequence.extend(stroke)

    result = predictor.predict(flat_sequence)
    print("CRNN Vorhersage:", result)
