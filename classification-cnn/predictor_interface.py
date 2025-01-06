import ast
from .predictor import Predictor as CNNPredictor


def load_annotations_from_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read().strip()
    annotations = ast.literal_eval(content)
    return annotations


if __name__ == "__main__":
    file_path = 'classification/annotations_data.txt'

    annotations = load_annotations_from_file(file_path)

    predictor = CNNPredictor(
        model_path='classification-cnn/cnn_quickdraw_model.h5',
        label_path='classification-cnn/label_classes.npy'
    )

    flat_sequence = []
    for stroke in annotations:
        flat_sequence.extend(stroke)

    result = predictor.predict(flat_sequence)
    print("CNN Vorhersage:", result)
