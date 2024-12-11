import ast
from predictor import Predictor as CRNNPredictor

def load_annotations_from_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read().strip()
    annotations = ast.literal_eval(content)
    return annotations

if __name__ == "__main__":
    # Passe den Pfad hier an, wenn die Datei woanders liegt.
    # Angenommen, die Textdatei liegt eine Ebene höher im Projektverzeichnis:
    file_path = 'classification/annotations_data.txt'

    annotations = load_annotations_from_file(file_path)

    # Erstelle eine Instanz des CRNN-Predictors
    predictor = CRNNPredictor(
        model_path='classification-crnn/crnn_quickdraw_model.h5',
        label_path='classification-crnn/label_classes.npy'
    )

    result = predictor.predict(annotations[0])  # annotations[0] ist der erste Stroke oder du nimmst direkt annotations als Ganzes
    # Achtung: Dein CRNN-Predictor erwartet eine flache Liste, also Strokes zusammenfügen
    flat_sequence = []
    for stroke in annotations:
        flat_sequence.extend(stroke)
    
    result = predictor.predict(flat_sequence)
    print("CRNN Vorhersage:", result)
