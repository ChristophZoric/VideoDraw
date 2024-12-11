import ast
from predictor import Predictor as CRNNPredictor

def load_annotations_from_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read().strip()
    annotations = ast.literal_eval(content)
    return annotations

if __name__ == "__main__":
    file_path = 'classification/annotations_data.txt'

    annotations = load_annotations_from_file(file_path)

    predictor = CRNNPredictor(
        model_path='classification-crnn/crnn_quickdraw_model.h5',
        label_path='classification-crnn/label_classes.npy'
    )

    result = predictor.predict(annotations[0])  
    flat_sequence = []
    for stroke in annotations:
        flat_sequence.extend(stroke)
    
    result = predictor.predict(flat_sequence)
    print("CRNN Vorhersage:", result)
