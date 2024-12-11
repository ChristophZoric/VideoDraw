from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import numpy as np

def preprocess_sequences(sequences, max_length=128):
    normalized_sequences = [np.array(seq) / 255.0 for seq in sequences]
    padded_sequences = pad_sequences(normalized_sequences, maxlen=max_length, dtype='float32', padding='post', truncating='post')
    return padded_sequences

class Predictor:
    def __init__(self, model_path='classification-crnn/crnn_quickdraw_model.h5', label_path='classification-crnn/label_classes.npy'):
        self.model = load_model(model_path)
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.load(label_path)

    def predict(self, drawing_sequence):
        processed_sequence = preprocess_sequences([drawing_sequence], max_length=128)
        prediction = self.model.predict(processed_sequence)
        predicted_class = np.argmax(prediction[0])
        return self.label_encoder.inverse_transform([predicted_class])[0]

if __name__ == "__main__":
    predictor = Predictor(
        model_path='classification-crnn/crnn_quickdraw_model.h5',
        label_path='classification-crnn/label_classes.npy'
    )

    new_drawing_sequence = [
        [[10, 30, 50, 70], [70, 50, 50, 70]], 
        [[20, 30, 30, 20], [80, 80, 90, 90]],  
        [[50, 60, 60, 50], [80, 80, 90, 90]],  
        [[15, 25, 25, 15], [60, 60, 70, 70]],  
        [[45, 55, 55, 45], [60, 60, 70, 70]]   
    ]

    flat_sequence = []
    for stroke in new_drawing_sequence:
        flat_sequence.extend(list(zip(stroke[0], stroke[1])))

    predicted_class = predictor.predict(flat_sequence)
    print(f"Vorhergesagte Klasse: {predicted_class}")
