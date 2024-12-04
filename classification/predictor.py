import numpy as np
from tensorflow.keras.models import load_model


class Predictor:
    def __init__(self, model_path='classification/saved_model/coordinate_model.h5'):
        self.model = load_model(model_path)
        self.class_names = ['basketball', 'car', 'cloud', 'duck', 'plane']

    def predict(self, np_file_path):
        print("entered Predictor")

        # Koordinatenbild laden und in das Modell-Format umwandeln
        data = np.load(np_file_path)
        coordinates = data.reshape(
            (1, 36, 36, 1))  # Anpassung an 64x64 Pixel

        # Vorhersage
                
        predictions = self.model.predict(coordinates)
        predicted_class = np.argmax(predictions[0])
        
        if predictions[0][predicted_class] < 0.5:
            return "Other"

        return self.class_names[predicted_class]
