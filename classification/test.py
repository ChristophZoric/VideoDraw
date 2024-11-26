import numpy as np
from tensorflow.keras.models import load_model


class Predictor:
    @staticmethod
    def load_and_test_coordinates(np_file_path, model_path='classification/saved_model/coordinate_model.h5'):
        print("entered Predictor")
        model = load_model(model_path)

        # Koordinatenbild laden und in das Modell-Format umwandeln
        class_data = np.load(np_file_path)
        coordinates = class_data.reshape(
            (1, 36, 36, 1))  # Anpassung an 64x64 Pixel

        # Vorhersage
        prediction = model.predict(coordinates)
        predicted_class = np.argmax(prediction[0])

        # Klassennamen
        class_names = ['basketball', 'car', 'plane']
        print(f'Predicted class: {class_names[predicted_class]}')
