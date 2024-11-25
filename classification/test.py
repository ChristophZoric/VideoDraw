import numpy as np
from tensorflow.keras.models import load_model

class Predictor:
    @staticmethod
    def load_and_test_coordinates(np_file_path, model_path='/home/rafaelcanete/Uni/3Semester/DeepLearning/VideoDraw/src/saved_model/coordinate_model.h5'):
        model = load_model(model_path)

        # Koordinatenbild laden und in das Modell-Format umwandeln
        class_data = np.load(np_file_path)
        coordinates = class_data.reshape((1, 28, 28, 1))  # Passendes Format fÃ¼r das Modell

        # Vorhersage
        prediction = model.predict(coordinates)
        predicted_class = np.argmax(prediction[0])

        # Klassennamen
        class_names = ['plane', 'basketball', 'car']
        print(f'Predicted class: {class_names[predicted_class]}')

    # Testen des gespeicherten Arrays   
    #load_and_test_coordinates('/home/mooyil/vscode-workspace/koordinaten_klassifikation/predict_data/test_image.npy')