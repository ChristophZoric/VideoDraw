from classification.convert_np import convert_annotations
from classification.predictor import Predictor
# Muhammeds Werk

def main(annotation_file='classification/annotations_data.txt'):
    print("\n Starting prediction pipeline...")

    converted_data_path = convert_annotations(annotation_file)
    print(f"Converted annotations saved at: {converted_data_path}")

    predictor = Predictor()
    prediction = predictor.predict(converted_data_path)

    print(f"Predicted class: {prediction}")
    return prediction


if __name__ == "__main__":
    main()
