# Datei: visualize_cnn.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

def plot_training_history(history_dict, output_path='cnn_training_history.png'):
    """Speichert den Accuracy- und Loss-Verlauf als Bild."""
    train_loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    train_acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Val Accuracy')
    plt.title('CNN Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.title('CNN Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()                

def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix', output_path='cnn_confusion_matrix.png'):
    """Erzeugt eine Konfusionsmatrix und speichert sie als Bild."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=class_names, 
                yticklabels=class_names, 
                cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(output_path)   
    plt.close()                

if __name__ == "__main__":
    cnn_model_path = 'classification-cnn/cnn_quickdraw_model.h5'
    val_data_path  = 'classification-cnn/val_data.npy'
    val_labels_path= 'classification-cnn/val_labels.npy'
    history_path   = 'classification-cnn/history_cnn.npy'
    class_names_path = 'classification-cnn/label_classes.npy'

    model = load_model(cnn_model_path)
    class_names = np.load(class_names_path, allow_pickle=True)

    val_data = np.load(val_data_path)
    val_labels = np.load(val_labels_path) 
    y_true = np.argmax(val_labels, axis=1)

    history_cnn = np.load(history_path, allow_pickle=True).item()

    plot_training_history(history_cnn, output_path='cnn_training_history.png')

    y_pred_probs = model.predict(val_data)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("Classification Report:\n", classification_report(y_true, y_pred, target_names=class_names))

    plot_confusion_matrix(y_true, y_pred, class_names, 
                          title="CNN Confusion Matrix", 
                          output_path='cnn_confusion_matrix.png')

    print("Plots wurden gespeichert (cnn_training_history.png und cnn_confusion_matrix.png).")
