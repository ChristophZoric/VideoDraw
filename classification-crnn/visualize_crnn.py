# Datei: visualize_crnn.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

def plot_training_history(history_dict, output_path='crnn_training_history.png'):
    """Speichert den Accuracy- und Loss-Verlauf als Bild (PNG)."""
    train_loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    train_acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Val Accuracy')
    plt.title('CRNN Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.title('CRNN Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)  
    plt.close()               

def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix', output_path='crnn_confusion_matrix.png'):
    """Erzeugt eine Konfusionsmatrix und speichert sie als Bild (PNG)."""
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
    crnn_model_path = 'classification-crnn/crnn_quickdraw_model.h5'
    val_data_path   = 'classification-crnn/val_data.npy'
    val_labels_path = 'classification-crnn/val_labels.npy'
    history_path    = 'classification-crnn/history_crnn.npy'
    class_names_path = 'classification-crnn/label_classes.npy'

    model = load_model(crnn_model_path)
    class_names = np.load(class_names_path, allow_pickle=True)

    val_data = np.load(val_data_path)
    val_labels = np.load(val_labels_path)
    y_true = np.argmax(val_labels, axis=1)

    history_crnn = np.load(history_path, allow_pickle=True).item()

    plot_training_history(history_crnn, output_path='crnn_training_history.png')

    y_pred_probs = model.predict(val_data)
    y_pred = np.argmax(y_pred_probs, axis=1)

    from sklearn.metrics import classification_report
    print("Classification Report:\n", classification_report(y_true, y_pred, target_names=class_names))

    plot_confusion_matrix(y_true, y_pred, class_names,
                          title="CRNN Confusion Matrix",
                          output_path='crnn_confusion_matrix.png')

    print("Plots wurden gespeichert (crnn_training_history.png und crnn_confusion_matrix.png).")
