import numpy as np
file_path = 'data/car.npy'  # Ersetzen Sie dies durch den Pfad zu einer Ihrer Dateien
data = np.load(file_path)
print(data.shape)
print(data[0])
