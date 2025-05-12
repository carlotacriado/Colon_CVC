import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
from HSV_Exp import * 

# leer todas las imagenes

base_dir = r'C:\Users\ccriado\Desktop\CVC\Colon_CVC_rep\Colon_CVC\Kather_texture_2016_image_tiles_5000'  # Ajusta la ruta según sea necesario

X = []  # Vectores de características
y = []  # Etiquetas de clase

class_folders = sorted(os.listdir(base_dir))  
benigno_classes = [class_folders[0], class_folders[1]] # Clases que suponemos como benignas


for class_folder in class_folders:
    class_path = os.path.join(base_dir, class_folder)
    
    if os.path.isdir(class_path):
        if class_folder in benigno_classes:
            label = 0  # Benigno
        else:
            label = 1  # El resto

        # De momento BENIGNO vs MALIGNO (las ultimas dos clases asumimos que son las benignas)

        for filename in os.listdir(class_path):
            img_path = os.path.join(class_path, filename)
            
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is not None:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv)
                
                # Calculamos el vector de características (histogramas)
                feature_vector = create_heatmap(hsv, img)
                
                X.append(feature_vector)
                y.append(label)

X = np.array(X) # vector de caracteristicas (h s v)
y = np.array(y) #clase a la que pertenecen 

# Verificar que los datos se han cargado correctamente
print(f"Características extraídas: {X.shape[0]} imágenes, {X.shape[1]} características por imagen")


# -------------- SVM SENCILLO ----------------
# Dividir el conjunto de datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del clasificador: {accuracy * 100:.2f}%")

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))