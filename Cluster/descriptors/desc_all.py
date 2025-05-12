import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Función para calcular los descriptores
def calculate_histograms(h, s, v, bins=32):
    hist_hue = cv2.calcHist([h], [0], None, [bins], [0, 256])
    hist_saturation = cv2.calcHist([s], [0], None, [bins], [0, 256])
    hist_value = cv2.calcHist([v], [0], None, [bins], [0, 256])

    hist_hue = hist_hue / hist_hue.sum()
    hist_saturation = hist_saturation / hist_saturation.sum()
    hist_value = hist_value / hist_value.sum()

    return np.concatenate([hist_hue.flatten(), hist_saturation.flatten(), hist_value.flatten()])


base_dir = r'C:\Users\ccriado\Desktop\CVC\Colon_CVC_rep\Colon_CVC\Kather_texture_2016_image_tiles_5000'  # Ajusta la ruta según sea necesario

X = []  # Vectores de características
y = []  # Etiquetas de clase

class_folders = sorted(os.listdir(base_dir))  
benigno_classes = [class_folders[-2], class_folders[-1]] # Clases que suponemos como benignas


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
                feature_vector = calculate_histograms(h, s, v)
                
                X.append(feature_vector)
                y.append(label)

X = np.array(X)
y = np.array(y)

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





# ----------- Que caracteristicas son mas importantes? -------------
# Obtener los coeficientes del clasificador SVM
coef = clf.coef_.flatten()  # Aplanamos el coeficiente para que sea un vector unidimensional

# Visualizar la importancia de cada característica (abs valor de los coeficientes)
feature_importance = np.abs(coef)

# Graficar la importancia de las características
plt.figure(figsize=(12, 6))
plt.bar(range(len(feature_importance)), feature_importance)
plt.title("Importancia de las características (Hue, Saturation, Value)")
plt.xlabel("Características (Bins de los histogramas de H, S, V)")
plt.ylabel("Importancia (Valor absoluto de los coeficientes)")
plt.show()



# ------------ Testear una imagen grande ----------
# Función para dividir la imagen en patches de tamaño específico
def extract_patches(img, patch_size=150):
    patches = []
    h, w, _ = img.shape
    
    # Iteramos sobre la imagen en pasos del tamaño de patch
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = img[i:i+patch_size, j:j+patch_size]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patches.append(patch)
    
    return patches

img_path = r'C:\Users\ccriado\Desktop\CVC\Colon_CVC_rep\Colon_CVC\Kather_texture_2016_larger_images_10\CRC-Prim-HE-09_APPLICATION.tif'  # Cambiar a la ruta de tu imagen grande
img = cv2.imread(img_path)
patches = extract_patches(img)

# Mostrar algunos patches
plt.figure(figsize=(12, 8))
for i, patch in enumerate(patches[:5]):  
    patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 5, i+1)
    plt.imshow(patch_rgb)
    plt.title(f"Patch {i+1}")
    plt.axis('off')

plt.tight_layout()
plt.show()

patch_features = []
for patch in patches:
    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_patch)
    feature_vector = calculate_histograms(h, s, v)
    patch_features.append(feature_vector)

patch_features = np.array(patch_features)

predictions = clf.predict(patch_features)

for i, pred in enumerate(predictions[:5]):
    print(f"Patch {i+1} - Predicción: {'No cáncer (0)' if pred == 0 else 'Cáncer (1)'}")


