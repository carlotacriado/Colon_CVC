import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

# Cargar la imagen
#img = cv2.imread(r'C:\Users\ccriado\Desktop\CVC\Colon_CVC_rep\Colon_CVC\Kather_texture_2016_image_tiles_5000\02_STROMA\10DE3_CRC-Prim-HE-02_015.tif_Row_451_Col_2401.tif', cv2.IMREAD_COLOR)

"""def calculate_histograms(h, s, v, bins=32):
    hist_hue = cv2.calcHist([h], [0], None, [bins], [0, 256])
    hist_saturation = cv2.calcHist([s], [0], None, [bins], [0, 256])
    hist_value = cv2.calcHist([v], [0], None, [bins], [0, 256])

    hist_hue = hist_hue / hist_hue.sum()
    hist_saturation = hist_saturation / hist_saturation.sum()
    hist_value = hist_value / hist_value.sum()

    return np.concatenate([hist_hue.flatten(), hist_saturation.flatten(), hist_value.flatten()])"""

#hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Extraemos Hue, Saturación y Valor
#h, s, v = cv2.split(hsv)

#result = calculate_histograms(h, s, v)
# Definir rangos de interés en los componentes Hue, Saturación y Valor (ajusta los valores según tus necesidades)
# Ejemplo de valores para zonas de cáncer (puedes ajustar estos rangos)
def create_heatmap(hsv, img):
    hue_min = 130   # Tono mínimo (púrpura)
    hue_max = 170  # Tono máximo
    saturation_min = 30  # Saturación mínima
    saturation_max = 180  # Saturación máxima
    value_min = 0  # Valor mínimo (para zonas más oscuras)
    value_max = 150  # Valor máximo (para zonas más brillantes)

    # Crear una máscara basada en estos rangos
    lower_bound = np.array([hue_min, saturation_min, value_min])
    upper_bound = np.array([hue_max, saturation_max, value_max])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Aplicar máscara a la imagen original para extraer las zonas relevantes
    highlighted_image = cv2.bitwise_and(img, img, mask=mask)

    # Crear el heatmap
    # Convertir la imagen con las zonas resaltadas a un formato de heatmap (con colores más cálidos)
    heatmap = cv2.applyColorMap(highlighted_image, cv2.COLORMAP_JET)

    heatmap_flatten = heatmap.flatten()
    return heatmap_flatten()


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


"""# Mostrar el heatmap usando Matplotlib
plt.figure(figsize=(10, 6))
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))  # Convertir BGR a RGB para Matplotlib
plt.title('Heatmap de Zonas Importantes (Cáncer)')
plt.axis('off')  # Desactivar los ejes

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convertir BGR a RGB para Matplotlib
plt.title('Real image')
plt.axis('off')  # Desactivar los ejes
plt.show()

# Convertir la imagen a formato RGB (porque OpenCV la carga en BGR)
image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convertir el heatmap a RGB
heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

# Superponer el heatmap en la imagen original con un nivel de transparencia (opacidad)
overlay = cv2.addWeighted(image_rgb, 0.7, heatmap_rgb, 0.3, 0)

# Mostrar la imagen con el heatmap superpuesto
plt.figure(figsize=(10, 6))
plt.imshow(overlay)
plt.title('Heatmap Superpuesto sobre la Imagen Original')
plt.axis('off')  # Desactivar los ejes
plt.show()"""