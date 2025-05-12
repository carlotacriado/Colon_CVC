import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os


#=====================
# Descriptor HSV
#=====================

def calculate_histograms(h, s, v, bins=32):
    hist_hue = cv2.calcHist([h], [0], None, [bins], [0, 256])
    hist_saturation = cv2.calcHist([s], [0], None, [bins], [0, 256])
    hist_value = cv2.calcHist([v], [0], None, [bins], [0, 256])

    hist_hue = hist_hue / hist_hue.sum()
    hist_saturation = hist_saturation / hist_saturation.sum()
    hist_value = hist_value / hist_value.sum()

    return np.concatenate([hist_hue.flatten(), hist_saturation.flatten(), hist_value.flatten()])


def create_heatmap(hsv, img):
    hue_min = 130   
    hue_max = 170  
    saturation_min = 30  
    saturation_max = 180  
    value_min = 0  
    value_max = 150  

    lower_bound = np.array([hue_min, saturation_min, value_min])
    upper_bound = np.array([hue_max, saturation_max, value_max])

    mask = cv2.inRange(hsv, lower_bound, upper_bound) #InRange pone a 255 los valores mas grandes del upperbound y a 0 si está por debajo del lower bound

    highlighted_image = cv2.bitwise_and(img, img, mask=mask)

    # Crear el heatmap
    heatmap = cv2.applyColorMap(highlighted_image, cv2.COLORMAP_JET)

    #heatmap_flatten = heatmap.flatten()
    return heatmap


img_path = r'C:\Users\ccriado\Desktop\CVC\Colon_CVC_rep\Colon_CVC\Kather_texture_2016_image_tiles_5000\01_TUMOR\1D8F_CRC-Prim-HE-02_007b.tif_Row_151_Col_1.tif'
img = cv2.imread(img_path, cv2.IMREAD_COLOR)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
heatmap = create_heatmap(hsv,img)


# Mostrar el heatmap usando Matplotlib
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
overlay = cv2.addWeighted(image_rgb, 0.8, heatmap_rgb, 0.2, 0)

# Mostrar la imagen con el heatmap superpuesto
plt.figure(figsize=(10, 6))
plt.imshow(overlay)
plt.title('Heatmap Superpuesto sobre la Imagen Original')
plt.axis('off')  # Desactivar los ejes
plt.show()



