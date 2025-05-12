import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tifffile as tiff
import pandas as pd
from scipy.stats import skew, kurtosis, moment
import matplotlib.image as mpimg
import cv2
from skimage.feature import local_binary_pattern,  graycomatrix, graycoprops
from skimage.filters import gabor
from skimage import measure
from openpyxl.workbook import Workbook

# Ruta del conjunto de imágenes
path_images = Path("Colon_CVC/Kather_texture_2016_image_tiles_5000")

data = []

# Función para procesar una imagen y calcular los descriptores
def extract_features(image_path, class_name):
    # Cargar la imagen
    img = tiff.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # --- LBP ---
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp_image = np.uint8(lbp)
    lbp_image = cv2.normalize(lbp_image, None, 0, 255, cv2.NORM_MINMAX)
    
    """# --- Gabor Filters ---
    frequencies = [0.1, 0.2, 0.3]  # 3 frecuencias a probar
    gabor_feats = []
    gabor_images = []

    for freq in frequencies:
        filt_real, filt_imag = gabor(gray, frequency=freq)
        gabor_feats.append(filt_real)
        gabor_images.append(filt_real)

    # --- GLCM ---
    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)

    # Propiedades de GLCM: contrast, correlation, energy, homogeneity
    glcm_contrast = graycoprops(glcm, 'contrast')
    glcm_homogeneity = graycoprops(glcm, 'homogeneity')"""

    # --- Momentos ---
    mean = gray.mean()
    skewness = skew(gray.flatten())
    kurtosis_v = kurtosis(gray.flatten())
    moment_5 = moment(gray.flatten(), 5)
    moments = [moment(gray.flatten(), n) for n in range(2, 12)]
    
    # --- Almacenar características ---
    image_data = {
        "class name": class_name,
        "mean color": mean,
        "skewness": skewness,
        "kurtosis": kurtosis_v,
        "5th moment": moment_5,
        "contrast_perceptual": gray.std(),
        "roughness": np.mean(np.abs(np.diff(gray.astype(float), axis=0))) + np.mean(np.abs(np.diff(gray.astype(float), axis=1))),
    }

    # Agregar momentos 2-11
    for i, m in enumerate(moments, start=2):
        image_data[f"moment_{i}"] = m

    # Agregar LBP al diccionario
    for i, val in enumerate(lbp_image.flatten()):
        image_data[f"lbp_{i}"] = val

    """# Agregar GLCM al diccionario
    image_data['glcm_contrast'] = glcm_contrast.mean()
    image_data['glcm_homogeneity'] = glcm_homogeneity.mean()

    # Agregar Gabor features al diccionario
    for i, val in enumerate(gabor_feats):
        image_data[f"gabor_{i}"] = val.mean()"""

    return image_data

# Procesar todas las imágenes en el directorio
for folder in path_images.iterdir():
    if folder.is_dir():
        class_name = folder.name
        for file in folder.iterdir():
            if file.suffix == '.tif':  # Asegúrate de que es un archivo de imagen
                features = extract_features(file, class_name)
                data.append(features)

# Crear DataFrame con todos los descriptores
df = pd.DataFrame(data)

# Guardar el DataFrame en un archivo Excel
df.to_csv("stats_all_descriptors_output.csv", index=False)

# Mostrar los primeros resultados
print(df.head())




