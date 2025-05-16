import os
import cv2
import numpy as np
import re
import pandas as pd
from skimage.feature import local_binary_pattern
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- Parámetros LBP ---
radius = 2
n_points = 8 * radius
method = 'uniform'

# --- Función para extraer características ---
def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method)
    hist_lbp, _ = np.histogram(lbp.ravel(),
                               bins=np.arange(0, n_points + 3),
                               range=(0, n_points + 2), density=True)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()
    return np.concatenate([hist_lbp, hist_h, hist_s, hist_v])

# --- Función para extraer ID imagen ---
def extract_image_id(filename):
    match = re.search(r'(CRC-Prim-HE-\d+_\d+\.tif)', filename)
    return match.group(1) if match else None

# --- Cargar datos ---
base_path = '/home/ccriado/CVC/Colon_CVC/Cluster/Kather_texture_2016_image_tiles_5000'
class_names = sorted(os.listdir(base_path))

grouped_by_image = []
for label, class_name in enumerate(class_names):
    class_dir = os.path.join(base_path, class_name)
    for fname in os.listdir(class_dir):
        if fname.endswith(".tif"):
            img_path = os.path.join(class_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
            features = extract_features(img)
            image_id = extract_image_id(fname)
            if image_id:
                grouped_by_image.append((features, label, image_id))

df = pd.DataFrame(grouped_by_image, columns=['features', 'label', 'image_id'])
df['class_name'] = df['label'].map({i: name for i, name in enumerate(class_names)})


