import os
import cv2
import numpy as np
import re
import pandas as pd
from skimage.feature import local_binary_pattern
from sklearn.model_selection import StratifiedGroupKFold, train_test_split,GridSearchCV
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

# --- Función de aumento simple ---
def augment_image(img):
    rows, cols = img.shape[:2]
    imgs = [img]  # Imagen original
    # Rotar +15 grados
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 15, 1)
    img_rot = cv2.warpAffine(img, M, (cols, rows))
    imgs.append(img_rot)
    # Flip horizontal
    img_flip = cv2.flip(img, 1)
    imgs.append(img_flip)
    return imgs

# --- Cargar y aumentar datos ---
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
            image_id = extract_image_id(fname)
            if not image_id:
                continue
            augmented_imgs = augment_image(img)
            for aug_img in augmented_imgs:
                features = extract_features(aug_img)
                grouped_by_image.append((features, label, image_id))

grouped_df = pd.DataFrame(grouped_by_image, columns=['features', 'label', 'image_id'])
label_to_class = {i: name for i, name in enumerate(class_names)}
grouped_df['class_name'] = grouped_df['label'].map(label_to_class)

# --- Separar clases minoritarias ---
minor_classes = ['07_ADIPOSE', '08_EMPTY']
df_major = grouped_df[~grouped_df['class_name'].isin(minor_classes)]
df_minor = grouped_df[grouped_df['class_name'].isin(minor_classes)]

minor_feats = np.stack(df_minor['features'].values)
minor_labels = df_minor['label'].values
minor_X_train, minor_X_test, minor_y_train, minor_y_test = train_test_split(
    minor_feats, minor_labels, test_size=0.2, stratify=minor_labels, random_state=42
)

features = np.stack(df_major['features'].values)
labels = df_major['label'].values
groups = df_major['image_id'].values

param_grid = {'C': [10], 'gamma': [0.1]}
results = []

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(features, labels, groups)):
    X_train = np.concatenate([features[train_idx], minor_X_train])
    y_train = np.concatenate([labels[train_idx], minor_y_train])
    X_test = np.concatenate([features[test_idx], minor_X_test])
    y_test = np.concatenate([labels[test_idx], minor_y_test])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=25)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
    grid.fit(X_train_pca, y_train)
    y_pred = grid.predict(X_test_pca)

    report = classification_report(
        y_test, y_pred, labels=np.arange(len(class_names)), target_names=class_names, output_dict=True, zero_division=0
    )
    for class_name in class_names:
        metrics = report.get(class_name, {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0})
        results.append({
            'Fold': fold_idx + 1,
            'Class': class_name,
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1-score'],
            'Support': metrics['support'],
            'Setup': 'StratifiedGroupKFold + MinorSplit + Augmentation'
        })

df_results = pd.DataFrame(results)
output_path = '/home/ccriado/CVC/Colon_CVC/Cluster/fold_results_augmented.csv'
df_results.to_csv(output_path, index=False)
print("Resultados guardados en", output_path)
