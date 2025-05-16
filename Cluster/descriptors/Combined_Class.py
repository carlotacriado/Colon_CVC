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

# --- Separar minoritarias y mayoritarias ---
minor_classes = ['07_ADIPOSE', '08_EMPTY']
df_minor = df[df['class_name'].isin(minor_classes)].copy()
df_major = df[~df['class_name'].isin(minor_classes)].copy()

# --- Mapas para clf1 y clf2 ---
label_map_clf1 = {
    0: 0,                # 01_TUMOR
    1: 1, 2: 1, 3: 1,    # 02_STROMA, 03_COMPLEX, 04_LYMPHO juntas
    4: 2,                # 05_DEBRIS
    5: 3                 # 06_MUCOSA
}
minor_map_clf1 = {
    6: 4,                # 07_ADIPOSE
    7: 5                 # 08_EMPTY
}
clf2_map = {
    1: 0,  # 02_STROMA
    2: 1,  # 03_COMPLEX
    3: 2   # 04_LYMPHO
}

# --- Concatenar todo para hacer StratifiedGroupKFold juntos ---
features_minor = np.stack(df_minor['features'].values)
labels_minor = df_minor['label'].values
groups_minor = np.array(['minor'] * len(df_minor))  # asignar grupo único para minoritarias

features_major = np.stack(df_major['features'].values)
labels_major = df_major['label'].values
groups_major = df_major['image_id'].values

features_all = np.concatenate([features_major, features_minor])
labels_all = np.concatenate([labels_major, labels_minor])
groups_all = np.concatenate([groups_major, groups_minor])

# Lista completa de nombres (ordenados por etiqueta original)
all_target_names = [
    "01_TUMOR",
    "02_03_04_STROMA_COMPLEX_LYMPHO",
    "05_DEBRIS",
    "06_MUCOSA",
    "07_ADIPOSE",
    "08_EMPTY"
]

# --- Preparar el split con grupos para todo el dataset ---
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(features_all, labels_all, groups_all)):
    X_train = features_all[train_idx]
    y_train_raw = labels_all[train_idx]
    X_test = features_all[test_idx]
    y_test_raw = labels_all[test_idx]
    print("Soporte por clase en test fold", fold_idx+1)
    unique, counts = np.unique(y_test_raw, return_counts=True)
    print(dict(zip(unique, counts)))


    # Mapear para clf1 (tanto mayoritarias como minoritarias)
    y_train_clf1 = pd.Series(y_train_raw).map({**label_map_clf1, **minor_map_clf1}).values
    y_test_clf1 = pd.Series(y_test_raw).map({**label_map_clf1, **minor_map_clf1}).values

    # Filtrar posibles NaNs (por si acaso)
    mask_train = ~pd.isna(y_train_clf1)
    mask_test = ~pd.isna(y_test_clf1)

    X_train = X_train[mask_train]
    y_train_clf1 = y_train_clf1[mask_train].astype(int)
    X_test = X_test[mask_test]
    y_test_clf1 = y_test_clf1[mask_test].astype(int)

    # Escalado y PCA para clf1
    scaler1 = StandardScaler()
    X_train_scaled1 = scaler1.fit_transform(X_train)
    X_test_scaled1 = scaler1.transform(X_test)

    pca1 = PCA(n_components=25)
    X_train_pca1 = pca1.fit_transform(X_train_scaled1)
    X_test_pca1 = pca1.transform(X_test_scaled1)

    clf1 = SVC(C=10, gamma=0.1, kernel='rbf')
    clf1.fit(X_train_pca1, y_train_clf1)

    # Preparar clf2 (solo para muestras etiquetadas como grupo 1 en clf1)
    idx_train_clf2 = np.where(y_train_clf1 == 1)[0]
    idx_test_clf2 = np.where(y_test_clf1 == 1)[0]

    X_train_clf2 = X_train[idx_train_clf2]
    X_test_clf2 = X_test[idx_test_clf2]

    y_train_raw_clf2 = y_train_raw[mask_train][idx_train_clf2]
    y_test_raw_clf2 = y_test_raw[mask_test][idx_test_clf2]

    y_train_clf2 = pd.Series(y_train_raw_clf2).map(clf2_map).values
    y_test_clf2 = pd.Series(y_test_raw_clf2).map(clf2_map).values

    mask_train_clf2 = ~pd.isna(y_train_clf2)
    mask_test_clf2 = ~pd.isna(y_test_clf2)

    X_train_clf2 = X_train_clf2[mask_train_clf2]
    y_train_clf2 = y_train_clf2[mask_train_clf2].astype(int)
    X_test_clf2 = X_test_clf2[mask_test_clf2]
    y_test_clf2 = y_test_clf2[mask_test_clf2].astype(int)

    # Escalado y PCA para clf2
    scaler2 = StandardScaler()
    X_train_scaled2 = scaler2.fit_transform(X_train_clf2)
    X_test_scaled2 = scaler2.transform(X_test_clf2)

    pca2 = PCA(n_components=25)
    X_train_pca2 = pca2.fit_transform(X_train_scaled2)
    X_test_pca2 = pca2.transform(X_test_scaled2)

    clf2 = SVC(C=10, gamma=0.1, kernel='rbf')
    clf2.fit(X_train_pca2, y_train_clf2)

    # --- Predicciones combinadas ---
    def predict_combined(X_raw):
        map_back = {0: 0, 2: 4, 3: 5, 4: 6, 5: 7}
        X_scaled1 = scaler1.transform(X_raw)
        X_pca1 = pca1.transform(X_scaled1)
        y_pred_clf1 = clf1.predict(X_pca1)

        final_pred = []
        for i, pred in enumerate(y_pred_clf1):
            if pred == 1:
                x2_scaled = scaler2.transform([X_raw[i]])
                x2_pca = pca2.transform(x2_scaled)
                y_pred_clf2 = clf2.predict(x2_pca)
                final_pred.append(y_pred_clf2[0] + 1)
            else:
                final_pred.append(map_back.get(pred, -1))
        return final_pred

    y_pred_combined = predict_combined(X_test)

    # --- Reportes ---
    labels_in_data = np.unique(y_test_clf1)
    target_names_filtered = [all_target_names[i] for i in labels_in_data]

    print(f"\nFold {fold_idx + 1} - Clasificador 1 (clases agrupadas):")
    y_pred_clf1 = clf1.predict(X_test_pca1)
    print(classification_report(
        y_test_clf1,
        y_pred_clf1,
        labels=labels_in_data,
        target_names=target_names_filtered,
        zero_division=0
    ))

    print(f"\nFold {fold_idx + 1} - Clasificador 2 (subclases de grupo 1):")
    print(classification_report(
        y_test_clf2,
        clf2.predict(X_test_pca2),
        target_names=["02_STROMA", "03_COMPLEX", "04_LYMPHO"],
        zero_division=0
    ))

    labels_combined = np.unique(np.concatenate([y_test_clf1, y_pred_combined]))
    target_names_combined = [class_names[i] for i in labels_combined]

    print(f"\nFold {fold_idx + 1} - Clasificación combinada (8 clases originales):")
    print(classification_report(
        y_test_clf1,
        y_pred_combined,
        labels=labels_combined,
        target_names=target_names_combined,
        zero_division=0
    ))

