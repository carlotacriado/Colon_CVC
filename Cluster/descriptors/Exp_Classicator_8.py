import os
import cv2
import numpy as np
import re
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd

# --- Parámetros LBP ---
radius = 2
n_points = 8 * radius
method = 'uniform'

# --- Rutas y clases ---
base_path = r'/home/ccriado/CVC/Colon_CVC/Cluster/Kather_texture_2016_image_tiles_5000'
class_names = sorted(os.listdir(base_path))  

# --- Extracción de características ---
def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method)
    hist_lbp, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2), density=True)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()
    return np.concatenate([hist_lbp, hist_h, hist_s, hist_v])

def extract_image_id(filename):
    match = re.search(r'(CRC-Prim-HE-\d+_\d+\.tif)', filename)
    return match.group(1) if match else None

# --- Cargar datos ---
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

# --- Separar clases minoritarias ---
minor_classes = ['07_ADIPOSE', '08_EMPTY']
grouped_df = pd.DataFrame(grouped_by_image, columns=['features', 'label', 'image_id'])
label_to_class = {i: name for i, name in enumerate(class_names)}
grouped_df['class_name'] = grouped_df['label'].map(label_to_class)

df_major = grouped_df[~grouped_df['class_name'].isin(minor_classes)]
df_minor = grouped_df[grouped_df['class_name'].isin(minor_classes)]

minor_feats = np.stack(df_minor['features'].values)
minor_labels = df_minor['label'].values
minor_X_train, minor_X_test, minor_y_train, minor_y_test = train_test_split(
    minor_feats, minor_labels, test_size=0.2, stratify=minor_labels, random_state=42
)

# --- StratifiedGroupKFold ---
features = np.stack(df_major['features'].values)
labels = df_major['label'].values
groups = df_major['image_id'].values
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(sgkf.split(features, labels, groups))

X_train = np.concatenate([features[train_idx], minor_X_train])
y_train = np.concatenate([labels[train_idx], minor_y_train])
X_test = np.concatenate([features[test_idx], minor_X_test])
y_test = np.concatenate([labels[test_idx], minor_y_test])

# --- Normalizar y PCA ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
pca = PCA(n_components=25)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# --- Entrenar SVM ---
clf = SVC(kernel='rbf')
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1]}
grid = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', verbose=0, n_jobs=-1)
grid.fit(X_train_pca, y_train)
best_model = grid.best_estimator_

y_pred = best_model.predict(X_test_pca)
print("==== StratifiedGroupKFold + Minor split ====")
print("Mejores parámetros:", grid.best_params_)
print(classification_report(
    y_test, y_pred,
    labels=np.arange(len(class_names)),
    target_names=class_names,
    zero_division=0
))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.title('Matriz de Confusión - GroupKFold + Split 80/20 en Minor')
plt.show()

# ================================================
# === COMPARACIÓN: KFold ALEATORIO por parche ====
# ================================================

print("\n==== KFold sin agrupamiento ni stratified ====")
X_all = np.stack([f for f, _, _ in grouped_by_image])
y_all = np.array([l for _, l, _ in grouped_by_image])
kf = KFold(n_splits=5, shuffle=True, random_state=42)
kf_train_idx, kf_test_idx = next(kf.split(X_all))

X_train_kf, X_test_kf = X_all[kf_train_idx], X_all[kf_test_idx]
y_train_kf, y_test_kf = y_all[kf_train_idx], y_all[kf_test_idx]

scaler_kf = StandardScaler()
X_train_kf_scaled = scaler_kf.fit_transform(X_train_kf)
X_test_kf_scaled = scaler_kf.transform(X_test_kf)

pca_kf = PCA(n_components=25)
X_train_kf_pca = pca_kf.fit_transform(X_train_kf_scaled)
X_test_kf_pca = pca_kf.transform(X_test_kf_scaled)

grid_kf = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, scoring='accuracy', verbose=0, n_jobs=-1)
grid_kf.fit(X_train_kf_pca, y_train_kf)
best_model_kf = grid_kf.best_estimator_

y_pred_kf = best_model_kf.predict(X_test_kf_pca)
print("Mejores parámetros:", grid_kf.best_params_)
print(classification_report(
    y_test_kf, y_pred_kf,
    labels=np.arange(len(class_names)),
    target_names=class_names,
    zero_division=0
))

cm_kf = confusion_matrix(y_test_kf, y_pred_kf)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_kf, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.title('Matriz de Confusión - KFold por parche (sin agrupación)')
plt.show()
