import os
import cv2
import numpy as np
import re
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# --- Parámetros LBP ---
radius = 2
n_points = 8 * radius
method = 'uniform'

# --- Rutas y clases ---
base_path = r'C:\Users\ccriado\Desktop\CVC\Colon_CVC_rep\Colon_CVC\Kather_texture_2016_image_tiles_5000'
class_names = sorted(os.listdir(base_path))  

X = []
y = []

# --- Extraer características ---
def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method)
    hist_lbp, _ = np.histogram(lbp.ravel(),
                               bins=np.arange(0, n_points + 3),
                               range=(0, n_points + 2),
                               density=True)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])

    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()

    return np.concatenate([hist_lbp, hist_h, hist_s, hist_v])

# --- Extraer ID de imagen madre ---
def extract_image_id(filename):
    match = re.search(r'(CRC-Prim-HE-\d+_\d+\.tif)', filename)
    return match.group(1) if match else None

# --- Agrupar parches por imagen madre ---
grouped_by_image = []  # [(feature_vector, label, image_id), ...]

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

# --- Agrupar por imagen madre ---
image_to_data = {}
for feat, lbl, img_id in grouped_by_image:
    if img_id not in image_to_data:
        image_to_data[img_id] = []
    image_to_data[img_id].append((feat, lbl))

# --- Split por imagen madre ---
image_ids = list(image_to_data.keys())
train_ids, test_ids = train_test_split(image_ids, test_size=0.2, random_state=42) # Kfold cross validation

X_train, y_train, X_test, y_test = [], [], [], []

for img_id in train_ids:
    for feat, lbl in image_to_data[img_id]:
        X_train.append(feat)
        y_train.append(lbl)

for img_id in test_ids:
    for feat, lbl in image_to_data[img_id]:
        X_test.append(feat)
        y_test.append(lbl)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)


# --------- PCA ---------
# # PCA without scaled data
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_train)
# print(f"Varianza explicada: {pca.explained_variance_ratio_}")

# PCA with scaled data 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=25)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# print(f"Varianza explicada: {pca.explained_variance_ratio_}")

# pca_full = PCA()
# X_pca_full = pca_full.fit_transform(X_train_scaler)

# varianza_acumulada = np.cumsum(pca_full.explained_variance_ratio_)
# plt.figure(figsize=(8,5))
# plt.plot(varianza_acumulada, marker='o')
# plt.title('Varianza explicada acumulada por nº de componentes')
# plt.xlabel('Número de componentes')
# plt.ylabel('Varianza acumulada')
# plt.grid(True)
# plt.axhline(y=0.95, color='r', linestyle='--', label='95% varianza')
# plt.legend()
# plt.show()

# plt.figure(figsize=(8,6))
# plt.scatter(X_pca[np.array(y_train)==0, 0], X_pca[np.array(y_train)==0, 1], c='green', label='Benigno', alpha=0.5)
# plt.scatter(X_pca[np.array(y_train)==1, 0], X_pca[np.array(y_train)==1, 1], c='red', label='Maligno', alpha=0.5)
# plt.title("Visualización PCA (2D)")
# plt.xlabel("Componente 1")
# plt.ylabel("Componente 2")
# plt.legend()
# plt.grid(True)
# plt.show()



# --- Clasificador SVM ---
clf = SVC(kernel='rbf')
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1]
}

grid = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
grid.fit(X_train_pca, y_train)

print("Mejores parámetros:", grid.best_params_)
best_model = grid.best_estimator_

y_pred = best_model.predict(X_test_pca)
print(classification_report(
    y_test, y_pred,
    labels=np.arange(len(class_names)),
    target_names=class_names,
    zero_division=0  # para evitar warnings si alguna clase falta
))

# --- Matriz de confusión ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.title('Matriz de Confusión - LBP+HSV + SVM (Split por imagen madre)')
plt.show()
