import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


# Cargar el DataFrame desde el archivo CSV
df = pd.read_csv("stats_all_descriptors_output.csv")

# Verificar las primeras filas del DataFrame
print(df.head())

# Separar características (X) y etiquetas (y)
X = df.drop(columns=["class name"])  # Eliminar la columna "class name"
y = df["class name"]  # Etiquetas (clase de la imagen)

# Dividir los datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_classifier = SVC(kernel="rbf", random_state=42)
svm_classifier.fit(X_train, y_train)

# Realizar predicciones con el conjunto de prueba
y_pred = svm_classifier.predict(X_test)

# Evaluar el rendimiento del modelo
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))