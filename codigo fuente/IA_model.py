import csv
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
# import pandas as pd

# Supongamos que tienes un DataFrame 'data' con columnas 'url' y 'label'
# donde 'label' indica si la URL es legítima (0) o de phishing (1)

# Preprocesamiento de datos
X = []
y = []
with open("url_phishing1.csv", "r") as archivo_csv:
    lector_csv = csv.DictReader(archivo_csv)
    for fila in lector_csv:
        # `fila` es un diccionario donde las claves son los nombres de las columnas
        X.append(fila["url"])
        y.append(1)

with open("url_legitimo1.csv", "r") as archivo_csv:
    lector_csv = csv.DictReader(archivo_csv)
    for fila in lector_csv:
        X.append(fila["url"])
        y.append(0)


# División de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Extracción de características utilizando TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Entrenamiento de un clasificador SVM
svm_classifier = SVC(kernel="linear", probability=True)
svm_classifier.fit(X_train_tfidf, y_train)

# Evaluación del modelo
accuracy = svm_classifier.score(X_test_tfidf, y_test)
print("Accuracy:", accuracy)

joblib.dump(svm_classifier, "modelo_svm_1.joblib")
joblib.dump(tfidf_vectorizer, "vectorizador_tfidf_1.joblib")
