import joblib
import csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Cargar el modelo entrenado y el vectorizador TF-IDF
svm_classifier = joblib.load("modelo_svm_3.joblib")
tfidf_vectorizer = joblib.load("vectorizador_tfidf_3.joblib")

# Nuevos datos
X_nuevos = []  # Inserta aquí tus nuevos datos
y_nuevos = []  # Inserta aquí las etiquetas correspondientes

with open("url_phishing4.csv", "r") as archivo_csv:
    lector_csv = csv.DictReader(archivo_csv)
    for fila in lector_csv:
        X_nuevos.append(fila["url"])
        y_nuevos.append(1)

with open("url_legitimo4.csv", "r") as archivo_csv:
    lector_csv = csv.DictReader(archivo_csv)
    for fila in lector_csv:
        X_nuevos.append(fila["url"])
        y_nuevos.append(0)

# Dividir los nuevos datos o simplemente usarlos todos para el entrenamiento
X_train_nuevos, X_test_nuevos, y_train_nuevos, y_test_nuevos = train_test_split(
    X_nuevos, y_nuevos, test_size=0.2, random_state=42
)

# Extracción de características utilizando el vectorizador TF-IDF
X_train_tfidf_nuevos = tfidf_vectorizer.transform(X_train_nuevos)
X_test_tfidf_nuevos = tfidf_vectorizer.transform(X_test_nuevos)

# Entrenamiento adicional del clasificador SVM
svm_classifier.fit(X_train_tfidf_nuevos, y_train_nuevos)

# Evaluación del modelo con los nuevos datos
accuracy_nuevos = svm_classifier.score(X_test_tfidf_nuevos, y_test_nuevos)
print("Accuracy con nuevos datos:", accuracy_nuevos)

# Guardar el modelo actualizado
joblib.dump(svm_classifier, "modelo_svm_4.joblib")
joblib.dump(tfidf_vectorizer, "vectorizador_tfidf_4.joblib")
