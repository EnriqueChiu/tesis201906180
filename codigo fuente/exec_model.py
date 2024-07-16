import joblib
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Para cargar el modelo desde el archivo

# Cargar el modelo y el vectorizador TF-IDF
modelo_cargado = joblib.load("modelo_svm_5.joblib")
# Cargar el vectorizador TF-IDF desde el archivo
tfidf_vectorizer = joblib.load("vectorizador_tfidf_5.joblib")

# URL que quieres clasificar
url_nueva = "http://hensyouin.com/"

# Preprocesamiento de la URL utilizando el mismo vectorizador utilizado durante el entrenamiento
# tfidf_vectorizer = TfidfVectorizer()
url_preprocesada = tfidf_vectorizer.transform([url_nueva])

# Utiliza el modelo cargado para hacer la predicción
prediccion = modelo_cargado.predict(url_preprocesada)

# Si deseas obtener la probabilidad de cada clase
probabilidad = modelo_cargado.predict_proba(url_preprocesada)

# Imprime la predicción
if prediccion[0] == 0:
    print("La URL '{}' es legítima.".format(url_nueva))
else:
    print("La URL '{}' es de phishing.".format(url_nueva))

# Imprime las probabilidades de pertenecer a cada clase
print("Probabilidad de ser legítima:", probabilidad[0][0])
print("Probabilidad de ser de phishing:", probabilidad[0][1])
