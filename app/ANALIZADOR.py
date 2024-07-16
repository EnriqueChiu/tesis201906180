import joblib

def analizador(url, num):
    modelo_cargado = joblib.load("modelo_svm_{}.joblib".format(num))
    tfidf_vectorizer = joblib.load("vectorizador_tfidf_{}.joblib".format(num))

    url_preprocesada = tfidf_vectorizer.transform([url])

    prediccion = modelo_cargado.predict(url_preprocesada)
    probabilidad = modelo_cargado.predict_proba(url_preprocesada)

    result = {"legitima": "", "p_legitima": "", "p_phishing": ""}

    if prediccion[0] == 0:
        result["legitima"] = "La URL '{}' es legítima.".format(url)
    else:
        result["legitima"] = "La URL '{}' es de phishing.".format(url)

    result["p_legitima"] = "Probabilidad de ser legítima: {}".format(probabilidad[0][0])
    result["p_phishing"] = "Probabilidad de ser phishing: {}".format(probabilidad[0][1])

    return result
