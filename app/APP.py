from flask import Flask, render_template, request
import ANALIZADOR

app = Flask(__name__, template_folder='templates')

@app.route("/")
def index():
    return render_template('index.html')

@app.post('/modelo')
def login_post():
    url = request.form['url']
    modelo = request.form['modelo']
    result = ANALIZADOR.analizador(url, modelo)
    return render_template('index.html', result=result['legitima'], p_legitima=result['p_legitima'], p_phishing=result['p_phishing'], modelo=modelo)

