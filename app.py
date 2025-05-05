from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        news = request.form['news']
        vec = vectorizer.transform([news])
        pred = model.predict(vec)[0]
        prediction = "Real News" if pred == 1 else "Fake News"
    return render_template("index.html", prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
