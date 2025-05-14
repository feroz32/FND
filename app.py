from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and the vectorizer (replace 'model.pkl' with the actual file name of your model)
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))  # Assuming you saved the vectorizer too

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get the news article from the form
        news_article = request.form['news']

        # Convert the news article to a numerical format (vectorize it)
        news_vectorized = vectorizer.transform([news_article])  # This transforms the news article to numeric data

        # Make the prediction using the model
        prediction = model.predict(news_vectorized)[0]  # Predict and get the result

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
