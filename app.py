from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]

    # Convert text to numbers
    data = vectorizer.transform([message])

    # Predict
    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "Spam Message"
    else:
        result = "Not Spam Message"

    return render_template("index.html", prediction_text=result)

# Run app
if __name__ == "__main__":
    app.run(debug=True)