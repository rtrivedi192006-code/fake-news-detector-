from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)   # ✅ this enables all cross-origin requests

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return "Fake News Detector API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data["text"]

    transformed = vectorizer.transform([text])

    prediction = model.predict(transformed)[0]
    prob = model.predict_proba(transformed)[0]

    result = "REAL" if prediction == 1 else "FAKE"
    confidence = float(max(prob))

    return jsonify({
        "prediction": result,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True)