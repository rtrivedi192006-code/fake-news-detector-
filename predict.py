import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Take input
text = input("Enter news text: ")

# Transform text
transformed = vectorizer.transform([text])

# Predict
prediction = model.predict(transformed)[0]
prob = model.predict_proba(transformed)[0]

# Output
result = "REAL" if prediction == 1 else "FAKE"
confidence = max(prob)

print("Prediction:", result)
print("Confidence:", confidence)