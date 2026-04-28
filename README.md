📰 Fake News Detector
📌 Overview

This project is a command-line Fake News Detector that classifies input news text as Real or Fake using a pre-trained machine learning model. It uses a saved model (model.pkl) and vectorizer (vectorizer.pkl) to make instant predictions.

🚀 Features
Predicts whether news is Real or Fake
Accepts user input from terminal
Displays prediction + probability score
Uses pre-trained model (no training required at runtime)

🛠️ Technologies Used
Python
Scikit-learn
Pickle (for model storage)
NLP (TF-IDF Vectorization)

⚙️ How It Works
Loads trained model and vectorizer using pickle
Takes news text as input from user
Converts text into numerical format using vectorizer
Model predicts the label (Real/Fake)
Outputs prediction and confidence score

🧠 Model Details
Model loaded from: model.pkl
Vectorizer: vectorizer.pkl
Likely algorithms used: Logistic Regression / Naive Bayes
Input: Raw news text
Output: Label + Probability

⚠️ Limitations
Works only on text input (no URL/news scraping)
Accuracy depends on training dataset quality
Cannot detect very new or unseen misinformation patterns
