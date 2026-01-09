import pickle

def predict_news(text):
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    confidence = max(model.predict_proba(X)[0])

    label = "REAL" if prediction == 1 else "FAKE"
    return label, confidence

if __name__ == "__main__":
    sample_text = input("Enter news text:\n")
    label, confidence = predict_news(sample_text)
    print(f"\nPrediction: {label}")
    print(f"Confidence: {confidence:.2f}")
