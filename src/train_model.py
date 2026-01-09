import os
import joblib
from load_data import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "..", "data")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

def train():
    texts, labels = load_dataset(DATA_PATH)

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)
    y = labels

    model = LogisticRegression()
    model.fit(X, y)

    # 🔹 SAVE FILES
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("✅ Model saved to:", MODEL_PATH)
    print("✅ Vectorizer saved to:", VECTORIZER_PATH)

if __name__ == "__main__":
    train()
