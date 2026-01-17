# src/model.py

from pathlib import Path
import os
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "kmeans_model_k7.pkl"
SIL_PATH = BASE_DIR / "models" / "kmeans_results.pkl"


def run_silhouette(X):
    results = []

    for k in range(2, 11):
        model = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        results.append((k, score))

    SIL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(results, SIL_PATH)
    return results


def train_kmeans(X, df, k=7):
    # If model exists, load it (faster + consistent)
    if MODEL_PATH.exists():
        kmeans = joblib.load(MODEL_PATH)
        df["cluster"] = kmeans.predict(X)
        return df

    # Otherwise train and save
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    df["cluster"] = kmeans.fit_predict(X)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(kmeans, MODEL_PATH)
    return df
