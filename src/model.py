# src/model.py

import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


MODEL_PATH = r"C:\Clustering ML project\models\kmeans_model_k7.pkl"
SIL_PATH = r"C:\Clustering ML project\models\kmeans_results.pkl"


def run_silhouette(X):
    results = []

    for k in range(2, 11):
        model = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        results.append((k, score))

    joblib.dump(results, SIL_PATH)
    return results


def train_kmeans(X, df, k=7):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    df["cluster"] = kmeans.fit_predict(X)

    joblib.dump(kmeans, MODEL_PATH)
    return df
