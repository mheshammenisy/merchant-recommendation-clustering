🛍️ Merchant Recommendation System – Clustering-Based ML Project  
Streamlit App | Python 3.x | Scikit-learn | KMeans Clustering

An end-to-end machine learning project that builds a **cluster-based merchant recommendation system** using transactional customer data.  
The system segments users based on behavioral patterns and recommends merchants based on popularity within each behavioral cluster.

The project covers data cleaning, feature engineering, preprocessing pipelines, unsupervised learning, and deployment via Streamlit.

---

## 📊 Model Overview

- **Learning Type:** Unsupervised Learning  
- **Algorithm:** KMeans Clustering  
- **Number of Clusters:** 7 (chosen for interpretability and business actionability)  
- **Recommendation Logic:** Cluster-level merchant popularity  

---

## 🧠 How It Works

### 1️⃣ User Clustering
Users are grouped based on **behavioral similarity**, not identity.

Clustering is driven by:
- Transaction value
- Transaction recency
- Transaction frequency / rank
- Spending category (encoded)
- Derived value–recency metric

Each transaction is assigned to a behavioral cluster using KMeans.

---

### 2️⃣ Merchant Recommendation
For a given user:
1. Identify the user’s cluster
2. Look at all transactions within that cluster
3. Rank merchants by frequency of occurrence
4. Recommend the most popular merchants in that cluster

This creates an **interpretable, explainable recommendation system**.

---

## 🔍 Key Insights

- User behavior clusters reflect distinct spending patterns (value, recency, category focus)
- Merchant popularity varies significantly across clusters
- Recommendations are **relative to cluster behavior**, not individual user history
- Categories influence recommendations indirectly via clustering

---

## 🧠 Feature Engineering

- **Value Recency**
- Robust scaling applied to highly skewed variables (e.g. transaction value)
- Standard scaling applied to remaining numeric features
- One-hot encoding for categorical spending category

---

## ⚙️ Machine Learning Pipeline

The project uses a **modular Scikit-learn pipeline**:


This ensures:
- No data leakage
- Reproducible clustering
- Safe model serialization
- Easy deployment

---

## 🗂️ Project Structure

Clustering ML project/
├── data/
│ └── raw/
│ └── Cleaned_Data_Merchant_Level_2.csv
├── models/
│ ├── preprocess.joblib
│ ├── kmeans_results.pkl
│ └── kmeans_model_k7.pkl
├── src/
│ ├── data_pipeline.py
│ ├── model.py
│ └── app.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore


---

## 🚀 Streamlit App

The Streamlit app allows you to:
- Enter a **User ID**
- See the user’s assigned cluster
- Get ranked merchant recommendations
- Inspect user transaction history for validation

---

## 🛠️ How to Run Locally

```bash
git clone <your-repo-url>
cd "Clustering ML project"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run src/app.py

👤 Author

Mohamed Hesham Sayed
Master’s Student – Energy & Data Analysis
Data Science & Machine Learning Enthusiast