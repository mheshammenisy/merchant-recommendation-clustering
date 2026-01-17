import streamlit as st
import pandas as pd
from data_pipeline import load_and_prepare_data
from model import train_kmeans


def recommend_merchants(df, user_id, top_n=5):
    user_cluster = df.loc[df["user_id"] == user_id, "cluster"].iloc[0]

    recs = (
        df[df["cluster"] == user_cluster]
        .groupby("mer_id")
        .size()
        .sort_values(ascending=False)
        .head(top_n)
        .index
        .tolist()
    )

    return recs, user_cluster


# ================= STREAMLIT UI =================

st.title("Merchant Recommendation System")

# load data + model
df, X = load_and_prepare_data()
df = train_kmeans(X, df, k=7)

# input
user_id = st.number_input("Enter User ID", step=1)

if st.button("Recommend"):
    try:
        recs, cluster = recommend_merchants(df, user_id)

        st.success(f"User belongs to Cluster {cluster}")
        st.subheader("Recommended Merchants (Ranked)")

        labels = [
            "🔥 Most recommended",
            "⭐ Highly recommended",
            "👍 Recommended",
            "➖ Less recommended",
            "⬇️ Least recommended"
        ]

        for i, mer in enumerate(recs):
            label = labels[i] if i < len(labels) else f"Rank {i+1}"
            st.write(f"**{label}:** Merchant {mer}")

    except IndexError:
        st.error("User ID not found in dataset")
