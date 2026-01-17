# src/data_pipeline.py

import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer


DATA_PATH = r"C:\Clustering ML project\data\raw\Cleaned_Data_Merchant_Level_2.csv"
PREPROCESS_PATH = r"C:\Clustering ML project\models\preprocess.joblib"


def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    df["value_recency"] = df["trx_vlu"] / (df["trx_age"] + 1)

    df_model = df.drop(columns=["user_id", "mer_id", "points"], errors="ignore")

    num_cols = df_model.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_model.select_dtypes(exclude=[np.number]).columns.tolist()

    robust_cols = ["trx_vlu"]
    standard_cols = [c for c in num_cols if c not in robust_cols]

    numeric_robust = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])

    numeric_standard = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    preprocess = ColumnTransformer([
        ("num_robust", numeric_robust, robust_cols),
        ("num_std", numeric_standard, standard_cols),
        ("cat", categorical_pipe, cat_cols),
    ])

    X = preprocess.fit_transform(df_model)

    joblib.dump(preprocess, PREPROCESS_PATH)

    return df, X
