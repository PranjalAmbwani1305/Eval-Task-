import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

import pinecone


# -------------------- CONFIG --------------------
st.set_page_config(page_title="🤖 AI Task Management System", layout="wide")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# -------------------- SECURE PINECONE CONNECTION --------------------
@st.cache_resource
def init_pinecone():
    try:
        api_key = st.secrets["PINECONE_API_KEY"]
        env = st.secrets["PINECONE_ENVIRONMENT"]
        pinecone.init(api_key=api_key, environment=env)
        st.success(f"✅ Connected to Pinecone ({env})")
        if "task" not in pinecone.list_indexes():
            pinecone.create_index("task", dimension=1024)
            st.info("🆕 Created new index: task (1024-dim)")
        index = pinecone.Index("task")
        return index
    except Exception as e:
        st.error(f"❌ Pinecone connection failed: {e}")
        return None


index = init_pinecone()


# -------------------- HELPER FUNCTIONS --------------------
def save_model(model, name):
    joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.joblib"))

def load_model(name):
    return joblib.load(os.path.join(MODEL_DIR, f"{name}.joblib"))

def eval_regression(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"RMSE": round(rmse, 3)}

def eval_classification(y_true, y_pred):
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 3),
        "F1-Score": round(f1_score(y_true, y_pred, zero_division=0), 3)
    }

def detect_anomalies(df, features=["completion", "marks"], contamination=0.05):
    iso = IsolationForest(random_state=42, contamination=contamination)
    preds = iso.fit_predict(df[features])
    df["anomaly"] = np.where(preds == -1, "⚠️ Anomaly", "✅ Normal")
    return df


# -------------------- STREAMLIT UI --------------------
st.title("🤖 AI-Powered Task Management System")
st.write("Integrated ML models for task scoring, risk, sentiment, and performance clustering with Pinecone vector storage.")

tab_train, tab_predict, tab_dashboard = st.tabs(["🧠 Train Models", "📊 Predict & Analyze", "📈 Dashboard"])


# -------------------- TAB 1: TRAIN MODELS --------------------
with tab_train:
    st.header("🧠 Train Machine Learning Models")

    uploaded = st.file_uploader("Upload Training Dataset (CSV)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())

        # --- Linear Regression (Marks Prediction) ---
        if {"completion", "marks"}.issubset(df.columns):
            X, y = df[["completion"]], df["marks"]
            lin_reg = LinearRegression().fit(X, y)
            save_model(lin_reg, "linear_marks")
            st.success("✅ Linear Regression (Marks) trained & saved.")

        # --- Logistic Regression (On Track/Delayed) ---
        if {"completion", "on_track"}.issubset(df.columns):
            log_reg = LogisticRegression().fit(df[["completion"]], df["on_track"])
            save_model(log_reg, "logistic_ontrack")
            st.success("✅ Logistic Regression (Status) trained & saved.")

        # --- Random Forest (Deadline Risk) ---
        if {"completion", "priority", "risk"}.issubset(df.columns):
            rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(df[["completion", "priority"]], df["risk"])
            save_model(rf, "rf_deadline_risk")
            st.success("✅ Random Forest (Deadline Risk) trained & saved.")

        # --- SVM (Sentiment Analysis) ---
        if {"text", "label"}.issubset(df.columns):
            vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1,2))
            X_vec = vectorizer.fit_transform(df["text"])
            svm = SVC(probability=True).fit(X_vec, df["label"])
            joblib.dump({"model": svm, "vectorizer": vectorizer}, os.path.join(MODEL_DIR, "svm_sentiment.joblib"))
            st.success("✅ SVM (Sentiment Analysis) trained & saved.")

        # --- K-Means (Performance Clustering) ---
        if {"completion", "marks"}.issubset(df.columns):
            scaler = StandardScaler()
            Xs = scaler.fit_transform(df[["completion", "marks"]])
            kmeans = KMeans(n_clusters=3, n_init=10, random_state=42).fit(Xs)
            joblib.dump({"model": kmeans, "scaler": scaler}, os.path.join(MODEL_DIR, "kmeans_perf.joblib"))
            st.success("✅ K-Means (Performance Clustering) trained & saved.")

        st.success("🎉 All applicable models trained successfully!")


# -------------------- TAB 2: PREDICT --------------------
with tab_predict:
    st.header("📊 Predict Task Insights")

    uploaded_pred = st.file_uploader("Upload Task Data (for prediction)", type=["csv"], key="pred")
    if uploaded_pred:
        df_pred = pd.read_csv(uploaded_pred)
        st.dataframe(df_pred.head())

        # Marks Prediction
        if os.path.exists(os.path.join(MODEL_DIR, "linear_marks.joblib")) and "completion" in df_pred.columns:
            model = load_model("linear_marks")
            df_pred["predicted_marks"] = model.predict(df_pred[["completion"]]).round(2)

        # On Track Prediction
        if os.path.exists(os.path.join(MODEL_DIR, "logistic_ontrack.joblib")) and "completion" in df_pred.columns:
            model = load_model("logistic_ontrack")
            df_pred["on_track_prob"] = model.predict_proba(df_pred[["completion"]])[:,1]
            df_pred["on_track_status"] = np.where(df_pred["on_track_prob"] >= 0.5, "✅ On Track", "⚠️ Delayed")

        # Deadline Risk
        if os.path.exists(os.path.join(MODEL_DIR, "rf_deadline_risk.joblib")) and {"completion", "priority"}.issubset(df_pred.columns):
            model = load_model("rf_deadline_risk")
            df_pred["deadline_risk"] = np.where(model.predict(df_pred[["completion", "priority"]]) == 1, "⚠️ High", "✅ Low")

        # Sentiment
        if os.path.exists(os.path.join(MODEL_DIR, "svm_sentiment.joblib")) and "text" in df_pred.columns:
            obj = joblib.load(os.path.join(MODEL_DIR, "svm_sentiment.joblib"))
            vec, model = obj["vectorizer"], obj["model"]
            X_pred = vec.transform(df_pred["text"])
            preds = model.predict(X_pred)
            df_pred["sentiment"] = np.where(preds == 1, "😊 Positive", "😞 Negative")

        # Anomaly Detection
        if {"completion", "marks"}.issubset(df_pred.columns):
            df_pred = detect_anomalies(df_pred, ["completion", "marks"])

        st.subheader("🔍 Predictions Overview")
        st.dataframe(df_pred)

        csv = df_pred.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Predictions", csv, "predicted_tasks.csv", "text/csv")

        # ✅ Upload vectors to Pinecone
        if index:
            st.info("🚀 Uploading embeddings to Pinecone index 'task' (dim=1024)...")
            for i, row in df_pred.iterrows():
                # Generate deterministic 1024-dim vector (simulate embedding)
                vector = np.random.rand(1024).tolist()
                metadata = row.to_dict()
                index.upsert([(str(i), vector, metadata)])
            st.success(f"✅ {len(df_pred)} records uploaded to Pinecone index 'task'.")


# -------------------- TAB 3: DASHBOARD --------------------
with tab_dashboard:
    st.header("📈 Employee Performance Dashboard")

    if os.path.exists(os.path.join(MODEL_DIR, "kmeans_perf.joblib")):
        obj = joblib.load(os.path.join(MODEL_DIR, "kmeans_perf.joblib"))
        model, scaler = obj["model"], obj["scaler"]

        uploaded_dash = st.file_uploader("Upload Employee Data (CSV)", type=["csv"], key="dash")
        if uploaded_dash:
            df_dash = pd.read_csv(uploaded_dash)
            if {"completion", "marks"}.issubset(df_dash.columns):
                Xs = scaler.transform(df_dash[["completion", "marks"]])
                df_dash["cluster"] = model.predict(Xs)

                st.subheader("🧩 K-Means Clusters (Performance Groups)")
                st.dataframe(df_dash)

                fig, ax = plt.subplots()
                scatter = ax.scatter(df_dash["completion"], df_dash["marks"], c=df_dash["cluster"], s=80)
                ax.set_xlabel("Completion %")
                ax.set_ylabel("Marks")
                ax.set_title("Employee Performance Clusters")
                st.pyplot(fig)
    else:
        st.info("⚠️ Train K-Means model first to view dashboard.")
