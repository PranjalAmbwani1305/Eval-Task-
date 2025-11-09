# ===============================================
# AI Task Management Suite (Agentic AI + EvalTrack)
# Streamlit App – Fully Compatible with Pinecone v4+
# ===============================================

import streamlit as st
import numpy as np
import pandas as pd
import uuid
from datetime import datetime
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from pinecone import Pinecone, ServerlessSpec


# --------------------------
# Streamlit Page Setup
# --------------------------
st.set_page_config(page_title="AI Task Management Suite", page_icon="💼", layout="wide")
st.title("💼 AI-Powered Task Management & Performance Evaluation System")


# --------------------------
# Pinecone Initialization (v4+)
# --------------------------
INDEX_NAME = "task"
DIMENSION = 128  # or 1024 if you use embeddings

def init_pinecone():
    """Initialize Pinecone (latest version)"""
    try:
        api_key = st.secrets["PINECONE_API_KEY"]  # stored securely in secrets.toml
        pc = Pinecone(api_key=api_key)

        existing = [i["name"] for i in pc.list_indexes()]
        if INDEX_NAME not in existing:
            pc.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        index = pc.Index(INDEX_NAME)
        return pc, index

    except Exception as e:
        st.error("❌ Pinecone initialization failed. Check your API key or SDK version.")
        st.write(e)
        return None, None


pc, index = init_pinecone()
if not pc:
    st.stop()


# --------------------------
# Machine Learning Models
# --------------------------
lin_reg = LinearRegression().fit([[0], [100]], [0, 5])
log_reg = LogisticRegression(solver="liblinear").fit([[0], [50], [100]], [0, 0, 1])

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(["good work", "excellent", "needs improvement", "bad performance"])
y_train = [1, 1, 0, 0]
svm_clf = SVC().fit(X_train, y_train)


# --------------------------
# Helper Functions
# --------------------------
def random_vector(dim=DIMENSION):
    return np.random.rand(dim).tolist()

def safe_metadata(md: dict):
    clean = {}
    for k, v in md.items():
        if hasattr(v, "item"):
            v = v.item()
        clean[k] = v
    return clean

def classify_performance(tasks):
    perf = {}
    for task in tasks:
        emp = task.get('employee', 'Unknown')
        score = task.get('marks', 0)
        perf.setdefault(emp, []).append(score)
    out = {}
    for emp, scores in perf.items():
        avg = np.mean(scores)
        if avg >= 4:
            out[emp] = "High"
        elif avg >= 2.5:
            out[emp] = "Medium"
        else:
            out[emp] = "Low"
    return out


# --------------------------
# Role Selection
# --------------------------
role = st.sidebar.radio("🔑 Login as", ["👩‍💻 Team Member", "🧑‍💼 Manager", "👨‍💼 Client", "🛠️ Admin"])


# ======================================================
# 👩‍💻 TEAM MEMBER SECTION
# ======================================================
if role == "👩‍💻 Team Member":
    st.header("👩‍💻 Submit Your Task")

    company = st.text_input("🏢 Company Name")
    employee = st.text_input("👤 Your Name")
    task = st.text_input("📝 Task Title")
    completion = st.slider("✅ Completion Percentage", 0, 100, 0)

    if st.button("📩 Submit Task"):
        if not (company and employee and task):
            st.error("⚠️ Please fill all fields before submitting.")
        else:
            marks = float(lin_reg.predict([[completion]])[0])
            status = int(log_reg.predict([[completion]])[0])
            status_text = "On Track" if status == 1 else "Delayed"
            task_id = str(uuid.uuid4())
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            metadata = safe_metadata({
                "company": company,
                "employee": employee,
                "task": task,
                "completion": completion,
                "marks": marks,
                "status": status_text,
                "reviewed": False,
                "timestamp": timestamp
            })

            try:
                index.upsert(vectors=[{"id": task_id, "values": random_vector(), "metadata": metadata}])
                st.success(f"✅ Task '{task}' submitted successfully by {employee}")
            except Exception as e:
                st.error("❌ Failed to submit task.")
                st.write(e)


# ======================================================
# 👨‍💼 CLIENT SECTION
# ======================================================
elif role == "👨‍💼 Client":
    st.header("📊 Client View of Approved Tasks")

    company = st.text_input("🏢 Company Name")
    if st.button("🔍 View Approved Tasks") and company:
        try:
            res = index.query(
                vector=random_vector(),
                top_k=100,
                include_metadata=True,
                filter={"company": {"$eq": company}, "reviewed": {"$eq": True}}
            )
            matches = res.matches if hasattr(res, "matches") else res["matches"]
            if matches:
                df = pd.DataFrame([m.metadata for m in matches])
                st.dataframe(df[["employee", "task", "completion", "marks", "status", "sentiment"]])
            else:
                st.info("ℹ️ No approved tasks found for this company.")
        except Exception as e:
            st.error("❌ Failed to query Pinecone.")
            st.write(e)


# ======================================================
# 🧑‍💼 MANAGER SECTION
# ======================================================
elif role == "🧑‍💼 Manager":
    st.header("🧭 Manager Review & Dashboard")

    try:
        res = index.query(vector=random_vector(), top_k=200, include_metadata=True)
        matches = res.matches if hasattr(res, "matches") else res["matches"]
        tasks = [m.metadata for m in matches] if matches else []
    except Exception as e:
        st.error("❌ Failed to fetch tasks.")
        st.write(e)
        tasks = []

    if not tasks:
        st.warning("⚠️ No task data found.")
    else:
        df = pd.DataFrame(tasks)

        st.subheader("📈 Performance Summary")
        total = len(df)
        reviewed = len(df[df["reviewed"] == True])
        on_track = len(df[df["status"] == "On Track"])
        delayed = len(df[df["status"] == "Delayed"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Tasks", total)
        c2.metric("Reviewed", reviewed)
        c3.metric("On Track", on_track)
        c4.metric("Delayed", delayed)

        st.subheader("🔥 Employee Performance Classification")
        perf = classify_performance(tasks)
        perf_df = pd.DataFrame(list(perf.items()), columns=["Employee", "Category"])
        st.dataframe(perf_df)

        st.subheader("💬 Sentiment Analysis on Feedback")
        feedback = st.text_area("Enter feedback for sentiment analysis:")
        if st.button("🧠 Analyze Sentiment"):
            if feedback.strip():
                X_new = vectorizer.transform([feedback])
                sentiment = svm_clf.predict(X_new)[0]
                st.success(f"Sentiment: {'Positive' if sentiment == 1 else 'Negative'}")
            else:
                st.warning("⚠️ Please enter some feedback text.")


# ======================================================
# 🛠️ ADMIN SECTION
# ======================================================
elif role == "🛠️ Admin":
    st.header("🛠️ Admin Control Panel")

    try:
        res = index.query(vector=random_vector(), top_k=300, include_metadata=True)
        matches = res.matches if hasattr(res, "matches") else res["matches"]
        tasks = [m.metadata for m in matches] if matches else []
    except Exception as e:
        st.error("❌ Failed to query Pinecone data.")
        st.write(e)
        tasks = []

    if tasks:
        df = pd.DataFrame(tasks)

        st.subheader("📊 Global System Overview")
        total_tasks = len(df)
        total_companies = len(set(df.get("company", [])))
        total_employees = len(set(df.get("employee", [])))
        pending = len(df[df.get("reviewed", False) == False])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Tasks", total_tasks)
        c2.metric("Companies", total_companies)
        c3.metric("Employees", total_employees)
        c4.metric("Pending Reviews", pending)

        st.subheader("🏢 Company Summary")
        comp = df.groupby("company").agg(
            total_tasks=("task", "count"),
            avg_completion=("completion", "mean"),
            avg_marks=("marks", "mean")
        ).reset_index()
        st.dataframe(comp)

        st.subheader("👥 Employee Marks Distribution")
        if "employee" in df.columns and "marks" in df.columns:
            perf = df.groupby("employee")["marks"].mean().reset_index()
            st.bar_chart(perf.set_index("employee"))

        st.subheader("⬇️ Export Data")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download All Task Data", csv, "all_tasks.csv", "text/csv")

        st.subheader("⚙️ Pinecone Index Management")
        st.warning("Deleting the index will permanently remove all data!")

        if st.button("🧹 Delete Index"):
            try:
                pc.delete_index(INDEX_NAME)
                st.success(f"✅ Index '{INDEX_NAME}' deleted successfully.")
            except Exception as e:
                st.error("❌ Failed to delete index.")
                st.write(e)

        if st.button("🚀 Recreate Index"):
            try:
                pc.create_index(
                    name=INDEX_NAME,
                    dimension=DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                st.success("✅ Index recreated successfully.")
            except Exception as e:
                st.error("❌ Failed to recreate index.")
                st.write(e)
    else:
        st.info("ℹ️ No data available yet for admin review.")
