import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import pandas as pd
from datetime import datetime, date
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from textblob import TextBlob
import plotly.express as px
import uuid

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="🤖 AI-Driven Employee Management System", layout="wide")
st.title("🤖 AI-Driven Employee Management System")

# -----------------------------
# INITIALIZATION
# -----------------------------
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "task"
DIMENSION = 1024

# ✅ Ensure index exists
if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(INDEX_NAME)

# -----------------------------
# HELPERS
# -----------------------------
def now(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def rand_vec(): return np.random.rand(DIMENSION).tolist()
def safe_rerun():
    try: st.rerun()
    except Exception: pass
def safe_upsert(md):
    try:
        index.upsert([{"id": str(md.get("_id", uuid.uuid4())), "values": rand_vec(), "metadata": md}])
    except Exception as e:
        st.error(f"Pinecone upsert failed: {e}")

# -----------------------------
# ML MODELS
# -----------------------------
lin_reg = LinearRegression().fit([[0],[50],[100]],[0,2.5,5])
log_reg = LogisticRegression().fit([[0],[40],[80],[100]],[0,0,1,1])
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(["excellent work","needs improvement","bad performance","great job","average"])
svm_clf = SVC().fit(X_train,[1,0,0,1,0])
rf = RandomForestClassifier().fit(np.array([[10,2],[50,1],[90,0],[100,0]]),[1,0,0,0])

# -----------------------------
# FETCH DATA
# -----------------------------
def fetch_all():
    try:
        res = index.query(vector=rand_vec(), top_k=1000, include_metadata=True)
        rows=[]
        for m in res.matches:
            md=m.metadata or {}
            md["_id"]=m.id
            rows.append(md)
        return pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"⚠️ Unable to fetch data: {e}")
        return pd.DataFrame()

# -----------------------------
# SIDEBAR
# -----------------------------
role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client", "Admin"])
current_month = datetime.now().strftime("%B %Y")

# -----------------------------
# MANAGER DASHBOARD
# -----------------------------
if role == "Manager":
    st.header("👨‍💼 Manager Dashboard")
    tab1, tab2, tab3, tab4 = st.tabs(["Assign Task", "Review Tasks", "360° Overview", "Managerial Actions"])
    # ... (same as previous manager section) ...

# -----------------------------
# TEAM MEMBER PORTAL
# -----------------------------
elif role == "Team Member":
    st.header("👷 Team Member Portal")
    tab1, tab2, tab3 = st.tabs(["My Tasks", "AI Feedback", "Submit Leave"])
    # ... (same as previous team member section) ...

# -----------------------------
# CLIENT REVIEW
# -----------------------------
elif role == "Client":
    st.header("🧾 Client Review")
    # ... (same as previous client section) ...

# -----------------------------
# ADMIN DASHBOARD (NEW)
# -----------------------------
elif role == "Admin":
    st.header("🏢 Admin Dashboard – HR & Analytics")
    st.markdown("Gain organization-wide insights across departments, performance, and leaves.")

    df = fetch_all()
    if df.empty:
        st.warning("No data available yet.")
        st.stop()

    # --- Department Performance ---
    st.subheader("📊 Department-wise Performance Overview")
    if "department" not in df.columns:
        df["department"] = np.random.choice(["HR","Tech","Sales","Design","Operations"], size=len(df))  # Demo departments

    dep_perf = df.groupby("department").agg({
        "marks":"mean",
        "completion":"mean"
    }).reset_index()

    fig = px.bar(dep_perf, x="department", y="marks", color="completion",
                 title="Department Average Marks vs Completion",
                 text_auto=".2f")
    st.plotly_chart(fig, use_container_width=True)

    # --- Top Performers ---
    st.subheader("🏅 Top Performers (Based on Consistency + AI Marks)")
    if "employee" in df.columns:
        df["score"] = 0.7*df["marks"] + 0.3*df["completion"]
        top_df = df.groupby("employee")["score"].mean().reset_index().sort_values("score", ascending=False).head(10)
        st.dataframe(top_df)
        fig2 = px.bar(top_df, x="employee", y="score", title="Top 10 Employees (Weighted Performance)")
        st.plotly_chart(fig2, use_container_width=True)

    # --- Leave Overview ---
    st.subheader("🏖️ Company Leave Overview")
    leave_df = df[df["type"]=="leave"] if "type" in df.columns else pd.DataFrame()
    if not leave_df.empty:
        leave_df = leave_df[["employee","company","start_date","end_date","reason","status","requested_on"]]
        st.dataframe(leave_df)
        leave_chart = leave_df.groupby("company").size().reset_index(name="Leave Requests")
        fig3 = px.pie(leave_chart, names="company", values="Leave Requests", title="Leave Distribution by Company")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No leave requests yet.")

    # --- Department Heatmap ---
    st.subheader("🔥 Department Performance Heatmap")
    if "marks" in df.columns and "department" in df.columns:
        pivot = df.pivot_table(values="marks", index="department", columns="month", aggfunc="mean").fillna(0)
        st.dataframe(pivot.style.background_gradient(cmap="Greens"))

    # --- Summary Insights ---
    st.subheader("📈 HR Summary Insights")
    avg_marks = df["marks"].mean() if "marks" in df.columns else 0
    avg_completion = df["completion"].mean() if "completion" in df.columns else 0
    st.metric("Average Marks (All Employees)", f"{avg_marks:.2f}")
    st.metric("Average Completion %", f"{avg_completion:.1f}")
