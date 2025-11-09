import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import pandas as pd
import uuid
import json
import logging
import time
from datetime import datetime, date
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import plotly.express as px

# -----------------------------
# CONFIG & INIT
# -----------------------------
st.set_page_config(page_title="AI Workforce System", layout="wide")

st.title("💼 AI-Powered Workforce Performance & Task Management System")

# -----------------------------
# INTRODUCTORY OVERVIEW SECTION
# -----------------------------
with st.expander("📘 Overview: Unified AI Workforce Dashboard (Key Features)", expanded=True):
    st.markdown("""
    ### 🚀 **Unified AI Workforce Dashboard**
    
    #### **1️⃣ Manager’s Command Center**
    - 📊 **Task Summary** – Total, pending, in-progress, completed, and overdue tasks.  
    - 🌡️ **Team Performance Heatmap** – Visualize productivity by team, department, or individual.  
    - 🤖 **AI Alerts** – Automated insights on workload imbalance, delays, missed SLAs, or low engagement.  
    - 🎯 **Goal Tracker** – Track progress toward monthly, quarterly, or project-specific KPIs.  

    ---

    #### **2️⃣ AI-Driven Task Management Panel**
    - 📝 **Task Lifecycle Control** – Create, assign, prioritize, and track tasks.  
    - 🧠 **AI-Assisted Assignment** – Suggests task owners based on skills, workload, and past performance.  
    - ⏱️ **Predictive Deadline Estimation** – AI forecasts possible delays using historical trends.  
    - 🔔 **Auto-Reminders & Escalations** – Automated notifications for upcoming or overdue tasks.  
    - 📅 **Visual Views** – Switch between **Gantt**, **Kanban**, or **Calendar** layouts.  

    ---

    #### **3️⃣ 360° Employee Performance & Feedback Module**
    - 💬 **Feedback Collection** – Self, peer, manager, and cross-departmental input.  
    - 🤖 **AI Sentiment Analysis** – Detect tone, positivity, and improvement areas.  
    - 📈 **Performance Dashboard** – Aggregated KPIs and qualitative metrics.  
    - 🧩 **Growth Recommendations** – AI suggests upskilling or mentorship paths.  
    - 🕵️ **Confidential Feedback Summary** – Ensures honest communication.  

    ---

    #### **4️⃣ Managerial Actions & Approvals Hub**
    - 🔁 Reassign tasks or balance workload.  
    - ✅ Approve **leaves, overtime, deliverables, or escalations**.  
    - 💌 Send **recognition, warnings, or performance notes**.  
    - 📄 Generate **AI-compiled 360° appraisal summaries**.  

    ---

    #### 🌐 **5️⃣ Integration & Intelligence Layer**
    - 🧠 Powered by **Pinecone vector DB** for similarity matching & pattern recognition.  
    - 💻 Built with **Streamlit** for an interactive real-time interface.  
    - 🔐 Secure data access ensuring privacy in all analytics modules.  
    """)

st.divider()

# -----------------------------
# INITIALIZE PINECONE
# -----------------------------
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "task"
DIMENSION = 1024

if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(INDEX_NAME)

def now(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def rand_vec(): return np.random.rand(DIMENSION).tolist()

# -----------------------------
# SIMPLE AI MODELS
# -----------------------------
lin_reg = LinearRegression().fit([[0], [50], [100]], [0, 2.5, 5])
log_reg = LogisticRegression().fit([[0], [40], [80], [100]], [0, 0, 1, 1])

comments = ["excellent work", "needs improvement", "bad performance", "great job", "average"]
sentiments = [1, 0, 0, 1, 0]
vectorizer = CountVectorizer()
svm_clf = SVC().fit(vectorizer.fit_transform(comments), sentiments)

rf = RandomForestClassifier()
rf.fit(np.array([[10, 2], [50, 1], [90, 0], [100, 0]]), [0, 1, 0, 0])

# -----------------------------
# LOGGER
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("safe_meta")

# -----------------------------
# SAFE META & UPSERT
# -----------------------------
def safe_meta(md):
    clean, invalid = {}, {}
    for k, v in md.items():
        try:
            if v is None or (isinstance(v, float) and (pd.isna(v) or np.isnan(v))):
                v = ""
            elif isinstance(v, (datetime, date)):
                v = v.isoformat()
            elif isinstance(v, (np.generic, np.number)):
                v = float(v)
            elif isinstance(v, (list, dict)):
                v = json.dumps(v)
            elif not isinstance(v, (str, int, float, bool)):
                v = str(v)
        except Exception as e:
            invalid[k] = f"ERROR {type(e).__name__}: {e}"
            v = str(v)
        clean[k] = v

    if invalid:
        with st.expander("⚠️ Metadata Conversion Warnings", expanded=False):
            st.warning("Some metadata fields were corrected before upload.")
            for key, val in invalid.items():
                st.text(f"{key}: {val}")
        logger.warning(f"Invalid metadata fields: {invalid}")
    return clean

def safe_upsert(id_, vec, md, retries=2, delay=1.5):
    for attempt in range(retries):
        try:
            index.upsert([{"id": id_, "values": vec, "metadata": md}])
            return True
        except Exception as e:
            logger.warning(f"Upsert attempt {attempt+1} failed: {e}")
            time.sleep(delay)
    st.error("❌ Pinecone upsert failed after retries.")
    return False

# -----------------------------
# DATA FETCH
# -----------------------------
def fetch_all():
    try:
        res = index.query(vector=rand_vec(), top_k=1000, include_metadata=True)
        if not res.matches:
            return pd.DataFrame()
        rows = []
        for m in res.matches:
            md = m.metadata or {}
            md["_id"] = m.id
            rows.append(md)
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# -----------------------------
# ROLE SELECTION
# -----------------------------
role = st.sidebar.selectbox("🔐 Login as", ["Manager", "Team Member", "Client", "Admin"])
current_month = datetime.now().strftime("%B %Y")

# -----------------------------
# MANAGER
# -----------------------------
if role == "Manager":
    st.header("👑 Manager Command Center")
    tab1, tab2 = st.tabs(["📝 Assign Task", "📊 Boss Review & Adjustment"])

    with tab1:
        with st.form("assign"):
            company = st.text_input("Company Name")
            employee = st.text_input("Employee Name")
            task = st.text_input("Task Title")
            desc = st.text_area("Description")
            deadline = st.date_input("Deadline", value=date.today())
            month = st.text_input("Month", value=current_month)
            submit = st.form_submit_button("Assign Task")

            if submit and company and employee and task:
                tid = str(uuid.uuid4())
                md = safe_meta({
                    "company": company,
                    "employee": employee,
                    "task": task,
                    "description": desc,
                    "deadline": deadline.isoformat(),
                    "month": month,
                    "completion": 0,
                    "marks": 0,
                    "status": "Assigned",
                    "reviewed": False,
                    "client_reviewed": False,
                    "assigned_on": now()
                })
                safe_upsert(tid, rand_vec(), md)
                st.success(f"✅ Task '{task}' assigned to {employee}")

    with tab2:
        if st.button("🔄 Refresh Data"):
            st.session_state.pop("manager_df", None)

        if "manager_df" not in st.session_state:
            st.session_state["manager_df"] = fetch_all()
        df = st.session_state["manager_df"]

        if df.empty:
            st.warning("No tasks found.")
        else:
            if "completion" in df.columns:
                df["completion"] = pd.to_numeric(df["completion"], errors="coerce").fillna(0)
            else:
                df["completion"] = 0
            for _, r in df[df["completion"] > 0].iterrows():
                st.markdown(f"### {r['task']}")
                st.write(f"**Employee:** {r.get('employee', 'Unknown')}")
                st.write(f"**Reported Completion:** {r.get('completion', 0)}%")
                st.write(f"**Marks:** {r.get('marks', 0)}")
                adjusted_completion = st.slider(
                    f"Adjust Completion % for {r['task']}",
                    0, 100, int(r.get("completion", 0)), key=f"adj_{r['_id']}"
                )
                adjusted_marks = float(lin_reg.predict([[adjusted_completion]])[0])
                comments = st.text_area(f"Boss Comments for {r['task']}", key=f"boss_cmt_{r['_id']}")
                approve = st.radio(f"Approve {r['task']}?", ["Yes", "No"], key=f"boss_app_{r['_id']}")
                if st.button(f"Finalize {r['task']}", key=f"final_{r['_id']}"):
                    sentiment_val = int(svm_clf.predict(vectorizer.transform([comments]))[0])
                    sentiment = "Positive" if sentiment_val == 1 else "Negative"
                    md = safe_meta({
                        **r,
                        "completion": adjusted_completion,
                        "marks": adjusted_marks,
                        "manager_comments": comments,
                        "reviewed": True,
                        "sentiment": sentiment,
                        "approved_by_boss": approve == "Yes",
                        "reviewed_on": now()
                    })
                    safe_upsert(r["_id"], rand_vec(), md)
                    st.success(f"✅ Review finalized ({sentiment}).")
                    st.session_state.pop("manager_df", None)
                    st.rerun()

# (TEAM MEMBER / CLIENT / ADMIN sections unchanged)
