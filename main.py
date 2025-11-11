# ============================================================
# main.py — Enterprise Workforce Intelligence System (Final Stable)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import json, uuid, time
from datetime import datetime, date
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import plotly.express as px

# ============================================================
# Pinecone setup
# ============================================================
try:
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
except Exception:
    PINECONE_AVAILABLE = False

st.set_page_config(page_title="Enterprise Workforce Intelligence System", layout="wide")
st.title("🏢 Enterprise Workforce Intelligence System")
st.caption("AI-Driven Workforce Analytics • Productivity Intelligence • 360° Performance Overview")

# ============================================================
# Pinecone configuration
# ============================================================
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
INDEX_NAME = "task"
DIMENSION = 1024

pc, index = None, None
if PINECONE_AVAILABLE and PINECONE_API_KEY:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        existing = [i["name"] for i in pc.list_indexes()]
        if INDEX_NAME not in existing and existing:
            INDEX_NAME = existing[0]
        index = pc.Index(INDEX_NAME)
        st.success(f"✅ Connected to Pinecone index: {INDEX_NAME}")
    except Exception as e:
        st.warning(f"Pinecone connection failed — running local mode. ({e})")
else:
    st.warning("⚠️ Pinecone key missing — using local session data.")

# ============================================================
# Utilities
# ============================================================
def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def rand_vec():
    np.random.seed(int(time.time()) % 10000)
    return np.random.rand(DIMENSION).tolist()

def safe_meta(md):
    if isinstance(md, pd.Series):
        md = md.to_dict()
    elif not isinstance(md, dict):
        try:
            md = dict(md)
        except Exception:
            md = {"value": str(md)}
    clean = {}
    for k, v in md.items():
        try:
            if isinstance(v, (dict, list, np.ndarray)):
                clean[str(k)] = json.dumps(v, ensure_ascii=False)
            elif pd.isna(v):
                clean[str(k)] = ""
            else:
                clean[str(k)] = str(v)
        except Exception:
            clean[str(k)] = str(v)
    return clean

def upsert_data(id_, md):
    id_ = str(id_)
    md_clean = safe_meta(md)
    if not index:
        local = st.session_state.setdefault("LOCAL_DATA", {})
        local[id_] = md_clean
        return True
    try:
        index.upsert(vectors=[{"id": id_, "values": rand_vec(), "metadata": md_clean}])
        return True
    except Exception as e:
        st.error(f"Pinecone upsert failed: {e}")
        return False

def fetch_all():
    if not index:
        local = st.session_state.get("LOCAL_DATA", {})
        rows = [{"_id": k, **v} for k, v in local.items()]
        return pd.DataFrame(rows)
    try:
        res = index.query(vector=rand_vec(), top_k=1000, include_metadata=True)
        matches = res.get("matches", [])
        rows = []
        for m in matches:
            md = m.get("metadata", {})
            md["_id"] = m.get("id")
            rows.append(md)
        return pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"Fetch failed: {e}")
        return pd.DataFrame()

# ============================================================
# ML Models
# ============================================================
lin_reg = LinearRegression().fit([[0], [50], [100]], [0, 2.5, 5])
log_reg = LogisticRegression().fit([[0], [50], [100]], [0, 1, 1])

vectorizer = CountVectorizer()
train_texts = ["excellent work", "great job", "poor performance", "needs improvement", "bad result", "amazing effort"]
train_labels = [1, 1, 0, 0, 0, 1]
X_train = vectorizer.fit_transform(train_texts)
svm_clf = SVC(kernel="linear").fit(X_train, train_labels)

# ============================================================
# Role selection
# ============================================================
role = st.sidebar.selectbox("Access As", ["Manager", "Team Member", "Client", "HR Administrator"])

# ============================================================
# MANAGER DASHBOARD
# ============================================================
if role == "Manager":
    st.header("👨‍💼 Manager Dashboard — Tasks, Feedback, Leave, Meetings & Overview")
    df_all = fetch_all()
    tabs = st.tabs(["Task Management", "Feedback", "Leave Management", "Meeting Management", "360° Overview"])

    # --- Task Management ---
    with tabs[0]:
        st.subheader("Assign Task")
        with st.form("assign_task"):
            emp = st.text_input("Employee Name")
            dept = st.text_input("Department")
            task = st.text_input("Task Title")
            desc = st.text_area("Task Description")
            deadline = st.date_input("Deadline", value=date.today())
            submit = st.form_submit_button("Assign")
            if submit and emp and task:
                tid = str(uuid.uuid4())
                md = {
                    "type": "Task", "employee": emp, "department": dept, "task": task,
                    "description": desc, "deadline": str(deadline),
                    "completion": 0, "marks": 0, "status": "Assigned", "created": now()
                }
                upsert_data(tid, md)
                st.success(f"✅ Task '{task}' assigned to {emp}")

    # --- Feedback ---
    with tabs[1]:
        st.subheader("Manager Feedback & Evaluation")
        df_all = fetch_all()
        if df_all.empty:
            st.info("No tasks available.")
        else:
            search = st.text_input("Search by Employee or Task").strip().lower()
            if not search:
                st.info("Enter a search term to display completed tasks.")
            else:
                df_tasks = df_all[df_all["type"].astype(str).str.lower() == "task"]
                df_tasks["completion"] = pd.to_numeric(df_tasks["completion"], errors="coerce").fillna(0)
                matched = df_tasks[
                    df_tasks["employee"].astype(str).str.lower().str.contains(search, na=False)
                    | df_tasks["task"].astype(str).str.lower().str.contains(search, na=False)
                ]
