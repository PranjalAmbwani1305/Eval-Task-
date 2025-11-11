# ============================================================
# main.py — Enterprise Workforce Performance Intelligence System ⚙️
# ============================================================
# ✅ Requirements:
# pip install streamlit pinecone-client scikit-learn plotly huggingface-hub pandas openpyxl PyPDF2

import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import plotly.express as px
import numpy as np
import pandas as pd
import uuid
import json
import time
from datetime import datetime, date
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# optional
try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False
try:
    import PyPDF2
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

# ----------------------------
# App config
# ----------------------------
st.set_page_config(page_title="AI Workforce Intelligence Platform", layout="wide")
st.title("AI Workforce Intelligence Platform")

# ----------------------------
# Secrets & constants
# ----------------------------
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
HF_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")
INDEX_NAME = "task"   # must be lowercase alphanumeric or '-'
DIMENSION = 1024

# ----------------------------
# Pinecone init
# ----------------------------
pc = None
index = None
if PINECONE_API_KEY:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        existing = [i["name"] for i in pc.list_indexes()]
        if INDEX_NAME not in existing:
            pc.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            for _ in range(20):
                try:
                    desc = pc.describe_index(INDEX_NAME)
                    if desc.get("status", {}).get("ready"):
                        break
                except Exception:
                    pass
                time.sleep(1)
        index = pc.Index(INDEX_NAME)
        st.caption(f"Connected to Pinecone index: {INDEX_NAME}")
    except Exception as e:
        st.warning(f"Pinecone init failed — running local-only. ({e})")
else:
    st.warning("Pinecone API key missing — running local-only.")

# ----------------------------
# Utilities
# ----------------------------
def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def rand_vec():
    return np.random.rand(DIMENSION).tolist()

def safe_meta(md: dict) -> dict:
    clean = {}
    for k, v in (md or {}).items():
        try:
            if isinstance(v, (datetime, date)):
                clean[k] = v.isoformat()
            elif isinstance(v, (dict, list)):
                clean[k] = json.dumps(v)
            elif pd.isna(v):
                clean[k] = ""
            else:
                clean[k] = v
        except Exception:
            clean[k] = str(v)
    return clean

def upsert_data(id_, md: dict) -> bool:
    id_ = str(id_)
    if not index:
        loc = st.session_state.setdefault("LOCAL_DATA", {})
        loc[id_] = md
        return True
    try:
        index.upsert([{"id": id_, "values": rand_vec(), "metadata": safe_meta(md)}])
        return True
    except Exception as e:
        st.warning(f"Pinecone upsert failed: {e}")
        return False

def fetch_all() -> pd.DataFrame:
    if not index:
        local = st.session_state.get("LOCAL_DATA", {})
        rows = []
        for k, md in local.items():
            rec = dict(md)
            rec["_id"] = k
            rows.append(rec)
        return pd.DataFrame(rows)
    try:
        try:
            stats = index.describe_index_stats()
            if stats and stats.get("total_vector_count", 0) == 0:
                return pd.DataFrame()
        except Exception:
            pass
        res = index.query(vector=rand_vec(), top_k=1000, include_metadata=True)
        if not getattr(res, "matches", None):
            return pd.DataFrame()
        rows = []
        for m in res.matches:
            md = m.metadata or {}
            md["_id"] = m.id
            rows.append(md)
        return pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"Fetch error: {e}")
        return pd.DataFrame()

# ----------------------------
# Small ML helper for marks
# ----------------------------
lin_reg = LinearRegression().fit([[0], [50], [100]], [0, 2.5, 5])

# ----------------------------
# Helper: attendees parse
# ----------------------------
def parse_attendees_field(val):
    if isinstance(val, list):
        return [a.strip().lower() for a in val if isinstance(a, str) and a.strip()]
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s.replace("'", '"'))
                if isinstance(parsed, list):
                    return [a.strip().lower() for a in parsed if isinstance(a, str) and a.strip()]
            except Exception:
                pass
        return [a.strip().lower() for a in s.split(",") if a.strip()]
    return []

# ----------------------------
# Role selection
# ----------------------------
role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client", "HR (Admin)"])
current_month = datetime.now().strftime("%B %Y")

# ----------------------------
# MANAGER
# ----------------------------
if role == "Manager":
    st.header("Manager Dashboard")
    tabs = st.tabs(["Task Management", "Feedback & Review", "Meetings", "Leave Approvals", "Team Overview", "AI Insights"])

    # --- Task Management ----
    with tabs[0]:
        st.subheader("Assign Task")
        with st.form("assign_task"):
            company = st.text_input("Client Company Name")
            department = st.text_input("Department")
            employee = st.text_input("Employee Name")
            task_title = st.text_input("Task Title")
            description = st.text_area("Description")
            deadline = st.date_input("Deadline", value=date.today())
            submit = st.form_submit_button("Assign Task")
            if submit:
                if not employee or not task_title:
                    st.warning("Employee and Task Title required.")
                else:
                    tid = str(uuid.uuid4())
                    md = {
                        "type": "Task",
                        "company": company or "",
                        "department": department or "",
                        "employee": employee,
                        "task": task_title,
                        "description": description or "",
                        "deadline": str(deadline),
                        "completion": 0,
                        "marks": 0,
                        "status": "Assigned",
                        "created": now()
                    }
                    upsert_data(tid, md)
                    st.success(f"Assigned '{task_title}' to {employee}")

        df_all = fetch_all()
        df_tasks = pd.DataFrame()
        if not df_all.empty:
            if "type" in df_all.columns:
                df_tasks = df_all[df_all["type"] == "Task"]
            else:
                df_tasks = df_all[df_all.get("task", "").astype(str) != ""]
        if df_tasks.empty:
            st.info("No tasks found.")
        else:
            df_tasks["completion"] = pd.to_numeric(df_tasks["completion"], errors="coerce").fillna(0)
            st.dataframe(df_tasks[["company","employee","department","task","status","completion","deadline"]], use_container_width=True)

    # --- Manager Feedback & Marks ---
    with tabs[1]:
        st.subheader("Manager Feedback & Marks Review")
        df_all = fetch_all()
        if df_all.empty:
            st.info("No data available.")
        else:
            df_tasks = df_all[df_all.get("type") == "Task"]
            if df_tasks.empty:
                st.info("No tasks found.")
            else:
                df_tasks["completion"] = pd.to_numeric(df_tasks.get("completion", 0), errors="coerce").fillna(0)
                pending_review = df_tasks[(df_tasks["completion"] >= 100) & (df_tasks.get("status") != "Under Client Review")]
                if pending_review.empty:
                    st.info("No completed tasks awaiting review.")
                else:
                    task_sel = st.selectbox("Select completed task", pending_review["task"].dropna().unique())
                    row = pending_review[pending_review["task"] == task_sel].iloc[0].to_dict()
                    st.write(f"Employee: {row.get('employee','')}  |  Company: {row.get('company','')}")
                    st.write(f"Task: {row.get('task','')}")
                    final_marks = st.slider("Final Marks (0–5)", 0.0, 5.0, float(row.get("marks", 0)))
                    manager_feedback = st.text_area("Manager Comments")
                    if st.button("Finalize & Send to Client"):
                        row["marks"] = final_marks
                        row["manager_feedback"] = manager_feedback
                        row["manager_reviewed_on"] = now()
                        row["status"] = "Under Client Review"
                        upsert_data(row.get("_id") or str(uuid.uuid4()), row)
                        st.success("Task reviewed and sent for client evaluation.")
