# ============================================================
# main.py — Enterprise Workforce Performance Intelligence System (Final)
# ============================================================
# Requirements:
# pip install streamlit pinecone-client pandas scikit-learn plotly openpyxl PyPDF2 huggingface-hub tqdm

import streamlit as st
import pandas as pd
import numpy as np
import json
import uuid
import time
from datetime import datetime, date
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import plotly.express as px

# ------------------------------------------------------------
# Optional Libraries
# ------------------------------------------------------------
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except Exception:
    PINECONE_AVAILABLE = False

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

# ------------------------------------------------------------
# Streamlit Config
# ------------------------------------------------------------
st.set_page_config(page_title="Enterprise Workforce Intelligence", layout="wide")
st.title("Enterprise Workforce Performance Intelligence System")
st.caption("AI-Driven Workforce Insights | Task, HR, and Performance Intelligence")

# ------------------------------------------------------------
# Constants & Secrets
# ------------------------------------------------------------
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
HF_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")
INDEX_NAME = "task"
DIMENSION = 1024

# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------
def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def rand_vec():
    np.random.seed(int(time.time()) % 10000)
    return np.random.rand(DIMENSION).tolist()

def safe_meta(md: dict) -> dict:
    """Convert metadata safely to JSON-friendly structure."""
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
                clean[k] = str(v)
        except Exception:
            clean[k] = str(v)
    return clean

# ------------------------------------------------------------
# Pinecone Initialization
# ------------------------------------------------------------
pc, index = None, None
if PINECONE_AVAILABLE and PINECONE_API_KEY:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        existing = [i["name"] for i in pc.list_indexes()]
        if INDEX_NAME not in existing:
            st.info(f"Creating Pinecone index '{INDEX_NAME}'...")
            pc.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            for _ in range(30):
                try:
                    desc = pc.describe_index(INDEX_NAME)
                    if desc.get("status", {}).get("ready"):
                        break
                except Exception:
                    time.sleep(1)
        index = pc.Index(INDEX_NAME)
        st.success(f"Connected to Pinecone index: {INDEX_NAME}")
    except Exception as e:
        st.warning(f"Pinecone connection failed — using local mode. ({e})")
else:
    st.warning("Pinecone API key not configured — using local mode.")

if "LOCAL_DATA" not in st.session_state:
    st.session_state["LOCAL_DATA"] = {}

# ------------------------------------------------------------
# Universal Upsert Function
# ------------------------------------------------------------
def upsert_data(id_, md: dict):
    """Safe universal upsert (works with Pinecone + local fallback)."""
    id_ = str(id_).strip()
    md = dict(md)
    md = safe_meta(md)

    # ---- Local fallback ----
    if not index:
        local = st.session_state.setdefault("LOCAL_DATA", {})
        local[id_] = md
        return True

    # ---- Pinecone upsert ----
    try:
        vec = rand_vec()
        index.upsert(vectors=[{"id": id_, "values": vec, "metadata": md}])
        return True
    except Exception as e:
        st.warning(f"Pinecone upsert failed — storing locally instead. ({e})")
        st.session_state["LOCAL_DATA"][id_] = md
        return False

# ------------------------------------------------------------
# Find Existing Record (Prevents Duplicates)
# ------------------------------------------------------------
def find_existing_record(df: pd.DataFrame, type_: str, employee: str, task_title: str = None):
    """Find existing record ID by type, employee, and task (if provided)."""
    if df.empty:
        return None
    subset = df[df["type"] == type_]
    if "employee" in subset.columns:
        subset = subset[subset["employee"].astype(str).str.lower() == employee.lower()]
    if task_title and "task" in subset.columns:
        subset = subset[subset["task"].astype(str).str.lower() == task_title.lower()]
    if subset.empty:
        return None
    return subset.iloc[0].get("_id")

# ------------------------------------------------------------
# Fetch All Records
# ------------------------------------------------------------
@st.cache_data(ttl=60)
def fetch_all() -> pd.DataFrame:
    if not index:
        local = st.session_state.get("LOCAL_DATA", {})
        if not local:
            return pd.DataFrame()
        return pd.DataFrame([{"_id": k, **v} for k, v in local.items()])
    try:
        stats = index.describe_index_stats()
        if stats.get("total_vector_count", 0) == 0:
            return pd.DataFrame()
        res = index.query(vector=rand_vec(), top_k=1000, include_metadata=True)
        matches = getattr(res, "matches", []) or res.get("matches", [])
        rows = []
        for m in matches:
            md = m.metadata if hasattr(m, "metadata") else m.get("metadata", {})
            md = dict(md)
            md["_id"] = m.id if hasattr(m, "id") else m.get("id")
            rows.append(md)
        return pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"Fetch failed: {e}")
        return pd.DataFrame()

# ------------------------------------------------------------
# Linear Regression Helper
# ------------------------------------------------------------
lin_reg = LinearRegression().fit([[0], [50], [100]], [0, 2.5, 5])

# ------------------------------------------------------------
# Role Selection
# ------------------------------------------------------------
role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client", "HR Administrator"])

# ============================================================
# MANAGER DASHBOARD
# ============================================================
if role == "Manager":
    st.header("Manager Dashboard")
    tabs = st.tabs(["Task Management", "Feedback", "Meetings", "Leave Approvals", "Team Overview"])

    # ---------------- Task Management ----------------
    with tabs[0]:
        st.subheader("Assign Task")
        with st.form("assign_task"):
            company = st.text_input("Company Name")
            dept = st.text_input("Department")
            emp = st.text_input("Employee Name")
            task = st.text_input("Task Title")
            desc = st.text_area("Description")
            deadline = st.date_input("Deadline", value=date.today())
            submit = st.form_submit_button("Assign Task")
            if submit and emp and task:
                df_all = fetch_all()
                existing_id = find_existing_record(df_all, "Task", emp, task)
                tid = existing_id or str(uuid.uuid4())
                md = {
                    "type": "Task", "company": company, "department": dept,
                    "employee": emp, "task": task, "description": desc,
                    "deadline": str(deadline), "completion": 0, "marks": 0,
                    "status": "Assigned", "created": now()
                }
                upsert_data(tid, md)
                st.success(f"Task '{task}' assigned to {emp}")

        df = fetch_all()
        if not df.empty:
            df_tasks = df[df.get("type") == "Task"]
            if not df_tasks.empty:
                st.dataframe(df_tasks[["employee","task","status","completion","deadline"]], use_container_width=True)

    # ---------------- Feedback ----------------
    with tabs[1]:
        st.subheader("Manager Feedback")
        df = fetch_all()
        if not df.empty:
            df_tasks = df[df.get("type") == "Task"]
            completed = df_tasks[df_tasks["completion"].astype(float) >= 100]
            if not completed.empty:
                sel = st.selectbox(
                    "Select Completed Task",
                    sorted(list(set(completed["task"].dropna().tolist())))
                )
                marks = st.slider("Final Score (0–5)", 0.0, 5.0, 4.0)
                fb = st.text_area("Manager Feedback")
                if st.button("Finalize Review"):
                    rec = completed[completed["task"] == sel].iloc[0].to_dict()
                    rec["marks"] = marks
                    rec["manager_feedback"] = fb
                    rec["status"] = "Under Client Review"
                    rec["manager_reviewed_on"] = now()
                    existing_id = rec.get("_id") or find_existing_record(fetch_all(), "Task", rec.get("employee",""), rec.get("task",""))
                    tid = existing_id or str(uuid.uuid4())
                    upsert_data(tid, rec)
                    st.success("Feedback finalized and sent for client review.")

    # ---------------- Leave Approvals ----------------
    with tabs[3]:
        st.subheader("Leave Approvals")
        df = fetch_all()
        leaves = df[df.get("type") == "Leave"] if not df.empty else pd.DataFrame()
        if leaves.empty:
            st.info("No pending leave requests.")
        else:
            if "status" not in leaves.columns:
                leaves["status"] = ""
            pending_leaves = leaves[leaves["status"].astype(str).str.lower() == "pending"]
            if pending_leaves.empty:
                st.success("All leave requests processed.")
            else:
                for i, row in pending_leaves.iterrows():
                    emp = row.get("employee", "Unknown")
                    lt = row.get("leave_type", "Leave")
                    from_d = row.get("from", "-")
                    to_d = row.get("to", "-")
                    reason = row.get("reason", "-")
                    st.markdown(f"**{emp}** requested **{lt}** leave ({from_d} → {to_d})")
                    st.write(f"Reason: {reason}")
                    decision = st.radio(f"Decision for {emp}", ["Approve", "Reject"], key=f"dec_{i}")
                    if st.button(f"Finalize Decision for {emp}", key=f"btn_{i}"):
                        rec = row.to_dict()
                        rec["status"] = "Approved" if decision == "Approve" else "Rejected"
                        rec["approved_by"] = "Manager"
                        rec["approved_on"] = now()
                        audit_entry = f"Leave {decision} by Manager on {now()}"
                        rec["audit_log"] = rec.get("audit_log","") + "\n" + audit_entry
                        row_id = rec.get("_id", str(uuid.uuid4()))
                        success = upsert_data(row_id, rec)
                        if success:
                            st.success(f"Leave {rec['status']} for {emp}")
                            time.sleep(1)
                            st.experimental_rerun()

# ============================================================
# TEAM MEMBER DASHBOARD
# ============================================================
elif role == "Team Member":
    st.header("Team Member Dashboard")
    name = st.text_input("Enter your name")
    if name:
        df = fetch_all()
        if not df.empty:
            my = df[(df["type"] == "Task") & (df["employee"].str.lower() == name.lower())]
            st.subheader("My Tasks")
            if my.empty:
                st.info("No tasks assigned.")
            else:
                for _, r in my.iterrows():
                    task_title = r.get("task", "Untitled Task")
                    status = r.get("status", "Unknown")
                    st.markdown(f"**{task_title}** — Status: {status}")
                    manager_fb = r.get("manager_feedback", "")
                    client_fb = r.get("client_feedback", "")
                    client_rating = r.get("client_rating", "")
                    if manager_fb:
                        st.info(f"Manager Feedback: {manager_fb}")
                    if client_fb:
                        st.success(f"Client Feedback: {client_fb} (Rating: {client_rating}/5)")
                    curr = int(float(r.get("completion", 0) or 0))
                    comp = st.slider("Completion %", 0, 100, curr, key=r.get("_id"))
                    if st.button(f"Update {task_title}", key=f"upd_{r.get('_id')}"):
                        r = r.to_dict()
                        r["completion"] = comp
                        r["marks"] = float(lin_reg.predict([[comp]])[0])
                        r["status"] = "In Progress" if comp < 100 else "Completed"
                        upsert_data(r.get("_id"), r)
                        st.success("Progress updated successfully.")
                        st.experimental_rerun()

# ============================================================
# CLIENT DASHBOARD
# ============================================================
elif role == "Client":
    st.header("Client Dashboard")
    company = st.text_input("Enter Company Name")
    if company:
        df = fetch_all()
        df_client = df[df["company"].astype(str).str.lower() == company.lower()] if not df.empty else pd.DataFrame()
        if not df_client.empty:
            df_tasks = df_client[df_client["type"] == "Task"]
            st.dataframe(df_tasks[["employee","task","status","completion","marks"]], use_container_width=True)
            st.markdown("---")
            st.subheader("Provide Feedback for Completed Tasks")
            pending_review = df_tasks[(df_tasks["status"] == "Under Client Review") & (df_tasks.get("client_reviewed") != True)]
            if pending_review.empty:
                st.info("No tasks pending for your review.")
            else:
                task_sel = st.selectbox("Select Task for Feedback", sorted(pending_review["task"].dropna().unique().tolist()))
                fb = st.text_area("Client Feedback")
                rating = st.slider("Rating (1–5)", 1, 5, 3)
                if st.button("Submit Client Feedback"):
                    row = pending_review[pending_review["task"] == task_sel].iloc[0].to_dict()
                    row["client_feedback"] = fb
                    row["client_rating"] = rating
                    row["client_reviewed"] = True
                    row["client_reviewed_on"] = now()
                    row["status"] = "Client Reviewed"
                    row_id = row.get("_id", str(uuid.uuid4()))
                    success = upsert_data(row_id, row)
                    if success:
                        st.success(f"Feedback submitted for '{task_sel}'.")
                        st.experimental_rerun()

# ============================================================
# HR ADMIN DASHBOARD
# ============================================================
elif role == "HR Administrator":
    st.header("HR Administrator Dashboard")
    df = fetch_all()
    if not df.empty:
        tasks = df[df["type"] == "Task"]
        leaves = df[df["type"] == "Leave"]
        tabs = st.tabs(["Performance Clusters", "Leave Tracker"])
        with tabs[0]:
            if len(tasks) > 2:
                km = KMeans(n_clusters=3, random_state=42, n_init=10)
                tasks["cluster"] = km.fit_predict(tasks[["completion","marks"]])
                st.dataframe(tasks[["employee","completion","marks","cluster"]])
                fig = px.scatter(tasks, x="completion", y="marks", color="cluster", hover_data=["employee"])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data for clustering.")
        with tabs[1]:
            st.dataframe(leaves[["employee","leave_type","from","to","status"]], use_container_width=True)

# ============================================================
# END OF APP
# ============================================================
