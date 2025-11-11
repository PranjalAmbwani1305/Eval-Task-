# ============================================================
# main.py — Enterprise Workforce Intelligence System (Final)
# ============================================================
# Requirements:
# pip install streamlit pinecone-client scikit-learn plotly pandas openpyxl PyPDF2 huggingface-hub

import streamlit as st
import pandas as pd
import numpy as np
import json, uuid, time
from datetime import datetime, date
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import plotly.express as px

# --- Optional imports ---
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except Exception:
    PINECONE_AVAILABLE = False

# ============================================================
# Streamlit UI Config
# ============================================================
st.set_page_config(page_title="Enterprise Workforce Intelligence System", layout="wide")
st.title("Enterprise Workforce Intelligence System")
st.caption("AI-driven workforce analytics, client reviews, and HR intelligence")

# ============================================================
# API Keys / Constants
# ============================================================
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
INDEX_NAME = "task"
DIMENSION = 1024

# ============================================================
# Pinecone Setup
# ============================================================
pc, index = None, None
if PINECONE_AVAILABLE and PINECONE_API_KEY:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        existing = [i["name"] for i in pc.list_indexes()]
        if INDEX_NAME not in existing:
            INDEX_NAME = existing[0]
        index = pc.Index(INDEX_NAME)
        st.caption(f"✅ Connected to Pinecone index: {INDEX_NAME}")
    except Exception as e:
        st.warning(f"Pinecone connection failed — running local mode. ({e})")
else:
    st.warning("Pinecone API key not detected — running in local mode.")

# ============================================================
# Utilities
# ============================================================
def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def rand_vec():
    np.random.seed(int(time.time()) % 10000)
    return np.random.rand(DIMENSION).tolist()

def safe_meta(md):
    """Ensure metadata is 100% Pinecone-safe JSON."""
    if isinstance(md, pd.Series):
        md = md.to_dict()
    clean = {}
    for k, v in (md or {}).items():
        try:
            if isinstance(v, (np.generic,)):
                v = v.item()
            if isinstance(v, (datetime, date)):
                v = v.isoformat()
            if isinstance(v, (dict, list)):
                v = json.dumps(v)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                v = ""
            clean[str(k)] = str(v)
        except Exception:
            clean[str(k)] = str(v)
    return clean

def upsert_data(id_, md):
    """Safely upsert to Pinecone or local cache."""
    id_ = str(id_)
    payload = safe_meta(md)
    if not index:
        local = st.session_state.setdefault("LOCAL_DATA", {})
        local[id_] = payload
        return True
    try:
        index.upsert(vectors=[{"id": id_, "values": rand_vec(), "metadata": payload}])
        time.sleep(0.5)
        return True
    except Exception as e:
        st.error(f"❌ Pinecone upsert failed: {e}")
        return False

def fetch_all() -> pd.DataFrame:
    """Fetch all records from Pinecone or local cache."""
    if not index:
        local = st.session_state.get("LOCAL_DATA", {})
        if not local:
            return pd.DataFrame()
        return pd.DataFrame([{"_id": k, **v} for k, v in local.items()])
    try:
        res = index.query(vector=rand_vec(), top_k=1000, include_metadata=True)
        rows = []
        for m in res.get("matches", []):
            md = m.get("metadata", {})
            md["_id"] = m.get("id", str(uuid.uuid4()))
            rows.append(md)
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

# ============================================================
# ML Models
# ============================================================
lin_reg = LinearRegression().fit([[0], [50], [100]], [0, 2.5, 5])
log_reg = LogisticRegression().fit([[0], [50], [100]], [0, 1, 1])
vectorizer = TfidfVectorizer()
sample_texts = ["good job", "excellent performance", "poor work", "needs improvement"]
sample_labels = [1, 1, 0, 0]
X_train = vectorizer.fit_transform(sample_texts)
svm_clf = SVC()
svm_clf.fit(X_train, sample_labels)

# ============================================================
# Role Selection
# ============================================================
role = st.sidebar.selectbox("Access Portal As", ["Manager", "Team Member", "Client", "HR Administrator"])

# ============================================================
# MANAGER
# ============================================================
if role == "Manager":
    st.header("Manager Dashboard")
    tabs = st.tabs(["Task Management", "Feedback", "Leave", "Overview"])

    # --- Task Assignment ---
    with tabs[0]:
        st.subheader("Assign New Task")
        with st.form("assign_task"):
            company = st.text_input("Company Name")
            dept = st.text_input("Department")
            emp = st.text_input("Employee Name")
            task = st.text_input("Task Title")
            desc = st.text_area("Description")
            deadline = st.date_input("Deadline", value=date.today())
            submit = st.form_submit_button("Assign Task")
            if submit and emp and task:
                tid = str(uuid.uuid4())
                md = {
                    "type": "Task",
                    "company": company, "department": dept,
                    "employee": emp, "task": task, "description": desc,
                    "deadline": str(deadline), "completion": 0, "marks": 0,
                    "status": "Assigned", "created": now()
                }
                upsert_data(tid, md)
                st.success(f"Task '{task}' assigned to {emp}")

    # --- Manager Feedback & AI Evaluation ---
    with tabs[1]:
        st.subheader("Manager Feedback & Evaluation")

        df = fetch_all()
        if df.empty:
            st.info("No data found.")
        else:
            df["type"] = df["type"].astype(str).str.replace('"', '').str.strip()
            df["completion"] = pd.to_numeric(df["completion"], errors="coerce").fillna(0)
            df.loc[df["completion"] >= 100, "status"] = "Completed"
            df.loc[(df["completion"] >= 75) & (df["completion"] < 100), "status"] = "Ready for Review"

            df_tasks = df[df["type"].str.lower() == "task"]
            completed = df_tasks[df_tasks["completion"] >= 75].to_dict("records")

            if not completed:
                st.success("✅ No pending tasks for review.")
            else:
                with st.form(key="manager_review_form"):
                    for i, rec in enumerate(completed):
                        emp = rec.get("employee", "?")
                        task = rec.get("task", "?")
                        comp = float(rec.get("completion", 0))
                        st.write(f"👤 **{emp}** | Task: **{task}**")
                        st.slider(f"Completion ({emp})", 0, 100, int(comp), key=f"comp_{i}")
                        st.text_area(f"Manager Comments ({task})", key=f"comm_{i}")

                    submit = st.form_submit_button("💾 Save All Reviews")

                    if submit:
                        for i, rec in enumerate(completed):
                            rec_id = str(rec.get("_id", uuid.uuid4()))
                            completion = st.session_state[f"comp_{i}"]
                            comments = st.session_state[f"comm_{i}"]
                            predicted_marks = float(lin_reg.predict([[completion]])[0])
                            status_val = log_reg.predict([[completion]])[0]
                            status_text = "On Track" if status_val == 1 else "Delayed"

                            sentiment_text = "N/A"
                            if comments.strip():
                                try:
                                    X_new = vectorizer.transform([comments])
                                    sentiment = svm_clf.predict(X_new)[0]
                                    sentiment_text = "Positive" if sentiment == 1 else "Negative"
                                except Exception:
                                    pass

                            updated = {
                                **rec,
                                "completion": completion,
                                "marks": predicted_marks,
                                "status": status_text,
                                "manager_feedback": comments,
                                "sentiment": sentiment_text,
                                "manager_reviewed_on": now()
                            }
                            upsert_data(rec_id, updated)
                        st.success("✅ All reviews saved to Pinecone!")

    # --- Leave Approval ---
    with tabs[2]:
        st.subheader("Leave Approvals")
        df = fetch_all()
        leaves = df[df["type"] == "Leave"] if not df.empty else pd.DataFrame()
        if leaves.empty:
            st.info("No leave requests.")
        else:
            for i, r in leaves.iterrows():
                if r.get("status") == "Pending":
                    emp = r.get("employee")
                    st.markdown(f"**{emp}** → {r.get('from')} to {r.get('to')} ({r.get('reason')})")
                    dec = st.radio(f"Decision for {emp}", ["Approve", "Reject"], key=f"dec_{i}")
                    if st.button(f"Submit Decision for {emp}", key=f"btn_{i}"):
                        r["status"] = "Approved" if dec == "Approve" else "Rejected"
                        r["approved_on"] = now()
                        upsert_data(r.get("_id", str(uuid.uuid4())), r)
                        st.success(f"Leave {r['status']} for {emp}")
                        st.experimental_rerun()

    # --- Team Overview ---
    with tabs[3]:
        df = fetch_all()
        if not df.empty:
            df_tasks = df[df["type"] == "Task"]
            if not df_tasks.empty:
                df_tasks["completion"] = pd.to_numeric(df_tasks["completion"], errors="coerce").fillna(0)
                fig = px.bar(df_tasks, x="employee", y="completion", color="department",
                             title="Task Completion by Employee")
                st.plotly_chart(fig, use_container_width=True)
            st.download_button("⬇️ Export Tasks to CSV", df_tasks.to_csv(index=False), "tasks.csv")

# ============================================================
# TEAM MEMBER
# ============================================================
elif role == "Team Member":
    st.header("Team Member Dashboard")
    name = st.text_input("Enter your name")
    if name:
        df = fetch_all()
        if not df.empty:
            my_tasks = df[(df["type"] == "Task") & (df["employee"].str.lower() == name.lower())]
            for _, r in my_tasks.iterrows():
                st.markdown(f"**{r['task']}** — {r['status']}")
                val = st.slider("Progress %", 0, 100, int(float(r["completion"])), key=r["_id"])
                if st.button(f"Update {r['task']}", key=f"upd_{r['_id']}"):
                    r["completion"] = val
                    r["marks"] = float(lin_reg.predict([[val]])[0])
                    r["status"] = "In Progress" if val < 100 else "Completed"
                    upsert_data(r["_id"], r)
                    st.success("Progress updated.")

            st.markdown("---")
            st.subheader("Leave Request")
            lt = st.selectbox("Leave Type", ["Casual", "Sick", "Earned"])
            f = st.date_input("From")
            t = st.date_input("To")
            reason = st.text_area("Reason")
            if st.button("Submit Leave Request"):
                lid = str(uuid.uuid4())
                md = {
                    "type": "Leave",
                    "employee": name,
                    "leave_type": lt,
                    "from": str(f),
                    "to": str(t),
                    "reason": reason,
                    "status": "Pending",
                    "submitted": now()
                }
                upsert_data(lid, md)
                st.success("Leave requested successfully!")

# ============================================================
# CLIENT
# ============================================================
elif role == "Client":
    st.header("Client Review Portal")
    company = st.text_input("Enter Company Name")
    if company:
        df = fetch_all()
        df_client = df[df["company"].str.lower() == company.lower()] if not df.empty else pd.DataFrame()
        if not df_client.empty:
            df_tasks = df_client[df_client["type"] == "Task"].copy()
            st.dataframe(df_tasks[["employee", "task", "completion", "marks"]])

            st.subheader("Review Completed Tasks")
            reviewable = df_tasks[df_tasks["completion"] >= 75]
            if reviewable.empty:
                st.info("No tasks ready for review.")
            else:
                sel = st.selectbox("Select Task", reviewable["task"].unique())
                fb = st.text_area("Client Feedback")
                rating = st.slider("Rating (1–5)", 1, 5, 3)
                if st.button("Submit Client Review"):
                    rec = reviewable[reviewable["task"] == sel].iloc[0].to_dict()
                    rec["client_feedback"] = fb
                    rec["client_rating"] = rating
                    rec["client_reviewed_on"] = now()
                    rec["status"] = "Client Approved"
                    upsert_data(rec.get("_id", str(uuid.uuid4())), rec)
                    st.success("Client review submitted successfully!")

# ============================================================
# HR ADMIN
# ============================================================
elif role == "HR Administrator":
    st.header("HR Dashboard")
    df = fetch_all()
    if not df.empty:
        tasks = df[df["type"] == "Task"]
        if not tasks.empty:
            st.subheader("Performance Clustering")
            if len(tasks) >= 3:
                km = KMeans(n_clusters=3, random_state=42, n_init=10)
                tasks["cluster"] = km.fit_predict(tasks[["completion", "marks"]])
                fig = px.scatter(tasks, x="completion", y="marks", color="cluster", hover_data=["employee"])
                st.plotly_chart(fig, use_container_width=True)
            st.download_button("⬇️ Export HR Data", tasks.to_csv(index=False), "hr_data.csv")
