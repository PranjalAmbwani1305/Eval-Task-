import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import plotly.express as px
import numpy as np
import pandas as pd
import uuid, json, time
from datetime import datetime, date
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

# Optional modules
try:
    import PyPDF2
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

# ============================================================
# Streamlit UI Config
# ============================================================
st.set_page_config(page_title="Enterprise Workforce Performance Intelligence System", layout="wide")
st.title("Enterprise Workforce Performance Intelligence System")
st.caption("AI-Driven Insights • Performance Analytics • Organizational Intelligence")

# ============================================================
# Constants / Keys
# ============================================================
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
INDEX_NAME = "task"
DIMENSION = 1024

# ============================================================
# Pinecone Setup
# ============================================================
pc, index = None, None
if PINECONE_API_KEY:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        existing = [i["name"] for i in pc.list_indexes()]
        if INDEX_NAME not in existing:
            pc.create_index(
                name=INDEX_NAME, dimension=DIMENSION,
                metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        index = pc.Index(INDEX_NAME)
        st.caption(f"✅ Connected to Pinecone index: {INDEX_NAME}")
    except Exception as e:
        st.warning(f"Pinecone connection failed — running in local mode ({e}).")
else:
    st.warning("⚠️ Pinecone API key missing — running local mode only.")

# ============================================================
# Utility Functions
# ============================================================
def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def rand_vec():
    np.random.seed(int(time.time()) % 10000)
    return np.random.rand(DIMENSION).tolist()

def safe_meta(md):
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

def upsert_data(id_, md):
    id_ = str(id_)
    if not index:
        local = st.session_state.setdefault("LOCAL_DATA", {})
        local[id_] = md
        return True
    try:
        index.upsert(vectors=[{"id": id_, "values": rand_vec(), "metadata": safe_meta(md)}])
        return True
    except Exception as e:
        st.warning(f"Pinecone upsert failed: {e}")
        return False

def fetch_all():
    if not index:
        local = st.session_state.get("LOCAL_DATA", {})
        if not local:
            return pd.DataFrame()
        return pd.DataFrame([{"_id": k, **v} for k, v in local.items()])
    try:
        res = index.query(vector=rand_vec(), top_k=1000, include_metadata=True)
        rows = []
        for m in res["matches"]:
            md = m["metadata"]; md["_id"] = m["id"]
            rows.append(md)
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

def extract_file_text(fobj):
    try:
        name = fobj.name.lower()
        if name.endswith(".txt"):
            return fobj.read().decode("utf-8", errors="ignore")[:20000]
        if name.endswith(".pdf") and PDF_AVAILABLE:
            reader = PyPDF2.PdfReader(fobj)
            return "\n".join([p.extract_text() or "" for p in reader.pages])[:20000]
    except Exception:
        return ""
    return ""

# ============================================================
# ML Predictors
# ============================================================
lin_reg = LinearRegression().fit([[0], [50], [100]], [0, 2.5, 5])

def predict_ai(df):
    if df.empty:
        return df
    df["completion"] = pd.to_numeric(df["completion"], errors="coerce").fillna(0)
    df["marks"] = pd.to_numeric(df["marks"], errors="coerce").fillna(0)

    # train logistic regression for status
    df["label_status"] = (df["completion"] >= 100).astype(int)
    if df["label_status"].nunique() > 1:
        log_reg = LogisticRegression()
        X = df[["completion"]]; y = df["label_status"]
        log_reg.fit(X, y)
        df["predicted_status"] = log_reg.predict(X)
        df["predicted_status"] = df["predicted_status"].map({1: "Completed", 0: "In Progress"})
    else:
        df["predicted_status"] = np.where(df["completion"] >= 100, "Completed", "In Progress")

    # random forest for deadline risk
    try:
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        if len(df) > 1:
            df["days_left"] = np.random.randint(0, 10, len(df))
            y_risk = np.where((df["completion"] < 50) & (df["days_left"] < 3), 1, 0)
            rf.fit(df[["completion", "days_left"]], y_risk)
            pred = rf.predict(df[["completion", "days_left"]])
            df["predicted_risk"] = np.where(pred == 1, "At Risk", "On Track")
        else:
            df["predicted_risk"] = "On Track"
    except Exception:
        df["predicted_risk"] = "On Track"

    df["predicted_marks"] = np.round((df["completion"] / 100) * 5, 2)
    return df

# ============================================================
# Role-Based UI
# ============================================================
role = st.sidebar.selectbox("Access Portal As", ["Manager", "Team Member", "Client", "HR Administrator"])

# ---------------- MANAGER ----------------
if role == "Manager":
    st.header("Manager Command Center — Task & Team Oversight")
    tabs = st.tabs(["Task Management", "Feedback", "Meetings", "Leave Decisions", "Team Overview"])

    # Task Assignment
    with tabs[0]:
        st.subheader("Assign New Task")
        with st.form("assign_task"):
            company = st.text_input("Company Name")
            dept = st.text_input("Department")
            emp = st.text_input("Employee Name")
            task = st.text_input("Task Title")
            desc = st.text_area("Task Description")
            deadline = st.date_input("Deadline", value=date.today())
            attach = st.file_uploader("Optional Attachment (txt/pdf)", type=["txt","pdf"])
            submit = st.form_submit_button("Assign Task")
            if submit and emp and task:
                tid = str(uuid.uuid4())
                md = {
                    "type": "Task", "company": company, "department": dept,
                    "employee": emp, "task": task, "description": desc,
                    "deadline": str(deadline), "completion": 0, "marks": 0,
                    "status": "Assigned", "created": now()
                }
                if attach:
                    md["attachment"] = extract_file_text(attach)
                upsert_data(tid, md)
                st.success(f"Task '{task}' assigned to {emp}")

        df = fetch_all()
        if not df.empty:
            df_tasks = df[df.get("type") == "Task"]
            if not df_tasks.empty:
                df_pred = predict_ai(df_tasks)
                st.dataframe(df_pred[["employee","task","status","completion","predicted_marks","predicted_status","predicted_risk"]], use_container_width=True)

    # Feedback
    with tabs[1]:
        st.subheader("Manager Feedback & Evaluation")
        df = fetch_all()
        if not df.empty:
            df_tasks = df[df.get("type") == "Task"]
            df_tasks["completion"] = pd.to_numeric(df_tasks["completion"], errors="coerce").fillna(0)
            completed = df_tasks[df_tasks["completion"] >= 100]

            emp_search = st.text_input("Search by employee name (optional)")
            if emp_search:
                emp_search_clean = emp_search.strip().lower()
                completed = completed[completed["employee"].astype(str).str.lower().str.contains(emp_search_clean, na=False)]

            if completed.empty:
                st.info("No matching completed tasks.")
            else:
                sel = st.selectbox("Select completed task", completed["task"].unique())
                marks = st.slider("Final Performance Score (0–5)", 0.0, 5.0, 4.0)
                fb = st.text_area("Manager Feedback")
                if st.button("Finalize Review"):
                    rec = completed[completed["task"] == sel].iloc[0].to_dict()
                    rec["marks"] = marks
                    rec["manager_feedback"] = fb
                    rec["status"] = "Under Client Review"
                    rec["manager_reviewed_on"] = now()
                    upsert_data(rec["_id"] if "_id" in rec else uuid.uuid4(), rec)
                    st.success("✅ Feedback finalized and sent for client evaluation.")

    # Meetings
    with tabs[2]:
        st.subheader("Meeting Scheduler & Notes")
        with st.form("schedule_meet"):
            title = st.text_input("Meeting Title")
            date_meet = st.date_input("Date", date.today())
            attendees = st.text_area("Attendees (comma-separated)")
            note = st.file_uploader("Optional Notes", type=["txt","pdf"])
            submit = st.form_submit_button("Schedule")
            if submit:
                mid = str(uuid.uuid4())
                md = {
                    "type": "Meeting", "meeting_title": title, "meeting_date": str(date_meet),
                    "attendees": attendees, "created": now()
                }
                if note:
                    md["notes"] = extract_file_text(note)
                upsert_data(mid, md)
                st.success("Meeting scheduled.")

    # Leave Decisions
    with tabs[3]:
        st.subheader("Leave Decision Center")
        df = fetch_all()
        leaves = df[df["type"] == "Leave"] if not df.empty else pd.DataFrame()
        pending = leaves[leaves["status"] == "Pending"] if not leaves.empty else pd.DataFrame()
        if pending.empty:
            st.info("No pending leave requests.")
        else:
            for i, r in pending.iterrows():
                emp = r.get("employee")
                st.markdown(f"**{emp}** → {r.get('from')} to {r.get('to')} ({r.get('reason')})")
                dec = st.radio("Decision", ["Approve", "Reject"], key=f"dec_{i}")
                if st.button("Submit", key=f"btn_{i}"):
                    r["status"] = "Approved" if dec == "Approve" else "Rejected"
                    r["approved_on"] = now()
                    upsert_data(r["_id"], r)
                    st.success(f"Leave {r['status']} for {emp}")
                    st.experimental_rerun()

    # Team Overview
    with tabs[4]:
        st.subheader("Team Overview — AI Predictions")
        df = fetch_all()
        tasks = df[df["type"] == "Task"] if not df.empty else pd.DataFrame()
        if tasks.empty:
            st.info("No task data.")
        else:
            df_pred = predict_ai(tasks)
            st.dataframe(df_pred[["employee","completion","predicted_marks","predicted_status","predicted_risk"]], use_container_width=True)

# ---------------- TEAM MEMBER ----------------
elif role == "Team Member":
    st.header("Employee Workspace — Progress & Leave")
    name = st.text_input("Enter your name")
    if name:
        df = fetch_all()
        if not df.empty:
            my = df[(df["type"] == "Task") & (df["employee"].str.lower() == name.lower())]
            for _, r in my.iterrows():
                st.markdown(f"**{r['task']}** — {r['status']}")
                val = st.slider("Progress %", 0, 100, int(float(r["completion"])), key=r["_id"])
                attach = st.file_uploader("Optional Upload", type=["txt","pdf"], key=f"file_{r['_id']}")
                if st.button(f"Update {r['task']}", key=f"upd_{r['_id']}"):
                    r["completion"] = val
                    r["marks"] = float(lin_reg.predict([[val]])[0])
                    r["status"] = "In Progress" if val < 100 else "Completed"
                    if attach:
                        r["attachment"] = extract_file_text(attach)
                    upsert_data(r["_id"], r)
                    st.success("✅ Updated successfully.")
                    st.experimental_rerun()

            st.subheader("My Leave Applications")
            my_leaves = df[(df["type"] == "Leave") & (df["employee"].str.lower() == name.lower())]
            if not my_leaves.empty:
                st.dataframe(my_leaves[["leave_type","from","to","status"]], use_container_width=True)

            st.markdown("---")
            st.subheader("Apply for Leave")
            lt = st.selectbox("Leave Type", ["Casual","Sick","Earned"])
            f = st.date_input("From")
            t = st.date_input("To")
            reason = st.text_area("Reason")
            if st.button("Submit Leave"):
                lid = str(uuid.uuid4())
                md = {"type":"Leave","employee":name,"leave_type":lt,"from":str(f),"to":str(t),"reason":reason,"status":"Pending","submitted":now()}
                upsert_data(lid, md)
                st.success("Leave requested.")

# ---------------- CLIENT ----------------
elif role == "Client":
    st.header("Client Review Portal")
    company = st.text_input("Enter Company Name")
    if company:
        df = fetch_all()
        df_client = df[df["company"].str.lower() == company.lower()] if not df.empty else pd.DataFrame()
        if not df_client.empty:
            df_tasks = df_client[df_client["type"] == "Task"]
            df_pred = predict_ai(df_tasks)
            st.dataframe(df_pred[["employee","task","completion","predicted_marks","predicted_status","predicted_risk"]], use_container_width=True)

            st.markdown("---")
            st.subheader("Review Completed Tasks")
            completed = df_tasks[df_tasks["status"] == "Under Client Review"]
            if not completed.empty:
                sel = st.selectbox("Select task", completed["task"].unique())
                fb = st.text_area("Feedback")
                rating = st.slider("Rating (1–5)", 1, 5, 4)
                if st.button("Submit Feedback"):
                    rec = completed[completed["task"] == sel].iloc[0].to_dict()
                    rec["client_feedback"] = fb
                    rec["client_rating"] = rating
                    rec["status"] = "Client Approved"
                    rec["client_reviewed"] = True
                    rec["client_reviewed_on"] = now()
                    upsert_data(rec["_id"], rec)
                    st.success("Feedback submitted.")

# ---------------- HR ----------------
elif role == "HR Administrator":
    st.header("HR Analytics — Performance & Leave Intelligence")
    df = fetch_all()
    if not df.empty:
        tasks = df[df["type"] == "Task"]
        leaves = df[df["type"] == "Leave"]
        tabs = st.tabs(["Performance Clusters", "Leave Tracker", "Export"])
        with tabs[0]:
            df_pred = predict_ai(tasks)
            if len(df_pred) > 1:
                km = KMeans(n_clusters=min(3, len(df_pred)), random_state=42, n_init=10)
                df_pred["cluster"] = km.fit_predict(df_pred[["completion","marks"]])
            else:
                df_pred["cluster"] = 0
            st.dataframe(df_pred[["employee","completion","marks","predicted_risk","cluster"]], use_container_width=True)
        with tabs[1]:
            st.dataframe(leaves[["employee","leave_type","from","to","status"]], use_container_width=True)
        with tabs[2]:
            csv = df_pred.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV Export", data=csv, file_name="workforce_performance.csv", mime="text/csv")

# ============================================================
# END
# ============================================================
