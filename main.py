# ============================================================
# main.py — Enterprise Workforce Performance Intelligence System ⚙️
# ============================================================
# ✅ Requirements:
# pip install streamlit pinecone pandas scikit-learn plotly openpyxl PyPDF2 tqdm huggingface-hub

import streamlit as st
import pandas as pd
import numpy as np
import json, uuid, time
from datetime import datetime, date
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import plotly.express as px

# --- Optional modules ---
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

# ============================================================
# Streamlit UI Config
# ============================================================
st.set_page_config(page_title="Enterprise Workforce Performance Intelligence System ⚙️", layout="wide")
st.title("🏢 Enterprise Workforce Performance Intelligence System")
st.caption("AI-Driven Insights • Performance Analytics • Organizational Intelligence")

st.markdown("""
<div style='background-color:#f0f4ff;padding:10px;border-radius:10px;margin-top:10px;'>
<h4 style='color:#1a1a1a;margin:0;'>A next-generation platform integrating workforce analytics, HR intelligence, and AI-driven recommendations to enhance productivity and decision-making.</h4>
</div>
""", unsafe_allow_html=True)

# ============================================================
# API Keys / Constants
# ============================================================
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
HF_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")
INDEX_NAME = "task"
DIMENSION = 1024  # demo dimension

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
        st.warning(f"Pinecone connection failed — running in local mode. ({e})")
else:
    st.warning("Pinecone API key not detected — operating in local data mode.")

# ============================================================
# Utility Functions
# ============================================================
def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def rand_vec():
    np.random.seed(int(time.time()) % 10000)
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
                clean[k] = str(v)
        except Exception:
            clean[k] = str(v)
    return clean

def upsert_data(id_, md: dict):
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
        rows = []
        for m in res["matches"]:
            md = m["metadata"]
            md["_id"] = m["id"]
            rows.append(md)
        return pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"Fetch failed: {e}")
        return pd.DataFrame()

def parse_attendees_field(val):
    if isinstance(val, list):
        return [a.strip().lower() for a in val]
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("["):
            try:
                return [a.strip().lower() for a in json.loads(s.replace("'", '"'))]
            except Exception:
                pass
        return [a.strip().lower() for a in s.split(",") if a.strip()]
    return []

# ============================================================
# ML Helper
# ============================================================
lin_reg = LinearRegression().fit([[0], [50], [100]], [0, 2.5, 5])

# ============================================================
# Role-based Views
# ============================================================
role = st.sidebar.selectbox("Access Portal As", ["Manager", "Team Member", "Client", "HR Administrator"])

# ---------------- MANAGER ----------------
if role == "Manager":
    st.header("👨‍💼 Manager Command Center — Task & Team Oversight")
    tabs = st.tabs(["Task Management", "Feedback", "Meetings", "Leave Decisions", "Team Overview", "Cognitive Insights"])

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
            submit = st.form_submit_button("Assign Task")
            if submit and emp and task:
                tid = str(uuid.uuid4())
                md = {
                    "type": "Task", "company": company, "department": dept,
                    "employee": emp, "task": task, "description": desc,
                    "deadline": str(deadline), "completion": 0, "marks": 0,
                    "status": "Assigned", "created": now()
                }
                upsert_data(tid, md)
                st.success(f"✅ Task '{task}' assigned to {emp}")

        df = fetch_all()
        if not df.empty:
            df_tasks = df[df.get("type") == "Task"]
            if not df_tasks.empty:
                st.dataframe(df_tasks[["employee","task","status","completion","deadline"]], use_container_width=True)

    # Feedback
    with tabs[1]:
        st.subheader("Manager Feedback & Evaluation")
        df = fetch_all()
        if not df.empty:
            df_tasks = df[df.get("type") == "Task"]
            completed = df_tasks[df_tasks["completion"].astype(float) >= 100]
            if not completed.empty:
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
            time_meet = st.text_input("Time", "10:00 AM")
            attendees = st.text_area("Attendees (comma-separated)")
            submit = st.form_submit_button("Schedule Meeting")
            if submit:
                mid = str(uuid.uuid4())
                md = {
                    "type": "Meeting", "meeting_title": title, "meeting_date": str(date_meet),
                    "meeting_time": time_meet, "attendees": json.dumps(parse_attendees_field(attendees)), "created": now()
                }
                upsert_data(mid, md)
                st.success(f"✅ Meeting '{title}' successfully scheduled.")

    # Leave Approvals
    with tabs[3]:
        st.subheader("Leave Decision Center")
        df = fetch_all()
        leaves = df[df["type"] == "Leave"] if not df.empty else pd.DataFrame()
        if leaves.empty:
            st.info("No pending leave requests.")
        else:
            for i, r in leaves.iterrows():
                if r.get("status") == "Pending":
                    emp = r.get("employee")
                    st.markdown(f"**{emp}** → {r.get('from')} to {r.get('to')} ({r.get('reason')})")
                    dec = st.radio(f"Decision for {emp}", ["Approve", "Reject"], key=f"dec_{i}")
                    if st.button(f"Submit Decision for {emp}", key=f"btn_{i}"):
                        r["status"] = "Approved" if dec == "Approve" else "Rejected"
                        r["approved_on"] = now()
                        upsert_data(r["_id"] if "_id" in r else uuid.uuid4(), r)
                        st.success(f"Leave {r['status']} for {emp}")

    # Team Overview
    with tabs[4]:
        st.subheader("Team Overview — Task Performance Snapshot")
        df = fetch_all()
        tasks = df[df["type"] == "Task"] if not df.empty else pd.DataFrame()
        if tasks.empty:
            st.info("No task data available.")
        else:
            fig = px.bar(tasks, x="employee", y="completion", color="department", title="Task Completion by Employee")
            st.plotly_chart(fig, use_container_width=True)

    # AI Insights
    with tabs[5]:
        st.subheader("Cognitive Insights — Workforce Analytics")
        df_all = fetch_all()
        if df_all.empty:
            st.info("No data available for insights.")
        else:
            q = st.text_input("Ask AI (e.g., 'Who is underperforming this week?')")
            if st.button("Generate Insight"):
                if HF_AVAILABLE and HF_TOKEN:
                    try:
                        client = InferenceClient(token=HF_TOKEN)
                        summary = df_all.describe(include="all").to_dict()
                        prompt = f"You are an HR data analyst. Dataset summary:\n{summary}\nQuestion: {q}\nProvide concise insights."
                        res = client.text_generation(
                            model="microsoft/Phi-3-mini-4k-instruct",
                            inputs=prompt,
                            max_new_tokens=200
                        )
                        st.write(res)
                    except Exception as e:
                        st.error(f"AI query failed: {e}")
                else:
                    st.warning("⚠️ Hugging Face token not configured — add your token in Streamlit secrets.")

# ---------------- TEAM MEMBER ----------------
elif role == "Team Member":
    st.header("🧑‍💻 Employee Workspace — Task Progress & Leave Requests")
    name = st.text_input("Enter your name")
    if name:
        df = fetch_all()
        if not df.empty:
            my = df[(df["type"] == "Task") & (df["employee"].str.lower() == name.lower())]
            for _, r in my.iterrows():
                st.markdown(f"**{r['task']}** — {r['status']}")
                val = st.slider("Progress %", 0, 100, int(float(r["completion"])), key=r["_id"])
                if st.button(f"Update {r['task']}", key=f"upd_{r['_id']}"):
                    r["completion"] = val
                    r["marks"] = float(lin_reg.predict([[val]])[0])
                    r["status"] = "In Progress" if val < 100 else "Completed"
                    upsert_data(r["_id"], r)
                    st.success("✅ Progress updated successfully.")

# ---------------- CLIENT ----------------
elif role == "Client":
    st.header("🏢 Client Review Portal — Project Feedback & Meetings")
    company = st.text_input("Enter Company Name")
    if company:
        df = fetch_all()
        df_client = df[df["company"].str.lower() == company.lower()] if not df.empty else pd.DataFrame()
        if not df_client.empty:
            df_tasks = df_client[df_client["type"] == "Task"]
            st.dataframe(df_tasks[["employee","task","status","completion","marks"]], use_container_width=True)

# ---------------- HR ADMIN ----------------
elif role == "HR Administrator":
    st.header("👩‍💼 HR Analytics — Performance & Leave Intelligence")
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
                st.info("Not enough task data for clustering.")
        with tabs[1]:
            st.dataframe(leaves[["employee","leave_type","from","to","status"]], use_container_width=True)

# ============================================================
# END OF APP
# ============================================================
