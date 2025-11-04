import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import uuid
from datetime import datetime, date
from sklearn.linear_model import LinearRegression, LogisticRegression
import pandas as pd
import io
import plotly.express as px
import traceback

# --------------------------
# Init + Pinecone
# --------------------------
st.set_page_config(page_title="AI Task Review System", layout="wide")
st.title("✅ AI-Powered Task Completion & Review System")

try:
    PC_API_KEY = st.secrets["PINECONE_API_KEY"]
    pc = Pinecone(api_key=PC_API_KEY)
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
except Exception as e:
    st.error("❌ Pinecone connection failed. Check API key or internet connection.")
    st.stop()

# --------------------------
# Helpers
# --------------------------
def now_ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def random_vector(dim=DIMENSION):
    return np.random.rand(dim).tolist()

def safe_metadata(md: dict):
    clean = {}
    for k, v in md.items():
        if v is None:
            v = ""
        elif isinstance(v, (datetime, date)):
            v = v.isoformat()
        elif isinstance(v, (np.generic,)):
            v = v.item()
        elif isinstance(v, (float, int, bool, str)):
            pass
        else:
            v = str(v)
        clean[k] = v
    return clean

def fetch_all_tasks(top_k=1000):
    """Safely load all tasks from Pinecone"""
    try:
        res = index.query(vector=random_vector(), top_k=top_k, include_metadata=True)
        matches = res.matches or []
        rows = []
        for m in matches:
            md = m.metadata or {}
            md["_id"] = m.id
            rows.append(md)
        df = pd.DataFrame(rows)
        if "_id" not in df.columns:
            df["_id"] = ""
        return df
    except Exception as e:
        st.error("❌ Error fetching data from Pinecone.")
        st.code(traceback.format_exc())
        return pd.DataFrame()

def compute_perf_category(avg):
    if avg >= 4:
        return "High"
    if avg >= 2.5:
        return "Medium"
    return "Low"

# --------------------------
# Role selection
# --------------------------
role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client", "Admin"])
current_month = datetime.now().strftime("%B %Y")

# --------------------------
# Manager Section
# --------------------------
if role == "Manager":
    st.header("👨‍💼 Manager — Assign Tasks & Final Review")

    st.subheader("Assign New Task")
    with st.form("assign_task"):
        company = st.text_input("Company")
        employee = st.text_input("Employee")
        task_title = st.text_input("Task Title")
        description = st.text_area("Description")
        deadline = st.date_input("Deadline", value=date.today())
        month = current_month
        submit = st.form_submit_button("Assign Task")
        if submit:
            try:
                tid = str(uuid.uuid4())
                md = safe_metadata({
                    "company": company,
                    "employee": employee,
                    "task": task_title,
                    "description": description,
                    "deadline": deadline.isoformat(),
                    "month": month,
                    "completion": 0,
                    "marks": 0,
                    "status": "Assigned",
                    "assigned_on": now_ts(),
                    "client_approved": False,
                    "reviewed": False
                })
                index.upsert([{"id": tid, "values": random_vector(), "metadata": md}])
                st.success(f"✅ Task '{task_title}' assigned to {employee}")
            except Exception as e:
                st.error("❌ Failed to assign task.")
                st.code(traceback.format_exc())

    st.subheader("Review Client-Approved Tasks")
    company_review = st.text_input("Company Name for Review")
    if st.button("Load Client-Approved Tasks"):
        try:
            res = index.query(
                vector=random_vector(),
                top_k=500,
                include_metadata=True,
                include_values=True,
                filter={"company": {"$eq": company_review}, "client_approved": {"$eq": True}, "reviewed": {"$eq": False}}
            )
            st.session_state["mgr_tasks"] = [(m.id, m.metadata, m.values) for m in (res.matches or [])]
            st.success(f"Loaded {len(st.session_state['mgr_tasks'])} client-approved tasks.")
        except Exception as e:
            st.error("❌ Error loading client-approved tasks.")
            st.code(traceback.format_exc())

    if "mgr_tasks" in st.session_state and st.session_state["mgr_tasks"]:
        for tid, md, vals in st.session_state["mgr_tasks"]:
            st.markdown(f"### {md.get('task')}")
            st.write(f"Employee: {md.get('employee')}")
            marks = st.number_input(f"Final Marks for {md.get('task')}", 0.0, 5.0, step=0.1)
            comments = st.text_area(f"Manager Comments for {md.get('task')}")
            if st.button(f"Finalize Review: {md.get('task')}"):
                try:
                    md["marks"] = marks
                    md["manager_comments"] = comments
                    md["reviewed"] = True
                    md["status"] = "Completed"
                    md["reviewed_on"] = now_ts()
                    index.upsert([{"id": tid, "values": vals or random_vector(), "metadata": safe_metadata(md)}])
                    st.success(f"✅ Final review done for {md.get('task')}")
                except Exception as e:
                    st.error("❌ Review update failed.")
                    st.code(traceback.format_exc())

# --------------------------
# Team Member Section
# --------------------------
elif role == "Team Member":
    st.header("👷 Team Member — Update Progress")
    company = st.text_input("Company")
    employee = st.text_input("Your Name")
    month_input = st.text_input("Month", value=current_month)

    if st.button("Load My Tasks"):
        try:
            res = index.query(
                vector=random_vector(),
                top_k=500,
                include_metadata=True,
                include_values=True,
                filter={"company": {"$eq": company}, "employee": {"$eq": employee}, "month": {"$eq": month_input}}
            )
            st.session_state["tm_tasks"] = [(m.id, m.metadata, m.values) for m in (res.matches or [])]
            st.success(f"Loaded {len(st.session_state['tm_tasks'])} tasks.")
        except Exception as e:
            st.error("❌ Error loading tasks.")
            st.code(traceback.format_exc())

    if "tm_tasks" in st.session_state and st.session_state["tm_tasks"]:
        for tid, md, vals in st.session_state["tm_tasks"]:
            st.markdown(f"### {md.get('task')}")
            completion = st.slider(f"Completion (%) for {md.get('task')}", 0, 100, int(md.get("completion", 0)))
            if st.button(f"Submit Progress for {md.get('task')}"):
                try:
                    md["completion"] = completion
                    md["submitted_on"] = now_ts()
                    md["client_approved"] = False  # Reset client approval when updated
                    index.upsert([{"id": tid, "values": vals or random_vector(), "metadata": safe_metadata(md)}])
                    st.success(f"✅ Progress updated for {md.get('task')}")
                except Exception as e:
                    st.error("❌ Progress submission failed.")
                    st.code(traceback.format_exc())

# --------------------------
# Client Section
# --------------------------
elif role == "Client":
    st.header("💼 Client — Approve Completed Tasks")
    company = st.text_input("Company")
    if st.button("Load Completed Tasks"):
        try:
            res = index.query(
                vector=random_vector(),
                top_k=500,
                include_metadata=True,
                include_values=True,
                filter={"company": {"$eq": company}, "completion": {"$gte": 99}, "client_approved": {"$eq": False}}
            )
            st.session_state["client_tasks"] = [(m.id, m.metadata, m.values) for m in (res.matches or [])]
            st.success(f"Loaded {len(st.session_state['client_tasks'])} tasks for review.")
        except Exception as e:
            st.error("❌ Failed to load client tasks.")
            st.code(traceback.format_exc())

    if "client_tasks" in st.session_state and st.session_state["client_tasks"]:
        for tid, md, vals in st.session_state["client_tasks"]:
            st.markdown(f"### {md.get('task')}")
            comments = st.text_area(f"Feedback for {md.get('task')}")
            if st.button(f"Approve {md.get('task')}"):
                try:
                    md["client_approved"] = True
                    md["client_comments"] = comments
                    md["client_approved_on"] = now_ts()
                    index.upsert([{"id": tid, "values": vals or random_vector(), "metadata": safe_metadata(md)}])
                    st.success(f"✅ Task '{md.get('task')}' approved.")
                except Exception as e:
                    st.error("❌ Approval update failed.")
                    st.code(traceback.format_exc())

# --------------------------
# Admin Section
# --------------------------
elif role == "Admin":
    st.header("🧾 Admin — Overview & Export")
    df = fetch_all_tasks()
    if df.empty:
        st.warning("No tasks found in Pinecone.")
    else:
        st.dataframe(df)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", csv, "all_tasks.csv", "text/csv")

# --------------------------
# End
# --------------------------
