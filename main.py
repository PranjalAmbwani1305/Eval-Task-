# main.py — AI Workforce Intelligence Platform (Client-Linked Enterprise Edition)

import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
import numpy as np
import uuid, json, time
from datetime import datetime, date
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from huggingface_hub import InferenceClient

# Optional PDF support
try:
    import PyPDF2
    PDF_AVAILABLE = True
except:
    PDF_AVAILABLE = False

# -------------------------------
# Streamlit Setup
# -------------------------------
st.set_page_config(page_title="AI Workforce Intelligence Platform", layout="wide")
st.title("AI Workforce Intelligence Platform")

# -------------------------------
# Config and Constants
# -------------------------------
INDEX_NAME = "task"
DIMENSION = 1024
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
HF_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")

# -------------------------------
# Pinecone Setup
# -------------------------------
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
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        index = pc.Index(INDEX_NAME)
        st.caption(f"✅ Connected to Pinecone index: {INDEX_NAME}")
    except Exception as e:
        st.warning(f"Pinecone init failed — using local mode. ({e})")
else:
    st.warning("⚠️ Pinecone key missing. Running in local mode.")

# -------------------------------
# Helpers
# -------------------------------
def now(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def rand_vec(): return np.random.rand(DIMENSION).tolist()

def safe_meta(md):
    clean = {}
    for k, v in md.items():
        if isinstance(v, (datetime, date)): v = v.isoformat()
        elif isinstance(v, (list, dict)): v = json.dumps(v)
        elif pd.isna(v): v = ""
        clean[k] = v
    return clean

def upsert_data(id_, md):
    if not index:
        st.session_state.setdefault("LOCAL_DATA", {})[id_] = md
        return
    try:
        index.upsert([{"id": id_, "values": rand_vec(), "metadata": safe_meta(md)}])
    except Exception as e:
        st.warning(f"Upsert failed: {e}")

def fetch_all():
    if not index:
        return pd.DataFrame(st.session_state.get("LOCAL_DATA", {}).values())
    try:
        res = index.query(vector=rand_vec(), top_k=1000, include_metadata=True)
        rows = []
        for m in res.matches:
            md = m.metadata or {}
            md["_id"] = m.id
            rows.append(md)
        return pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"Fetch error: {e}")
        return pd.DataFrame()

def hf_generate(prompt, max_new_tokens=200):
    if not HF_TOKEN:
        return "⚠️ Hugging Face token not found."
    try:
        client = InferenceClient(token=HF_TOKEN)
        return client.text_generation(model="mistralai/Mixtral-8x7B-Instruct", inputs=prompt, max_new_tokens=max_new_tokens)
    except Exception as e:
        return f"AI generation failed: {e}"

# -------------------------------
# Simple ML model for marks
# -------------------------------
lin_reg = LinearRegression().fit([[0], [50], [100]], [0, 2.5, 5])

# -------------------------------
# Role Login
# -------------------------------
role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client", "HR (Admin)"])

# -------------------------------
# MANAGER DASHBOARD
# -------------------------------
if role == "Manager":
    st.header("Manager Dashboard")
    tabs = st.tabs(["Task Management", "Feedback & Review", "Meetings", "AI Insights"])

    # ------------------- Task Management -------------------
    with tabs[0]:
        st.subheader("Assign Task to Employee (Linked to Client Company)")
        with st.form("assign_task"):
            company = st.text_input("Client Company Name")
            department = st.text_input("Department")
            employee = st.text_input("Employee Name")
            task = st.text_input("Task Title")
            desc = st.text_area("Task Description")
            deadline = st.date_input("Deadline", value=date.today())
            submit = st.form_submit_button("Assign Task")
            if submit:
                tid = str(uuid.uuid4())
                md = {
                    "type": "Task",
                    "company": company,
                    "department": department,
                    "employee": employee,
                    "task": task,
                    "description": desc,
                    "deadline": str(deadline),
                    "completion": 0,
                    "marks": 0,
                    "status": "Assigned",
                    "created": now()
                }
                upsert_data(tid, md)
                st.success(f"✅ Task '{task}' assigned to {employee} under {company}")

        df = fetch_all()
        if not df.empty:
            df_tasks = df[df["type"] == "Task"]
            if not df_tasks.empty:
                st.dataframe(df_tasks[["company", "employee", "task", "status", "completion"]], use_container_width=True)

    # ------------------- Feedback -------------------
    with tabs[1]:
        st.subheader("Manager Feedback & Review")
        df = fetch_all()
        if not df.empty:
            tasks = df[(df["type"] == "Task") & (df["completion"].astype(float) >= 100)]
            if tasks.empty:
                st.info("No completed tasks to review.")
            else:
                choice = st.selectbox("Select Task to Review", tasks["task"].unique())
                r = tasks[tasks["task"] == choice].iloc[0].to_dict()
                marks = st.slider("Marks (0–5)", 0.0, 5.0, float(r.get("marks", 0)))
                feedback = st.text_area("Manager Feedback")
                if st.button("Finalize Review"):
                    r["marks"] = marks
                    r["manager_feedback"] = feedback
                    r["status"] = "Under Client Review"
                    r["manager_reviewed_on"] = now()
                    upsert_data(r["_id"], r)
                    st.success("✅ Review submitted and sent to client.")

    # ------------------- Meetings -------------------
    with tabs[2]:
        st.subheader("Schedule Meeting (Linked to Client)")
        with st.form("schedule_meet"):
            title = st.text_input("Meeting Title")
            company = st.text_input("Client Company Name")
            date_ = st.date_input("Meeting Date", value=date.today())
            time_ = st.text_input("Time (HH:MM)", value="10:00")
            attendees = st.text_area("Attendees (comma-separated)")
            submit = st.form_submit_button("Create Meeting")
            if submit:
                attendees_clean = [a.strip().lower() for a in attendees.split(",") if a.strip()]
                mid = str(uuid.uuid4())
                md = {
                    "type": "Meeting",
                    "company": company,
                    "meeting_title": title,
                    "meeting_date": str(date_),
                    "meeting_time": time_,
                    "attendees": attendees_clean,
                    "created": now()
                }
                upsert_data(mid, md)
                st.success(f"✅ Meeting '{title}' scheduled for {company}.")

        df = fetch_all()
        meets = df[df["type"] == "Meeting"] if not df.empty else pd.DataFrame()
        if not meets.empty:
            st.dataframe(meets[["company", "meeting_title", "meeting_date", "attendees"]], use_container_width=True)

    # ------------------- AI Insights -------------------
    with tabs[3]:
        st.subheader("AI Insights")
        df = fetch_all()
        if df.empty:
            st.info("No data for AI insights.")
        else:
            q = st.text_input("Ask AI (e.g., 'Which employee is performing best?')")
            if st.button("Generate Insight"):
                prompt = f"Analyze workforce data: {df.describe().to_dict()}\nQuestion: {q}\nAnswer:"
                ans = hf_generate(prompt)
                st.write(ans)

# -------------------------------
# TEAM MEMBER DASHBOARD
# -------------------------------
elif role == "Team Member":
    st.header("Team Member Dashboard")
    name = st.text_input("Enter your name")
    company = st.text_input("Enter your company name")

    if name and company:
        df = fetch_all()
        my_tasks = df[(df["type"] == "Task") & (df["employee"].str.lower() == name.lower()) & (df["company"].str.lower() == company.lower())]
        st.subheader("My Tasks")
        if my_tasks.empty:
            st.info("No tasks assigned yet.")
        else:
            for _, r in my_tasks.iterrows():
                st.markdown(f"**Task:** {r['task']} | Status: {r['status']}")
                comp = st.slider("Completion %", 0, 100, int(r.get("completion", 0)), key=r["_id"])
                if st.button(f"Update {r['task']}", key=f"u_{r['_id']}"):
                    r["completion"] = comp
                    r["marks"] = float(lin_reg.predict([[comp]])[0])
                    r["status"] = "In Progress" if comp < 100 else "Completed"
                    upsert_data(r["_id"], r)
                    st.success("✅ Progress updated.")

        st.markdown("---")
        st.subheader("Meetings I'm Invited To")
        meets = df[df["type"] == "Meeting"]
        if meets.empty:
            st.info("No meetings scheduled.")
        else:
            def invited(row): return name.lower() in [a.strip().lower() for a in json.loads(row["attendees"])] if isinstance(row["attendees"], str) else name.lower() in row.get("attendees", [])
            invited_meets = meets[meets.apply(invited, axis=1)]
            if invited_meets.empty:
                st.info("No meetings for you.")
            else:
                st.dataframe(invited_meets[["company", "meeting_title", "meeting_date", "meeting_time"]])

# -------------------------------
# CLIENT DASHBOARD
# -------------------------------
elif role == "Client":
    st.header("Client Dashboard")
    company = st.text_input("Enter your Company Name")

    if company:
        df = fetch_all()
        df_client = df[df["company"].str.lower() == company.lower()] if not df.empty else pd.DataFrame()
        if df_client.empty:
            st.info("No records found for this company.")
        else:
            st.subheader("Tasks from Manager")
            tasks = df_client[df_client["type"] == "Task"]
            if not tasks.empty:
                st.dataframe(tasks[["employee", "task", "status", "completion", "marks"]], use_container_width=True)
                task_sel = st.selectbox("Select task for feedback", tasks["task"].unique())
                fb = st.text_area("Client Feedback")
                rating = st.slider("Client Rating (1–5)", 1, 5, 3)
                if st.button("Submit Feedback"):
                    r = tasks[tasks["task"] == task_sel].iloc[0].to_dict()
                    r["client_feedback"] = fb
                    r["client_rating"] = rating
                    r["client_reviewed"] = True
                    upsert_data(r["_id"], r)
                    st.success("✅ Feedback submitted.")

            st.subheader("Company Meetings")
            meets = df_client[df_client["type"] == "Meeting"]
            if not meets.empty:
                st.dataframe(meets[["meeting_title", "meeting_date", "meeting_time", "attendees"]], use_container_width=True)

# -------------------------------
# HR DASHBOARD
# -------------------------------
elif role == "HR (Admin)":
    st.header("HR Dashboard — Performance & Leave Clustering")
    df = fetch_all()
    if df.empty:
        st.info("No records available.")
    else:
        df_tasks = df[df["type"] == "Task"]
        if df_tasks.empty:
            st.info("No task data.")
        else:
            df_tasks["completion"] = pd.to_numeric(df_tasks["completion"], errors="coerce").fillna(0)
            df_tasks["marks"] = pd.to_numeric(df_tasks["marks"], errors="coerce").fillna(0)
            if len(df_tasks) >= 3:
                km = KMeans(n_clusters=3, random_state=42, n_init=10)
                df_tasks["cluster"] = km.fit_predict(df_tasks[["completion", "marks"]])
                centers = km.cluster_centers_
                order = np.argsort(centers[:, 0] + centers[:, 1])
                label_map = {order[2]: "High Performer", order[1]: "Average", order[0]: "Needs Improvement"}
                df_tasks["Performance Group"] = df_tasks["cluster"].map(label_map)
            st.dataframe(df_tasks[["company", "employee", "completion", "marks", "Performance Group"]], use_container_width=True)
