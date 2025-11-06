import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
import numpy as np
import uuid
from datetime import datetime, date
from textblob import TextBlob
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
import plotly.express as px

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="AI Task Management System", layout="wide")
st.title("AI Task Management System")

# -----------------------------
# INIT
# -----------------------------
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "taskt"
DIMENSION = 1024

if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# -----------------------------
# LOCAL EMBEDDINGS (NO OPENAI)
# -----------------------------
def simple_embed(text):
    """A lightweight hash-based embedding."""
    np.random.seed(abs(hash(text)) % (2**32))
    return np.random.rand(DIMENSION).tolist()

# -----------------------------
# DB HELPERS
# -----------------------------
def store_task(task):
    """Save task in Pinecone"""
    try:
        vec = simple_embed(f"{task['task']} {task.get('description','')} {task.get('comments','')}")
        index.upsert([{"id": task["_id"], "values": vec, "metadata": task}])
    except Exception as e:
        st.error(f"Failed to store in Pinecone: {e}")

def fetch_tasks(company=None, employee=None):
    """Fetch all tasks"""
    try:
        res = index.query(vector=np.random.rand(DIMENSION).tolist(), top_k=200, include_metadata=True)
        rows = [m.metadata for m in res.matches if "metadata" in m]
        df = pd.DataFrame(rows)
        if not df.empty:
            if company and "company" in df.columns:
                df = df[df["company"].str.lower() == company.lower()]
            if employee and "employee" in df.columns:
                df = df[df["employee"].str.lower() == employee.lower()]
        return df
    except Exception as e:
        st.warning(f"Unable to fetch tasks: {e}")
        return pd.DataFrame()

# -----------------------------
# NLP FEEDBACK SUMMARIZER
# -----------------------------
def summarize_feedback(feedback_list):
    """Summarize text feedback using NLP sentiment"""
    if not feedback_list:
        return "No feedback available."
    full_text = " ".join(feedback_list)
    blob = TextBlob(full_text)
    sentiment = blob.sentiment.polarity
    summary = " ".join([str(s) for s in blob.sentences[:2]]) if blob.sentences else full_text[:150]
    if sentiment > 0.2:
        mood = "Overall Positive Feedback."
    elif sentiment < -0.2:
        mood = "Overall Negative Feedback."
    else:
        mood = "Overall Neutral Feedback."
    return f"{mood} Summary: {summary}"

# -----------------------------
# BASIC MODELS
# -----------------------------
lin_reg = LinearRegression()
lin_reg.fit([[0], [50], [100]], [0, 2.5, 5])

log_reg = LogisticRegression()
log_reg.fit([[0], [50], [100]], [0, 1, 1])

# -----------------------------
# ROLE SELECTION
# -----------------------------
role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client"])
current_month = datetime.now().strftime("%B %Y")

# -----------------------------
# MANAGER VIEW
# -----------------------------
if role == "Manager":
    st.header("Manager Dashboard")
    tab1, tab2, tab3, tab4 = st.tabs(["Assign Task", "Review Tasks", "360° Analytics", "Managerial Actions & Approvals"])

    # --- Assign Task ---
    with tab1:
        st.subheader("Assign New Task")
        with st.form("assign_task"):
            company = st.text_input("Company Name")
            employee = st.text_input("Employee Name")
            task_title = st.text_input("Task Title")
            description = st.text_area("Task Description")
            deadline = st.date_input("Deadline", value=date.today())
            priority = st.selectbox("Priority", ["Low", "Medium", "High"])
            submit = st.form_submit_button("Assign Task")

            if submit and company and employee and task_title:
                task_data = {
                    "_id": str(uuid.uuid4()),
                    "company": company,
                    "employee": employee,
                    "task": task_title,
                    "description": description,
                    "deadline": deadline.isoformat(),
                    "priority": priority,
                    "completion": 0,
                    "marks": 0,
                    "status": "Assigned",
                    "month": current_month,
                    "assigned_on": now()
                }
                store_task(task_data)
                st.success(f"Task '{task_title}' assigned to {employee}.")

    # --- Review Tasks ---
    with tab2:
        st.subheader("Review Tasks")
        company = st.text_input("Filter by Company")
        df = fetch_tasks(company)
        if df.empty:
            st.info("No tasks found.")
        else:
            for _, r in df.iterrows():
                st.markdown(f"### {r.get('task', 'Unnamed Task')}")
                adj = st.slider(f"Completion ({r.get('employee', '')})", 0, 100, int(r.get("completion", 0)))
                comments = st.text_area(f"Manager Comments ({r.get('task')})", key=f"c_{r['_id']}")
                if st.button(f"Finalize Review {r.get('task')}", key=f"f_{r['_id']}"):
                    marks = float(lin_reg.predict([[adj]])[0])
                    sentiment = TextBlob(comments).sentiment.polarity
                    sentiment_label = "Positive" if sentiment > 0 else "Negative"
                    r["completion"] = adj
                    r["marks"] = marks
                    r["comments"] = comments
                    r["sentiment"] = sentiment_label
                    r["reviewed_on"] = now()
                    store_task(r)
                    st.success(f"Review saved: {sentiment_label}")

    # --- 360° Analytics ---
    with tab3:
        st.subheader("360° Analytics Overview")
        df = fetch_tasks()
        if not df.empty and "completion" in df.columns and "marks" in df.columns:
            df["completion"] = pd.to_numeric(df["completion"], errors="coerce")
            df["marks"] = pd.to_numeric(df["marks"], errors="coerce")
            kmeans = KMeans(n_clusters=3, n_init=10).fit(df[["completion", "marks"]].fillna(0))
            df["cluster"] = kmeans.labels_
            st.plotly_chart(px.scatter(df, x="completion", y="marks", color=df["cluster"].astype(str),
                                       hover_data=["employee", "task"]))
        else:
            st.info("Not enough data for clustering.")

    # --- Managerial Actions ---
    with tab4:
        st.subheader("Managerial Actions & Approvals")
        df = fetch_tasks()
        if df.empty:
            st.info("No tasks found for actions.")
        else:
            for _, r in df.iterrows():
                st.markdown(f"**Task:** {r.get('task', '')} | **Employee:** {r.get('employee', '')}")
                action = st.selectbox(f"Action for {r.get('task')}",
                                      ["None", "Reassign", "Approve", "Escalate"],
                                      key=f"a_{r['_id']}")
                if action == "Reassign":
                    new_emp = st.text_input("New Employee Name", key=f"new_{r['_id']}")
                    if st.button(f"Confirm Reassign {r.get('task')}", key=f"r_{r['_id']}"):
                        r["employee"] = new_emp
                        r["status"] = "Reassigned"
                        store_task(r)
                        st.success(f"Task reassigned to {new_emp}.")
                elif action == "Approve":
                    if st.button(f"Approve {r.get('task')}", key=f"ap_{r['_id']}"):
                        r["status"] = "Approved"
                        store_task(r)
                        st.success(f"Task '{r.get('task')}' approved.")

# -----------------------------
# TEAM MEMBER VIEW
# -----------------------------
elif role == "Team Member":
    st.header("Team Member Portal")
    tab1, tab2 = st.tabs(["My Tasks", "AI Feedback Summarization"])

    # --- My Tasks ---
    with tab1:
        company = st.text_input("Company")
        employee = st.text_input("Your Name")
        if st.button("Load Tasks"):
            df = fetch_tasks(company, employee)
            st.session_state["tasks"] = df.to_dict(orient="records")
        for task in st.session_state.get("tasks", []):
            st.markdown(f"### {task['task']}")
            new_val = st.slider(f"Completion ({task['task']})", 0, 100, int(task.get("completion", 0)))
            if st.button(f"Submit {task['task']}", key=task["_id"]):
                marks = float(lin_reg.predict([[new_val]])[0])
                task["completion"] = new_val
                task["marks"] = marks
                task["status"] = "In Progress" if new_val < 100 else "Completed"
                store_task(task)
                st.success(f"Updated {task['task']} ({task['status']})")

    # --- AI Feedback Summarization ---
    with tab2:
        st.subheader("AI Feedback Summarization")
        company = st.text_input("Company Name (for summary)")
        employee = st.text_input("Your Name (for summary)")
        if st.button("Generate Feedback Summary"):
            df = fetch_tasks(company, employee)
            if not df.empty and "comments" in df.columns:
                feedbacks = df["comments"].dropna().tolist()
                summary = summarize_feedback(feedbacks)
                st.info(summary)
            else:
                st.warning("No feedback found for summarization.")

# -----------------------------
# CLIENT VIEW
# -----------------------------
elif role == "Client":
    st.header("Client Review")
    company = st.text_input("Company Name")
    if st.button("Load Completed Tasks"):
        df = fetch_tasks(company)
        if df.empty:
            st.warning("No completed tasks found.")
        else:
            df = df[df["status"].str.lower().isin(["completed", "approved", "client approved"])]
            for _, r in df.iterrows():
                st.markdown(f"### {r['task']}")
                st.write(f"Employee: {r['employee']}")
                st.write(f"Completion: {r['completion']}%")
                st.write(f"Marks: {r['marks']}")
                feedback = st.text_area(f"Client Feedback ({r['task']})", key=f"cf_{r['_id']}")
                if st.button(f"Approve {r['task']}", key=f"app_{r['_id']}"):
                    r["client_feedback"] = feedback
                    r["client_approved_on"] = now()
                    r["status"] = "Client Approved"
                    store_task(r)
                    st.success(f"Client approved '{r['task']}'.")
