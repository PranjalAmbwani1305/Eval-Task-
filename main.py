import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from textblob import TextBlob
import uuid

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="AI Task Management System", layout="wide")
st.title("AI-Powered Task Management System")

# -----------------------------
# Helper Functions
# -----------------------------
def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def init_session():
    if "tasks" not in st.session_state:
        st.session_state["tasks"] = []

def save_task(task):
    df = pd.DataFrame(st.session_state["tasks"])
    # Replace if task exists
    if "_id" in task:
        df = df[df["_id"] != task["_id"]]
    df = pd.concat([df, pd.DataFrame([task])], ignore_index=True)
    st.session_state["tasks"] = df.to_dict("records")

def fetch_tasks(company=None, employee=None):
    """Safe fetch: never crashes if columns missing."""
    tasks = st.session_state.get("tasks", [])
    if not tasks:
        return pd.DataFrame(columns=["company", "employee", "task", "completion", "marks", "reviewed"])
    df = pd.DataFrame(tasks)
    expected_cols = ["company", "employee", "task", "completion", "marks", "reviewed"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None
    df = df.dropna(subset=["task"], how="all")
    if company:
        df = df[df["company"].fillna("") == company]
    if employee:
        df = df[df["employee"].fillna("") == employee]
    return df

# -----------------------------
# Machine Learning Setup
# -----------------------------
lin_reg = LinearRegression().fit([[0], [50], [100]], [1, 2, 3])
log_reg = LogisticRegression().fit([[0], [50], [100]], [0, 0, 1])

# -----------------------------
# NLP Summarization
# -----------------------------
def summarize_feedback(feedback_list):
    if not feedback_list:
        return "No feedback to summarize."
    text = " ".join(feedback_list)
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        sentences = blob.sentences if blob.sentences else [text]
        summary = " ".join(str(s) for s in sentences[:2])
        if polarity > 0.2:
            tone = "Positive"
        elif polarity < -0.2:
            tone = "Negative"
        else:
            tone = "Neutral"
        return f"Overall Sentiment: {tone}. Summary: {summary}"
    except Exception:
        return "Error analyzing feedback text."

# -----------------------------
# App Structure
# -----------------------------
init_session()
role = st.sidebar.selectbox("Select Role", ["Manager", "Team Member", "Client"])

# =============================
# MANAGER DASHBOARD
# =============================
if role == "Manager":
    st.header("Manager Dashboard")
    tab1, tab2, tab3, tab4 = st.tabs(["Assign Task", "Review Tasks", "360° Analytics", "Managerial Actions & Approvals"])

    # --- Assign Task ---
    with tab1:
        st.subheader("Assign New Task")
        company = st.text_input("Company Name")
        employee = st.text_input("Employee Name")
        task = st.text_input("Task Title")
        priority = st.selectbox("Priority", ["High", "Medium", "Low"])
        if st.button("Assign Task"):
            new_task = {
                "_id": str(uuid.uuid4()),
                "company": company,
                "employee": employee,
                "task": task,
                "priority": priority,
                "completion": 0,
                "marks": 0,
                "reviewed": False,
                "assigned_on": now()
            }
            save_task(new_task)
            st.success(f"Task '{task}' assigned to {employee}.")

    # --- Review Tasks ---
    with tab2:
        df = fetch_tasks()
        if df.empty:
            st.info("No tasks to review yet.")
        else:
            for _, r in df.iterrows():
                tname = r.get("task", "Unnamed Task")
                employee = r.get("employee", "Unknown")
                st.markdown(f"**{tname}** — Assigned to: {employee}")
                completion = st.slider(f"Completion for {tname}", 0, 100, int(r.get("completion", 0)))
                comments = st.text_area(f"Manager Feedback for {tname}", "")
                approve = st.radio(f"Approve {tname}?", ["Yes", "No"], key=f"a_{r.get('_id')}")
                if st.button(f"Finalize Review for {tname}", key=f"btn_{r.get('_id')}"):
                    marks = float(lin_reg.predict([[completion]])[0])
                    blob = TextBlob(comments)
                    sentiment = "Positive" if blob.sentiment.polarity > 0 else "Negative"
                    reviewed_task = {
                        **r.to_dict(),
                        "completion": completion,
                        "marks": marks,
                        "reviewed": True,
                        "comments": comments,
                        "sentiment": sentiment,
                        "status": "Approved" if approve == "Yes" else "Pending",
                        "reviewed_on": now()
                    }
                    save_task(reviewed_task)
                    st.success(f"Review saved for task '{tname}' ({sentiment})")

    # --- 360° Analytics ---
    with tab3:
        df = fetch_tasks()
        if df.empty:
            st.info("No data for analytics.")
        else:
            st.dataframe(df[["company", "employee", "task", "completion", "marks", "sentiment"]])
            avg_completion = df["completion"].astype(float).mean()
            st.metric("Average Task Completion", f"{avg_completion:.2f}%")
            st.bar_chart(df[["completion"]])

    # --- Managerial Actions ---
    with tab4:
        df = fetch_tasks()
        if df.empty:
            st.info("No tasks found for actions.")
        else:
            st.dataframe(df[["company", "employee", "task", "status"]])
            if st.button("Generate Performance Summary"):
                perf_summary = df.groupby("employee")["completion"].mean().reset_index()
                st.dataframe(perf_summary)

# =============================
# TEAM MEMBER PORTAL
# =============================
elif role == "Team Member":
    st.header("Team Member Portal")
    tab1, tab2 = st.tabs(["My Tasks", "AI Feedback Summarization"])

    with tab1:
        company = st.text_input("Company Name")
        employee = st.text_input("Your Name")
        df = fetch_tasks(company, employee)
        if df.empty:
            st.warning("No tasks assigned yet.")
        else:
            for _, r in df.iterrows():
                st.markdown(f"**Task:** {r['task']} — Progress: {r['completion']}% — Status: {r.get('status', 'Pending')}")
                progress = st.slider(f"Update progress for {r['task']}", 0, 100, int(r['completion']))
                if st.button(f"Update {r['task']}", key=f"update_{r['_id']}"):
                    updated = {**r.to_dict(), "completion": progress}
                    save_task(updated)
                    st.success(f"Progress updated for {r['task']}")

    with tab2:
        st.subheader("AI Feedback Summarization")
        company = st.text_input("Company Name (for summary)")
        employee = st.text_input("Your Name (for summary)")
        df = fetch_tasks(company, employee)
        if st.button("Generate Summary"):
            if df.empty:
                st.warning("No feedback found to summarize.")
            else:
                feedbacks = df["comments"].dropna().tolist()
                summary = summarize_feedback(feedbacks)
                st.info(summary)

# =============================
# CLIENT REVIEW PANEL
# =============================
else:
    st.header("Client Review")
    company = st.text_input("Company Name")
    df = fetch_tasks(company)
    if df.empty:
        st.warning("No project data found for this company.")
    else:
        st.dataframe(df[["employee", "task", "completion", "status", "sentiment"]])
        avg_score = df["completion"].astype(float).mean()
        st.metric("Overall Completion", f"{avg_score:.2f}%")
