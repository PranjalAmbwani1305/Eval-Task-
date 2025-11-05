import streamlit as st
import pandas as pd
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
import plotly.express as px
import random, string
from datetime import datetime

# -----------------------------
# CONFIG & INIT
# -----------------------------
st.set_page_config(page_title="AI Task System", layout="wide")
st.title("AI-Powered Task Management System")

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "task"
DIMENSION = 1024

# Create Pinecone index if missing
if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(INDEX_NAME)

# -----------------------------
# HELPERS
# -----------------------------
def rand_vec():
    return np.random.rand(DIMENSION).tolist()

def safe_meta(md):
    clean = {}
    for k, v in md.items():
        if v is None:
            v = ""
        elif isinstance(v, datetime):
            v = v.isoformat()
        clean[k] = v
    return clean

# -----------------------------
# UI
# -----------------------------
page = st.sidebar.selectbox(
    "Navigation",
    ["Manager Dashboard", "Team Member Dashboard", "Client Review"]
)

# -----------------------------
# MANAGER DASHBOARD
# -----------------------------
if page == "Manager Dashboard":
    st.header("Manager Dashboard")

    st.subheader("Assign Task")
    company = st.text_input("Company Name")
    employee = st.text_input("Employee Name")
    task = st.text_input("Task Title")
    description = st.text_area("Task Description")
    deadline = st.date_input("Deadline")
    month = st.text_input("Month (e.g., Nov 2025)")

    if st.button("Assign Task"):
        if company and employee and task:
            tid = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
            md = safe_meta({
                "company": company,
                "employee": employee,
                "task": task,
                "description": description,
                "deadline": deadline,
                "month": month,
                "completion": 0,
                "marks": 0,
                "reviewed": False,
                "client_reviewed": False,
                "status": "Assigned",
                "assigned_on": datetime.now()
            })
            index.upsert([{"id": tid, "values": rand_vec(), "metadata": md}])
            st.success(f"Task '{task}' assigned to {employee}.")
        else:
            st.warning("Please fill all required fields.")

    st.divider()
    st.subheader("Boss Review & Adjustment")

    res = index.query(vector=rand_vec(), top_k=200, include_metadata=True)
    tasks = [m for m in res.matches if not m.metadata.get("reviewed", False)]

    if not tasks:
        st.info("No tasks pending review.")
    else:
        for m in tasks:
            md = m.metadata
            st.write(f"### {md['task']}")
            st.write(f"Employee: {md['employee']}")
            st.write(f"Reported Completion: {md.get('completion', 0)}%")
            st.write(f"Current Marks: {md.get('marks', 0)}")

            new_completion = st.slider(
                f"Adjust Completion % for {md['task']}",
                0, 100, int(md.get("completion", 0)), 5
            )
            new_marks = round(5 * new_completion / 100, 2)
            comment = st.text_area(f"Boss Comments for {md['task']}")
            approve = st.radio(f"Approve Task {md['task']}?", ["Yes", "No"], key=m.id)

            if st.button(f"Finalize Review for {md['task']}"):
                md.update({
                    "completion": new_completion,
                    "marks": new_marks,
                    "reviewed": True,
                    "boss_comments": comment,
                    "approved": approve == "Yes",
                })
                index.upsert([{"id": m.id, "values": rand_vec(), "metadata": safe_meta(md)}])
                st.success(f"Review finalized for {md['task']} ✅")

# -----------------------------
# TEAM MEMBER DASHBOARD
# -----------------------------
elif page == "Team Member Dashboard":
    st.header("Team Member Dashboard")

    company = st.text_input("Company")
    employee = st.text_input("Your Name")

    if st.button("Load Tasks"):
        try:
            res = index.query(vector=rand_vec(), top_k=200, include_metadata=True)
            matches = [m for m in res.matches if
                       m.metadata.get("company") == company and
                       m.metadata.get("employee") == employee]
            st.session_state["tasks"] = matches
            st.success(f"Loaded {len(matches)} tasks.")
        except Exception as e:
            st.error(f"Error fetching tasks: {e}")

    for m in st.session_state.get("tasks", []):
        md = m.metadata
        st.write(f"### {md['task']}")
        st.write(f"Description: {md['description']}")
        completion = st.slider(
            f"Update completion for {md['task']}",
            0, 100, int(md.get("completion", 0)), 5
        )
        marks = round(5 * completion / 100, 2)
        if st.button(f"Submit Progress for {md['task']}", key=m.id):
            md.update({"completion": completion, "marks": marks})
            index.upsert([{"id": m.id, "values": rand_vec(), "metadata": safe_meta(md)}])
            st.success(f"Progress updated for {md['task']}")

# -----------------------------
# CLIENT REVIEW
# -----------------------------
elif page == "Client Review":
    st.header("Client Review")

    company = st.text_input("Enter Company Name")
    if st.button("Load Reviewed Tasks"):
        try:
            res = index.query(vector=rand_vec(), top_k=200, include_metadata=True)
            reviewed = [m for m in res.matches if
                        m.metadata.get("company") == company and
                        m.metadata.get("reviewed", False)]
            st.session_state["client_tasks"] = reviewed
            st.success(f"Loaded {len(reviewed)} reviewed tasks.")
        except Exception as e:
            st.error(f"Error fetching client tasks: {e}")

    for m in st.session_state.get("client_tasks", []):
        md = m.metadata
        st.write(f"### {md['task']}")
        st.write(f"Employee: {md['employee']}")
        st.write(f"Completion: {md['completion']}%")
        st.write(f"Marks: {md['marks']}")
        st.write(f"Boss Comments: {md.get('boss_comments', 'N/A')}")

        approve = st.radio(f"Client Approve {md['task']}?", ["Yes", "No"], key=m.id)
        if st.button(f"Finalize Client Review for {md['task']}"):
            md.update({
                "client_reviewed": approve == "Yes",
                "status": "Completed" if approve == "Yes" else "Revision Needed"
            })
            index.upsert([{"id": m.id, "values": rand_vec(), "metadata": safe_meta(md)}])
            st.success(f"Client review saved for {md['task']}")
