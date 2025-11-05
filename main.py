import streamlit as st
import pandas as pd
import numpy as np
import pinecone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
import plotly.express as px
import random, string
from datetime import datetime
import joblib

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index_name = "task-index"

if index_name not in [i.name for i in pc.list_indexes()]:
    pc.create_index(name=index_name, dimension=1024, metric="cosine")

index = pc.Index(index_name)


# Generate random vector for queries
def rand_vec(dim=128):
    return np.random.rand(dim).tolist()


st.title("AI-Powered Task Management System")

page = st.sidebar.selectbox(
    "Navigation",
    ["Manager Dashboard", "Team Member Dashboard", "Client Review"],
)


# ========== MANAGER DASHBOARD ==========
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
            task_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
            metadata = {
                "company": company,
                "employee": employee,
                "task": task,
                "description": description,
                "deadline": str(deadline),
                "month": month,
                "completion": 0,
                "marks": 0,
                "reviewed": False,
                "client_reviewed": False,
                "status": "Assigned",
                "assigned_on": str(datetime.now()),
                "deadline_risk": "Low",
            }
            index.upsert([(task_id, rand_vec(), metadata)])
            st.success(f"✅ Task '{task}' assigned successfully to {employee}!")
        else:
            st.warning("Please fill in all required fields.")

    st.divider()
    st.subheader("Boss Review & Adjustment")

    # Fetch all unreviewed tasks
    res = index.query(vector=rand_vec(), top_k=200, include_metadata=True)
    unreviewed_tasks = [m for m in res.matches if not m.metadata.get("reviewed", False)]

    if unreviewed_tasks:
        for m in unreviewed_tasks:
            meta = m.metadata
            st.write(f"### {meta['task']}")
            st.write(f"Employee: {meta['employee']}")
            st.write(f"Reported Completion: {meta.get('completion', 0)}%")
            st.write(f"Current Marks: {meta.get('marks', 0)}")
            st.write(f"Deadline Risk: {meta.get('deadline_risk', 'Low')}")

            new_completion = st.slider(
                f"Adjust Completion % for {meta['task']}",
                0,
                100,
                int(meta.get("completion", 0)),
                5,
            )

            new_marks = round(5 * new_completion / 100, 2)
            comment = st.text_area(f"Boss Comments for {meta['task']}")
            approve = st.radio(f"Approve Task {meta['task']}?", ["Yes", "No"], key=m.id)

            if st.button(f"Finalize Review for {meta['task']}"):
                meta.update(
                    {
                        "completion": new_completion,
                        "marks": new_marks,
                        "reviewed": True,
                        "boss_comments": comment,
                        "approved": approve == "Yes",
                    }
                )
                index.upsert([(m.id, rand_vec(), meta)])
                st.success(f"Review finalized for {meta['task']} ✅")
    else:
        st.info("No unreviewed tasks available.")


# ========== TEAM MEMBER DASHBOARD ==========
elif page == "Team Member Dashboard":
    st.header("Team Member Dashboard")

    company = st.text_input("Company")
    employee = st.text_input("Your Name")

    if st.button("Load Tasks"):
        try:
            res = index.query(
                vector=rand_vec(),
                top_k=500,
                include_metadata=True,
                filter={"company": company, "employee": employee},
            )
            st.session_state["tasks"] = [(m.id, m.metadata) for m in (res.matches or [])]
            st.success(f"Loaded {len(st.session_state['tasks'])} tasks.")
        except Exception as e:
            st.error(f"Error fetching tasks: {e}")

    if "tasks" in st.session_state and st.session_state["tasks"]:
        for task_id, meta in st.session_state["tasks"]:
            st.write(f"### {meta['task']}")
            st.write(f"Description: {meta['description']}")
            completion = st.slider(
                f"Update completion for {meta['task']}",
                0,
                100,
                int(meta.get("completion", 0)),
                5,
            )
            marks = round(5 * completion / 100, 2)
            if st.button(f"Submit Progress for {meta['task']}"):
                meta.update({"completion": completion, "marks": marks})
                index.upsert([(task_id, rand_vec(), meta)])
                st.success(f"Progress updated for {meta['task']}")
    else:
        st.info("No tasks found for this employee/company.")


# ========== CLIENT REVIEW ==========
elif page == "Client Review":
    st.header("Client Review")

    company = st.text_input("Enter Company Name")

    if st.button("Load Client Tasks"):
        try:
            res = index.query(
                vector=rand_vec(),
                top_k=200,
                include_metadata=True,
                filter={"company": company},
            )
            reviewed = [
                (m.id, m.metadata)
                for m in res.matches
                if m.metadata.get("reviewed", False)
            ]
            st.session_state["client_tasks"] = reviewed
            st.success(f"Loaded {len(reviewed)} reviewed tasks.")
        except Exception as e:
            st.error(f"Error fetching client tasks: {e}")

    if "client_tasks" in st.session_state and st.session_state["client_tasks"]:
        for task_id, meta in st.session_state["client_tasks"]:
            st.write(f"### {meta['task']}")
            st.write(f"Employee: {meta['employee']}")
            st.write(f"Completion: {meta['completion']}%")
            st.write(f"Marks: {meta['marks']}")
            st.write(f"Deadline: {meta['deadline']}")
            st.write(f"Boss Comments: {meta.get('boss_comments', 'N/A')}")
            client_approve = st.radio(
                f"Client Approve {meta['task']}?",
                ["Yes", "No"],
                key=task_id,
            )
            if st.button(f"Finalize Client Review for {meta['task']}"):
                meta["client_reviewed"] = client_approve == "Yes"
                meta["status"] = (
                    "Completed" if client_approve == "Yes" else "Revision Needed"
                )
                index.upsert([(task_id, rand_vec(), meta)])
                st.success(f"Client review saved for {meta['task']}")
    else:
        st.info("No reviewed tasks available for this company yet.")
