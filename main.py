import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import pandas as pd
import uuid
from datetime import datetime, date
import io

# --- Setup ---
st.set_page_config(page_title="AI Task Management System", layout="wide")
st.title("AI Task Management System")

# --- Pinecone setup ---
API_KEY = st.secrets["PINECONE_API_KEY"]
INDEX_NAME = "task"
DIM = 128

pc = Pinecone(api_key=API_KEY)
if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(INDEX_NAME, dimension=DIM, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
index = pc.Index(INDEX_NAME)

# --- Helpers ---
def rand_vec():
    return np.random.rand(DIM).tolist()

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def safe_meta(md):
    clean = {}
    for k, v in md.items():
        if isinstance(v, (datetime, date)):
            clean[k] = v.isoformat()
        elif v is None:
            clean[k] = ""
        else:
            clean[k] = str(v)
    return clean

def get_all():
    """Fetch all tasks safely from Pinecone"""
    try:
        res = index.query(vector=rand_vec(), top_k=1000, include_metadata=True)
        rows = []
        for m in res.matches:
            if m.metadata:
                row = m.metadata
                row["_id"] = m.id
                rows.append(row)
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception as e:
        st.error("Error fetching from Pinecone")
        st.exception(e)
        return pd.DataFrame()

# --- Role Selection ---
role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client"])
current_month = datetime.now().strftime("%B %Y")

# =========================================================
# MANAGER SECTION
# =========================================================
if role == "Manager":
    st.header("Manager — Assign and Review Tasks")

    # --- Assign Task ---
    with st.form("assign"):
        company = st.text_input("Company")
        employee = st.text_input("Employee")
        title = st.text_input("Task Title")
        desc = st.text_area("Description")
        deadline = st.date_input("Deadline")
        submit = st.form_submit_button("Assign Task")
        if submit:
            if not (company and employee and title):
                st.error("Please fill company, employee, and title.")
            else:
                tid = str(uuid.uuid4())
                meta = safe_meta({
                    "company": company,
                    "employee": employee,
                    "task": title,
                    "description": desc,
                    "deadline": deadline,
                    "month": current_month,
                    "completion": "0",
                    "status": "Assigned",
                    "client_reviewed": "False",
                    "reviewed": "False",
                    "assigned_on": now()
                })
                index.upsert([{"id": tid, "values": rand_vec(), "metadata": meta}])
                st.success(f"Assigned {title} to {employee}")

    st.divider()

    # --- Review Tasks ---
    st.subheader("Review Client-Approved Tasks")
    df = get_all()
    if not df.empty:
        company_filter = st.text_input("Enter Company to Review")
        if company_filter:
            df = df[df["company"].str.lower() == company_filter.lower()]
        df_approved = df[df.get("client_reviewed", "").astype(str).str.lower() == "true"]

        if df_approved.empty:
            st.info("No client-approved tasks found.")
        else:
            for _, row in df_approved.iterrows():
                st.markdown(f"### {row['task']}")
                st.write(f"Employee: {row['employee']}")
                st.write(f"Client Comments: {row.get('client_comments', '')}")
                marks = st.number_input(f"Marks for {row['task']}", 0.0, 5.0, step=0.1)
                comments = st.text_area(f"Manager Comments for {row['task']}")
                if st.button(f"Finalize {row['task']}", key=row["_id"]):
                    updated = {**row, "marks": marks, "manager_comments": comments, "reviewed": "True"}
                    index.upsert([{"id": row["_id"], "values": rand_vec(), "metadata": safe_meta(updated)}])
                    st.success(f"Finalized review for {row['task']}")
    else:
        st.info("No data found in Pinecone yet.")

# =========================================================
# TEAM MEMBER SECTION
# =========================================================
elif role == "Team Member":
    st.header("Team Member — My Tasks")

    company = st.text_input("Company")
    employee = st.text_input("Your Name")

    if st.button("Load Tasks"):
        df = get_all()
        if df.empty:
            st.warning("No tasks found.")
        else:
            df_emp = df[
                (df["company"].str.lower() == company.lower()) &
                (df["employee"].str.lower() == employee.lower())
            ]
            if df_emp.empty:
                st.info("No tasks found for you.")
            else:
                for _, row in df_emp.iterrows():
                    st.markdown(f"### {row['task']}")
                    st.write(row.get("description", ""))
                    st.write(f"Deadline: {row.get('deadline', '')}")
                    comp = st.slider(f"Completion (%) for {row['task']}", 0, 100, int(float(row.get('completion', 0))))
                    if st.button(f"Update {row['task']}", key=row["_id"]):
                        updated = {**row, "completion": str(comp), "submitted_on": now()}
                        index.upsert([{"id": row["_id"], "values": rand_vec(), "metadata": safe_meta(updated)}])
                        st.success(f"Updated progress for {row['task']}")

# =========================================================
# CLIENT SECTION
# =========================================================
elif role == "Client":
    st.header("Client — Review and Approve Tasks")

    company = st.text_input("Company Name")
    if st.button("Load Tasks"):
        df = get_all()
        if df.empty:
            st.warning("No tasks in Pinecone.")
        else:
            df_company = df[df["company"].str.lower() == company.lower()]
            df_done = df_company[df_company["completion"].astype(float) >= 100]
            if df_done.empty:
                st.info("No completed tasks found.")
            else:
                for _, row in df_done.iterrows():
                    st.markdown(f"### {row['task']}")
                    st.write(f"Employee: {row['employee']}")
                    st.write(f"Completion: {row['completion']}%")
                    feedback = st.text_area(f"Feedback for {row['task']}")
                    if st.button(f"Approve {row['task']}", key=row["_id"]):
                        updated = {**row, "client_reviewed": "True", "client_comments": feedback, "client_approved_on": now()}
                        index.upsert([{"id": row["_id"], "values": rand_vec(), "metadata": safe_meta(updated)}])
                        st.success(f"Approved {row['task']} for manager review.")
