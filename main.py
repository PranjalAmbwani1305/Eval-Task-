# main.py — AI Workforce Intelligence Platform (Enterprise Edition)
# Dependencies: streamlit, pinecone-client, pandas, scikit-learn, plotly, huggingface-hub
# Author: Enterprise-ready version with Manager Feedback + Leave Fix + HR Analytics

import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import pandas as pd
import uuid, json, time
from datetime import datetime, date
from sklearn.linear_model import LinearRegression
from huggingface_hub import InferenceClient
import plotly.express as px

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="AI Workforce Intelligence Platform", layout="wide")
st.title("AI Workforce Intelligence Platform")

# ---------------------------------------------------------
# CONNECTIONS
# ---------------------------------------------------------
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
HF_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")

INDEX_NAME = "task"
DIMENSION = 1024

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
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            with st.spinner("Creating Pinecone index..."):
                while True:
                    desc = pc.describe_index(INDEX_NAME)
                    if desc["status"].get("ready"):
                        break
                    time.sleep(2)
        index = pc.Index(INDEX_NAME)
        st.caption(f"Connected to Pinecone index: {INDEX_NAME}")
    except Exception as e:
        st.warning(f"Pinecone init failed; running local only. ({e})")
else:
    st.warning("Pinecone API key missing — running local only.")

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def now(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def rand_vec(): return np.random.rand(DIMENSION).tolist()

def safe_meta(md):
    clean = {}
    for k, v in md.items():
        if isinstance(v, (datetime, date)):
            v = v.isoformat()
        elif isinstance(v, (list, dict)):
            v = json.dumps(v)
        elif pd.isna(v):
            v = ""
        clean[k] = v
    return clean

def upsert_data(id_, md):
    if not index:
        st.session_state.setdefault("LOCAL_DATA", {})[id_] = md
        return True
    md = safe_meta(md)
    try:
        index.upsert([{"id": str(id_), "values": rand_vec(), "metadata": md}])
        return True
    except Exception as e:
        st.warning(f"Pinecone upsert failed: {e}")
        return False

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

# ---------------------------------------------------------
# MODELS
# ---------------------------------------------------------
lin_reg = LinearRegression().fit([[0], [50], [100]], [0, 2.5, 5])

# ---------------------------------------------------------
# ROLE SELECTION
# ---------------------------------------------------------
role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client", "HR (Admin)"])
current_month = datetime.now().strftime("%B %Y")

# ---------------------------------------------------------
# MANAGER DASHBOARD
# ---------------------------------------------------------
if role == "Manager":
    st.header("Manager Dashboard")
    tabs = st.tabs(["Task Management", "Feedback & Review", "Leave Approvals", "Team Overview"])

    # --- Task Management ---
    with tabs[0]:
        st.subheader("Assign or Reassign Tasks")
        with st.form("assign_task"):
            company = st.text_input("Company")
            department = st.text_input("Department")
            employee = st.text_input("Employee")
            task = st.text_input("Task Title")
            desc = st.text_area("Description")
            deadline = st.date_input("Deadline", value=date.today())
            submit = st.form_submit_button("Assign Task")

            if submit and all([employee, task]):
                tid = str(uuid.uuid4())
                md = {
                    "type": "Task",
                    "company": company,
                    "department": department or "General",
                    "employee": employee,
                    "task": task,
                    "description": desc,
                    "deadline": deadline.isoformat(),
                    "completion": 0,
                    "marks": 0,
                    "status": "Assigned",
                    "created": now(),
                }
                upsert_data(tid, md)
                st.success(f"Task '{task}' assigned to {employee}.")

        df_all = fetch_all()
        if not df_all.empty:
            df = df_all[df_all.get("type") == "Task"]
            df["completion"] = pd.to_numeric(df.get("completion", 0), errors="coerce").fillna(0)
            st.dataframe(df[["employee", "task", "completion", "status", "deadline"]])

            # Reassign
            st.markdown("---")
            st.subheader("Reassign Task")
            if not df.empty:
                task_choice = st.selectbox("Select Task", df["task"].unique())
                new_emp = st.text_input("New Employee Name")
                reason = st.text_area("Reason for Reassignment")
                if st.button("Reassign Task"):
                    r = df[df["task"] == task_choice].iloc[0].to_dict()
                    r.update({
                        "employee": new_emp,
                        "status": "Reassigned",
                        "reassigned_reason": reason,
                        "reassigned_on": now()
                    })
                    upsert_data(r.get("_id") or str(uuid.uuid4()), r)
                    st.success("Task reassigned successfully.")

    # --- Feedback & Review ---
    with tabs[1]:
        st.subheader("Manager Feedback & Marks Review")
        df = fetch_all()
        if df.empty:
            st.info("No task data available.")
        else:
            # Only completed tasks
            completed = df[
                (df.get("type") == "Task") &
                (pd.to_numeric(df.get("completion", 0), errors="coerce") >= 100) &
                (df.get("status") != "Under Client Review")
            ]
            if completed.empty:
                st.info("No completed tasks awaiting review.")
            else:
                selected = st.selectbox("Select Task", completed["task"].dropna().unique())
                row = completed[completed["task"] == selected].iloc[0].to_dict()
                st.write(f"**Employee:** {row.get('employee')}  |  **Department:** {row.get('department')}")
                st.write(f"**Description:** {row.get('description', '')}")

                marks = st.slider("Final Marks (0–5)", 0.0, 5.0, float(row.get("marks", 0)))
                feedback = st.text_area("Manager Feedback")
                if st.button("Finalize Review"):
                    row["marks"] = marks
                    row["manager_feedback"] = feedback
                    row["manager_reviewed_on"] = now()
                    row["status"] = "Under Client Review"
                    upsert_data(row.get("_id") or str(uuid.uuid4()), row)
                    st.success("Feedback saved. Task sent to Client for review.")

    # --- Leave Approvals ---
    with tabs[2]:
        st.subheader("Leave Approvals")
        df = fetch_all()
        leaves = df[df.get("type") == "Leave"] if not df.empty else pd.DataFrame()
        if leaves.empty:
            st.info("No leave requests pending.")
        else:
            for i, row in leaves.iterrows():
                if row.get("status") == "Pending":
                    emp = row.get("employee")
                    typ = row.get("leave_type", "Leave")
                    st.markdown(f"**{emp}** requested **{typ}** leave ({row.get('from')} → {row.get('to')})")
                    st.write(f"Reason: {row.get('reason', '-')}")
                    decision = st.radio(f"Decision for {emp}", ["Approve", "Reject"], key=f"dec_{i}")
                    if st.button(f"Finalize Decision for {emp}", key=f"btn_{i}"):
                        r = dict(row)
                        r["_id"] = str(row.get("_id") or uuid.uuid4())
                        r["status"] = "Approved" if decision == "Approve" else "Rejected"
                        r["approved_on"] = now()
                        upsert_data(r["_id"], r)
                        st.success(f"Leave {r['status']} for {emp}")

    # --- Team Overview ---
    with tabs[3]:
        st.subheader("Team Performance Overview")
        df = fetch_all()
        if df.empty:
            st.info("No data available.")
        else:
            df = df[df.get("type") == "Task"]
            st.dataframe(df[["employee", "department", "task", "completion", "status", "created"]], use_container_width=True)
            if not df.empty:
                fig = px.bar(df, x="employee", y="completion", color="department", title="Completion by Employee")
                st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# TEAM MEMBER DASHBOARD
# ---------------------------------------------------------
elif role == "Team Member":
    st.header("Team Member Dashboard")

    name = st.text_input("Enter your name")
    if name:
        df_all = fetch_all()
        if df_all.empty:
            st.info("No records found.")
        else:
            if "type" not in df_all.columns:
                df_all["type"] = df_all.apply(lambda r: "Task" if pd.notna(r.get("task")) else "Leave", axis=1)

            tasks = df_all[(df_all["employee"].str.lower() == name.lower()) & (df_all["type"] == "Task")]
            leaves = df_all[(df_all["employee"].str.lower() == name.lower()) & (df_all["type"] == "Leave")]

            tabs_t = st.tabs(["My Tasks", "My Leaves"])
            with tabs_t[0]:
                if tasks.empty:
                    st.info("No tasks assigned yet.")
                else:
                    for _, r in tasks.iterrows():
                        st.markdown(f"**Task:** {r.get('task')} | **Status:** {r.get('status')}")
                        comp = st.slider("Completion %", 0, 100, int(r.get("completion", 0)), key=r.get("_id"))
                        if st.button(f"Update {r.get('task')}", key=f"u_{r.get('_id')}"):
                            r["completion"] = comp
                            r["marks"] = float(lin_reg.predict([[comp]])[0])
                            r["status"] = "In Progress" if comp < 100 else "Completed"
                            upsert_data(r.get("_id") or str(uuid.uuid4()), r)
                            st.success(f"Updated {r.get('task')}.")

            with tabs_t[1]:
                st.subheader("Leave Request")
                typ = st.selectbox("Leave Type", ["Casual", "Sick", "Earned"])
                f = st.date_input("From")
                t = st.date_input("To")
                reason = st.text_area("Reason")
                if st.button("Submit Leave Request"):
                    lid = str(uuid.uuid4())
                    md = {
                        "type": "Leave",
                        "employee": name,
                        "leave_type": typ,
                        "from": str(f),
                        "to": str(t),
                        "reason": reason,
                        "status": "Pending",
                        "submitted": now()
                    }
                    upsert_data(lid, md)
                    st.success("Leave submitted successfully.")

# ---------------------------------------------------------
# CLIENT DASHBOARD
# ---------------------------------------------------------
elif role == "Client":
    st.header("Client Review Center")

    company = st.text_input("Company Name")
    if company:
        df = fetch_all()
        df_client = df[(df["company"].str.lower() == company.lower()) & (df["type"] == "Task")] if not df.empty else pd.DataFrame()

        if df_client.empty:
            st.info("No data found.")
        else:
            unreviewed = df_client[
                (df_client.get("client_reviewed") != True) &
                (df_client.get("status") == "Under Client Review")
            ]
            reviewed = df_client[df_client.get("client_reviewed") == True]

            st.subheader("Pending Reviews")
            if not unreviewed.empty:
                task_sel = st.selectbox("Select Task to Review", unreviewed["task"].unique())
                fb = st.text_area("Feedback")
                rating = st.slider("Rating (1–5)", 1, 5, 3)
                if st.button("Submit Feedback"):
                    row = unreviewed[unreviewed["task"] == task_sel].iloc[0].to_dict()
                    row.update({
                        "client_feedback": fb,
                        "client_rating": rating,
                        "client_reviewed": True,
                        "client_approved_on": now()
                    })
                    upsert_data(row.get("_id") or str(uuid.uuid4()), row)
                    st.success("Feedback submitted successfully.")

            st.markdown("---")
            st.subheader("Reviewed Tasks")
            if not reviewed.empty:
                st.dataframe(reviewed[["task", "employee", "client_feedback", "client_rating", "client_approved_on"]])
            else:
                st.info("No reviewed tasks yet.")

# ---------------------------------------------------------
# HR / ADMIN DASHBOARD
# ---------------------------------------------------------
elif role == "HR (Admin)":
    st.header("HR Analytics & Intelligence")
    df = fetch_all()
    if df.empty:
        st.info("No data available.")
    else:
        df["completion"] = pd.to_numeric(df.get("completion", 0), errors="coerce").fillna(0)
        df["marks"] = pd.to_numeric(df.get("marks", 0), errors="coerce").fillna(0)
        df["risk"] = (100 - df["completion"]) / 100 + (2 - df["marks"]) / 5
        st.dataframe(df[["employee", "department", "task", "completion", "marks", "risk"]], use_container_width=True)
        fig = px.scatter(df, x="completion", y="marks", color="risk", title="Performance Risk Clustering")
        st.plotly_chart(fig, use_container_width=True)
