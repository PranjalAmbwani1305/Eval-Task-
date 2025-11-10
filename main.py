# main.py — AI Workforce Intelligence Platform (Final Stable Enterprise Build)

import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import pandas as pd
import uuid, json, time
from datetime import datetime, date
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import plotly.express as px
from huggingface_hub import InferenceClient

# ---------------------------------------------------------
# APP CONFIG
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

pc, index = None, None
if PINECONE_API_KEY:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
            pc.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not pc.describe_index(INDEX_NAME)["status"].get("ready"):
                time.sleep(2)
        index = pc.Index(INDEX_NAME)
        st.caption(f"Connected to Pinecone index: {INDEX_NAME}")
    except Exception as e:
        st.warning(f"Pinecone init failed; local mode ({e})")
else:
    st.warning("Pinecone API key not found — running in local mode")

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
        return
    md = safe_meta(md)
    try:
        index.upsert([{"id": id_, "values": rand_vec(), "metadata": md}])
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

lin_reg = LinearRegression().fit([[0], [50], [100]], [0, 2.5, 5])

# ---------------------------------------------------------
# ROLE SELECTION
# ---------------------------------------------------------
role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client", "HR"])

# ---------------------------------------------------------
# MANAGER DASHBOARD
# ---------------------------------------------------------
if role == "Manager":
    st.header("Manager Command Center")

    df = fetch_all()
    tab1, tab2 = st.tabs(["Assign / Reassign Tasks", "Team Overview"])

    with tab1:
        st.subheader("Assign or Reassign Task")
        with st.form("assign_task"):
            company = st.text_input("Company")
            department = st.text_input("Department")
            employee = st.text_input("Employee Name")
            task = st.text_input("Task Title")
            desc = st.text_area("Task Description")
            deadline = st.date_input("Deadline", value=date.today())
            submit = st.form_submit_button("Assign Task")

            if submit and all([employee, task]):
                tid = str(uuid.uuid4())
                md = {
                    "company": company, "department": department,
                    "employee": employee, "task": task, "description": desc,
                    "deadline": deadline.isoformat(), "completion": 0,
                    "marks": 0, "status": "Assigned", "type": "Task", "created": now()
                }
                upsert_data(tid, md)
                st.success(f"Task '{task}' assigned to {employee}")

        if not df.empty:
            task_sel = st.selectbox("Select Task to Reassign", df["task"].dropna().unique())
            new_emp = st.text_input("Reassign to Employee")
            reason = st.text_area("Reason for Reassignment")
            if st.button("Reassign"):
                record = df[df["task"] == task_sel].iloc[0].to_dict()
                record.update({"employee": new_emp, "status": "Reassigned", "reason": reason, "reassigned": now()})
                upsert_data(record.get("_id") or str(uuid.uuid4()), record)
                st.success(f"Task '{task_sel}' reassigned to {new_emp}")

    with tab2:
        st.subheader("Team Task Overview")
        if df.empty:
            st.info("No team data yet.")
        else:
            df["completion"] = pd.to_numeric(df["completion"], errors="coerce").fillna(0)
            st.dataframe(df[["employee", "task", "completion", "status", "department"]])
            fig = px.bar(df, x="employee", y="completion", color="status", title="Team Task Completion Overview")
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# TEAM MEMBER DASHBOARD
# ---------------------------------------------------------
elif role == "Team Member":
    st.header("Team Member Dashboard")
    name = st.text_input("Enter your name")

    if name:
        df = fetch_all()
        if df.empty:
            st.info("No data available.")
        else:
            # Ensure missing fields handled
            df["type"] = df.get("type", "Task")
            my_tasks = df[(df["employee"].str.lower() == name.lower()) & (df["type"] == "Task")]
            my_leaves = df[(df["employee"].str.lower() == name.lower()) & (df["type"].isin(["Casual", "Sick", "Earned"]))]

            tab1, tab2 = st.tabs(["My Tasks", "Leave Management"])

            with tab1:
                if my_tasks.empty:
                    st.info("No tasks assigned yet.")
                else:
                    for _, r in my_tasks.iterrows():
                        st.markdown(f"### {r.get('task')} — {r.get('status', 'Pending')}")
                        comp = st.slider("Completion %", 0, 100, int(r.get("completion", 0)), key=r.get("_id"))
                        if st.button(f"Update {r.get('task')}", key=f"upd_{r.get('_id')}"):
                            r["completion"] = comp
                            r["marks"] = float(lin_reg.predict([[comp]])[0])
                            r["status"] = "In Progress" if comp < 100 else "Completed"
                            upsert_data(r.get("_id"), r)
                            st.success(f"Updated {r.get('task')} to {comp}%")

            with tab2:
                st.subheader("Leave Management")
                typ = st.selectbox("Leave Type", ["Casual", "Sick", "Earned"])
                f = st.date_input("From Date")
                t = st.date_input("To Date")
                reason = st.text_area("Reason")
                if st.button("Submit Leave Request"):
                    lid = str(uuid.uuid4())
                    md = {"employee": name, "type": typ, "from": str(f), "to": str(t), "reason": reason, "status": "Pending", "submitted": now()}
                    upsert_data(lid, md)
                    st.success("Leave submitted successfully.")
                if not my_leaves.empty:
                    st.dataframe(my_leaves[["type", "from", "to", "reason", "status"]], use_container_width=True)

# ---------------------------------------------------------
# CLIENT DASHBOARD
# ---------------------------------------------------------
elif role == "Client":
    st.header("Client Review Portal")
    company = st.text_input("Company Name")
    df = fetch_all()
    if not df.empty:
        dfc = df[df["company"].str.lower() == company.lower()] if company else pd.DataFrame()
        if dfc.empty:
            st.info("No tasks for this company.")
        else:
            tab1, tab2 = st.tabs(["Initial Review", "Re-Evaluation"])
            with tab1:
                unrev = dfc[dfc["client_reviewed"] != True]
                if unrev.empty:
                    st.info("No pending reviews.")
                else:
                    task = st.selectbox("Select Task", unrev["task"].unique())
                    fb = st.text_area("Feedback")
                    rating = st.slider("Rating (1–5)", 1, 5, 3)
                    if st.button("Submit Review"):
                        rec = unrev[unrev["task"] == task].iloc[0].to_dict()
                        rec.update({"client_feedback": fb, "client_rating": rating, "client_reviewed": True, "client_approved_on": now()})
                        upsert_data(rec.get("_id"), rec)
                        st.success("Review submitted.")
            with tab2:
                revd = dfc[dfc["client_reviewed"] == True]
                if revd.empty:
                    st.info("No tasks eligible for re-evaluation.")
                else:
                    task = st.selectbox("Select Task", revd["task"].unique())
                    fb = st.text_area("Re-Evaluation Feedback")
                    rating = st.slider("New Rating (1–5)", 1, 5, 3)
                    if st.button("Submit Re-Evaluation"):
                        rec = revd[revd["task"] == task].iloc[0].to_dict()
                        rec.update({"client_reval_feedback": fb, "client_reval_rating": rating, "client_revaluated": True, "client_reval_on": now()})
                        upsert_data(rec.get("_id"), rec)
                        st.success("Re-evaluation submitted successfully.")

# ---------------------------------------------------------
# HR DASHBOARD
# ---------------------------------------------------------
elif role == "HR":
    st.header("HR Intelligence & Performance Dashboard")
    df = fetch_all()
    if df.empty:
        st.info("No HR data yet.")
    else:
        for c in ["employee", "department", "task", "completion", "marks"]:
            if c not in df.columns:
                df[c] = ""
        df["completion"] = pd.to_numeric(df["completion"], errors="coerce").fillna(0)
        df["marks"] = pd.to_numeric(df["marks"], errors="coerce").fillna(0)

        tab1, tab2 = st.tabs(["Performance Overview", "Clustering Analytics"])

        with tab1:
            st.subheader("Performance Summary")
            st.dataframe(df[["employee", "department", "task", "completion", "marks"]], use_container_width=True)
            st.metric("Average Completion", f"{df['completion'].mean():.1f}%")
            st.metric("Average Marks", f"{df['marks'].mean():.2f}")
            st.metric("Active Employees", df["employee"].nunique())

        with tab2:
            st.subheader("Clustering Insights")
            if len(df) > 3:
                km = KMeans(n_clusters=3, n_init=10, random_state=42)
                df["cluster"] = km.fit_predict(df[["completion", "marks"]])
                fig = px.scatter(
                    df, x="completion", y="marks",
                    color=df["cluster"].astype(str),
                    hover_data=["employee", "department", "task"],
                    title="Performance Clusters"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data for clustering.")
