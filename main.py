# main.py — AI Workforce Intelligence Platform (Enterprise Edition)
# Author: Refined for professional EY-tier structure
# Dependencies: streamlit, pinecone-client, pandas, scikit-learn, plotly, huggingface-hub

import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import pandas as pd
import uuid, json, time
from datetime import datetime, date
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
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
role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client", "Admin"])

# ---------------------------------------------------------
# TEAM MEMBER DASHBOARD
# ---------------------------------------------------------
if role == "Team Member":
    st.header("Team Member Dashboard")
    name = st.text_input("Enter your name")

    if name:
        df = fetch_all()
        if df.empty:
            st.info("No data found.")
        else:
            df["type_flag"] = df.apply(
                lambda x: "Leave" if "type" in x and x["type"] in ["Casual", "Sick", "Earned"] else "Task", axis=1
            )
            my_tasks = df[(df["employee"].str.lower() == name.lower()) & (df["type_flag"] == "Task")]
            my_leaves = df[(df["employee"].str.lower() == name.lower()) & (df["type_flag"] == "Leave")]

            tab1, tab2 = st.tabs(["My Tasks", "Leave Management"])

            # ---- TAB 1: MY TASKS ----
            with tab1:
                st.subheader("My Tasks")
                if my_tasks.empty:
                    st.info("No tasks assigned yet.")
                else:
                    my_tasks["completion"] = pd.to_numeric(my_tasks.get("completion", 0), errors="coerce").fillna(0)
                    for _, r in my_tasks.iterrows():
                        task_title = r.get("task", "Untitled Task")
                        status = r.get("status", "Not Started")
                        completion = int(r.get("completion", 0))
                        st.markdown(f"**Task:** {task_title} | **Status:** {status}")
                        comp = st.slider("Completion %", 0, 100, completion, key=r.get("_id"))
                        if st.button(f"Update {task_title}", key=f"upd_{r.get('_id')}"):
                            r["completion"] = comp
                            r["marks"] = float(lin_reg.predict([[comp]])[0])
                            r["status"] = "In Progress" if comp < 100 else "Completed"
                            upsert_data(r.get("_id") or str(uuid.uuid4()), r)
                            st.success(f"Task '{task_title}' updated successfully.")

            # ---- TAB 2: LEAVE MANAGEMENT ----
            with tab2:
                st.subheader("Leave Management")
                typ = st.selectbox("Leave Type", ["Casual", "Sick", "Earned"])
                f = st.date_input("From Date")
                t = st.date_input("To Date")
                reason = st.text_area("Reason for Leave")
                if st.button("Submit Leave Request"):
                    lid = str(uuid.uuid4())
                    md = {
                        "employee": name,
                        "type": typ,
                        "from": str(f),
                        "to": str(t),
                        "reason": reason,
                        "status": "Pending",
                        "submitted": now()
                    }
                    upsert_data(lid, md)
                    st.success("Leave submitted successfully.")

                if not my_leaves.empty:
                    st.markdown("### My Leave History")
                    my_leaves = my_leaves[["type", "from", "to", "reason", "status"]].fillna("")
                    st.dataframe(my_leaves, use_container_width=True)
                else:
                    st.info("No leave history available.")

# ---------------------------------------------------------
# CLIENT DASHBOARD (WITH RE-EVALUATION)
# ---------------------------------------------------------
elif role == "Client":
    st.header("Client Review Center")

    company = st.text_input("Company Name")
    if company:
        df = fetch_all()
        df_client = df[df["company"].str.lower() == company.lower()] if not df.empty else pd.DataFrame()

        if df_client.empty:
            st.info("No tasks found.")
        else:
            tab1, tab2 = st.tabs(["Initial Review", "Re-Evaluation"])

            # ---- TAB 1: INITIAL REVIEW ----
            with tab1:
                st.subheader("Pending Reviews")
                unreviewed = df_client[df_client["client_reviewed"] != True]
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
                else:
                    st.info("No unreviewed tasks available.")

            # ---- TAB 2: RE-EVALUATION ----
            with tab2:
                st.subheader("Re-Evaluation")
                reviewed = df_client[df_client["client_reviewed"] == True]
                if reviewed.empty:
                    st.info("No tasks available for re-evaluation.")
                else:
                    task_sel = st.selectbox("Select Task to Re-Evaluate", reviewed["task"].unique())
                    fb = st.text_area("Updated Feedback (after improvement)")
                    rating = st.slider("Re-Evaluation Rating (1–5)", 1, 5, 3)
                    if st.button("Submit Re-Evaluation"):
                        row = reviewed[reviewed["task"] == task_sel].iloc[0].to_dict()
                        row.update({
                            "client_feedback_reval": fb,
                            "client_rating_reval": rating,
                            "client_revaluated": True,
                            "client_reval_on": now()
                        })
                        upsert_data(row.get("_id") or str(uuid.uuid4()), row)
                        st.success("Re-Evaluation submitted successfully.")

# ---------------------------------------------------------
# ADMIN DASHBOARD (WITH CLUSTERING)
# ---------------------------------------------------------
elif role == "Admin":
    st.header("Admin Dashboard & HR Intelligence")
    df = fetch_all()

    if df.empty:
        st.info("No data available.")
    else:
        for col in ["employee", "department", "task", "completion", "marks"]:
            if col not in df.columns:
                df[col] = ""
        df["completion"] = pd.to_numeric(df["completion"], errors="coerce").fillna(0)
        df["marks"] = pd.to_numeric(df["marks"], errors="coerce").fillna(0)

        tab1, tab2 = st.tabs(["Performance Summary", "Clustering Insights"])

        # ---- TAB 1: PERFORMANCE ----
        with tab1:
            st.subheader("Performance Summary")
            st.dataframe(df[["employee", "department", "task", "completion", "marks"]], use_container_width=True)
            avg_c = df["completion"].mean()
            avg_m = df["marks"].mean()
            st.metric("Avg Completion", f"{avg_c:.1f}%")
            st.metric("Avg Marks", f"{avg_m:.2f}")

        # ---- TAB 2: CLUSTERING ----
        with tab2:
            st.subheader("Employee Clustering Insights")
            if len(df) > 2:
                km = KMeans(n_clusters=3, n_init=10, random_state=42)
                df["cluster"] = km.fit_predict(df[["completion", "marks"]])
                fig = px.scatter(
                    df,
                    x="completion",
                    y="marks",
                    color=df["cluster"].astype(str),
                    hover_data=["employee", "department", "task"],
                    title="Performance Clusters"
                )
                st.plotly_chart(fig, use_container_width=True)
                st.info("Cluster 0: Top Performers | Cluster 1: Average | Cluster 2: Needs Improvement")
            else:
                st.info("Not enough data for clustering.")
