# ============================================================
# main.py — Enterprise Workforce Intelligence System (Final Intelligence Version)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import json, uuid, time
from datetime import datetime, date
import plotly.express as px
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Optional modules
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except Exception:
    PINECONE_AVAILABLE = False

# ============================================================
# Setup & Configuration
# ============================================================
st.set_page_config(page_title="Enterprise Workforce Intelligence", layout="wide")
st.title("Enterprise Workforce Intelligence System")
st.caption("AI-driven performance management • Intelligent feedback • Visual analytics")

# Keys & constants
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
INDEX_NAME = "task"
DIMENSION = 1024

# Pinecone setup
pc, index = None, None
if PINECONE_AVAILABLE and PINECONE_API_KEY:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        existing = [i["name"] for i in pc.list_indexes()]
        if INDEX_NAME not in existing:
            pc.create_index(name=INDEX_NAME, dimension=DIMENSION, metric="cosine",
                            spec=ServerlessSpec(cloud="aws", region="us-east-1"))
        index = pc.Index(INDEX_NAME)
        st.success(f"✅ Connected to Pinecone index: {INDEX_NAME}")
    except Exception as e:
        st.warning(f"Pinecone init failed: {e}")
else:
    st.warning("⚠️ Pinecone not available — using local mode.")

# ============================================================
# Helper functions
# ============================================================
def now(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def rand_vec(): return np.random.rand(DIMENSION).tolist()

def safe_meta(md):
    clean = {}
    for k, v in (md or {}).items():
        try:
            if isinstance(v, (dict, list)): v = json.dumps(v)
            elif pd.isna(v): v = ""
            clean[str(k)] = str(v)
        except Exception:
            clean[str(k)] = str(v)
    return clean

def upsert_data(id_, md):
    id_ = str(id_)
    if not index:
        local = st.session_state.setdefault("LOCAL_DATA", {})
        local[id_] = md
        return True
    try:
        index.upsert([{"id": id_, "values": rand_vec(), "metadata": safe_meta(md)}])
        return True
    except Exception as e:
        st.warning(f"Pinecone upsert failed: {e}")
        return False

def fetch_all():
    if not index:
        local = st.session_state.get("LOCAL_DATA", {})
        if not local: return pd.DataFrame()
        return pd.DataFrame([{"_id": k, **v} for k, v in local.items()])
    try:
        res = index.query(vector=rand_vec(), top_k=1000, include_metadata=True)
        rows = []
        for m in res["matches"]:
            md = m["metadata"]; md["_id"] = m["id"]; rows.append(md)
        return pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"Fetch failed: {e}"); return pd.DataFrame()

# ML
lin_reg = LinearRegression().fit([[0], [50], [100]], [0, 2.5, 5])
log_reg = LogisticRegression().fit([[0], [50], [100]], [0, 1, 1])

# ============================================================
# Role Selector
# ============================================================
role = st.sidebar.selectbox("Access Portal As", ["Manager", "Team Member", "Client", "HR Administrator"])

# ============================================================
# MANAGER DASHBOARD
# ============================================================
if role == "Manager":
    st.header("Manager Dashboard — Tasks, Feedback & Analytics")
    tabs = st.tabs(["Task Management", "Feedback", "Leave Management", "Overview"])

    # Task management
    with tabs[0]:
        st.subheader("Assign New Task")
        with st.form("assign_task"):
            emp = st.text_input("Employee Name")
            task = st.text_input("Task Title")
            dept = st.text_input("Department")
            company = st.text_input("Company")
            deadline = st.date_input("Deadline", value=date.today())
            desc = st.text_area("Task Description")
            submit = st.form_submit_button("Assign Task")
            if submit:
                tid = str(uuid.uuid4())
                md = {"type": "Task", "employee": emp, "task": task, "department": dept,
                      "company": company, "deadline": str(deadline), "completion": 0,
                      "marks": 0, "status": "Assigned", "created": now(), "description": desc}
                upsert_data(tid, md)
                st.success(f"Task '{task}' assigned to {emp}")

    # Feedback
    with tabs[1]:
        st.subheader("Manager Feedback & Evaluation")
        df = fetch_all()
        if df.empty: st.info("No data.")
        else:
            df = df[df.get("type") == "Task"]
            search = st.text_input("Search by Employee or Task").lower()
            if search:
                df = df[df["employee"].str.lower().str.contains(search) | df["task"].str.lower().str.contains(search)]
            df["completion"] = pd.to_numeric(df.get("completion", 0), errors="coerce").fillna(0)
            ready = df[df["completion"] >= 75]
            if ready.empty: st.info("No ready tasks.")
            else:
                r = ready.iloc[0].to_dict()
                st.write(f"**{r['employee']} — {r['task']}**")
                comp = st.slider("Completion %", 0, 100, int(float(r["completion"])))
                fb = st.text_area("Manager Feedback", r.get("manager_feedback", ""))
                if st.button("Save Review"):
                    marks = float(lin_reg.predict([[comp]])[0])
                    status = "On Track" if log_reg.predict([[comp]])[0] else "Delayed"
                    r.update({"completion": comp, "marks": marks, "status": status,
                              "manager_feedback": fb, "manager_reviewed_on": now()})
                    upsert_data(r["_id"], r)
                    st.success("✅ Review saved!")

    # Leave
    with tabs[2]:
        st.subheader("Leave Management")
        df = fetch_all()
        leaves = df[df["type"] == "Leave"] if not df.empty else pd.DataFrame()
        search = st.text_input("Search Employee").lower()
        if not leaves.empty and search:
            leaves = leaves[leaves["employee"].str.lower().str.contains(search)]
        pending = leaves[leaves["status"].str.lower() == "pending"]
        if pending.empty: st.success("No pending leaves.")
        else:
            for i, r in pending.iterrows():
                emp = r["employee"]
                st.markdown(f"**{emp}** → {r['from']} to {r['to']} — {r['reason']}")
                dec = st.radio(f"Decision for {emp}", ["Approve", "Reject"], key=f"d_{i}")
                if st.button(f"Submit {emp}", key=f"b_{i}"):
                    r["status"] = "Approved" if dec == "Approve" else "Rejected"
                    r["approved_on"] = now()
                    upsert_data(r["_id"], r)
                    st.success(f"Leave {r['status']} for {emp}")
                    st.experimental_rerun()

    # Overview with graphs
    with tabs[3]:
        st.subheader("Team Overview")
        df = fetch_all()
        if df.empty: st.info("No data.")
        else:
            df = df[df["type"] == "Task"]
            df["completion"] = pd.to_numeric(df["completion"], errors="coerce").fillna(0)
            fig = px.bar(df, x="employee", y="completion", color="department", title="Task Completion by Employee")
            st.plotly_chart(fig, use_container_width=True)
            avg_dept = df.groupby("department")["completion"].mean().reset_index()
            fig2 = px.pie(avg_dept, values="completion", names="department", title="Departmental Average Completion")
            st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# TEAM MEMBER DASHBOARD
# ============================================================
elif role == "Team Member":
    st.header("Team Member Dashboard")
    name = st.text_input("Enter your name")
    if name:
        df = fetch_all()
        my = df[(df["type"] == "Task") & (df["employee"].str.lower() == name.lower())]
        for _, r in my.iterrows():
            st.write(f"**{r['task']}** — {r['status']}")
            val = st.slider(f"Progress {r['task']}", 0, 100, int(float(r["completion"])))
            if st.button(f"Update {r['task']}", key=r["_id"]):
                r["completion"] = val
                r["marks"] = float(lin_reg.predict([[val]])[0])
                r["status"] = "Completed" if val >= 100 else "In Progress"
                upsert_data(r["_id"], r)
                st.success("Updated.")

        st.subheader("Leave Request")
        lt = st.selectbox("Leave Type", ["Casual", "Sick", "Earned"])
        f, t = st.date_input("From"), st.date_input("To")
        reason = st.text_area("Reason")
        if st.button("Submit Leave"):
            lid = str(uuid.uuid4())
            upsert_data(lid, {"type": "Leave", "employee": name, "from": str(f),
                              "to": str(t), "reason": reason, "status": "Pending", "submitted": now()})
            st.success("Leave requested.")

# ============================================================
# CLIENT DASHBOARD
# ============================================================
elif role == "Client":
    st.header("Client Review & Feedback with Similarity")
    company = st.text_input("Company Name")
    if company:
        df = fetch_all()
        df = df[df["type"] == "Task"]
        df_client = df[df["company"].str.lower() == company.lower()]
        if df_client.empty: st.info("No data.")
        else:
            st.dataframe(df_client[["employee","task","completion","status","marks"]])
            ready = df_client[df_client["completion"].astype(float) >= 75]
            sel = st.selectbox("Select task for feedback", ready["task"].unique())
            if sel:
                fb = st.text_area("Feedback")
                rating = st.slider("Rating", 1, 5, 4)
                if st.button("Submit Feedback"):
                    rec = ready[ready["task"] == sel].iloc[0].to_dict()
                    rec.update({"client_feedback": fb, "client_rating": rating,
                                "status": "Client Approved", "client_reviewed_on": now()})
                    upsert_data(rec["_id"], rec)
                    st.success("Feedback saved.")

                # Similarity analysis
                if len(df_client) > 1 and fb.strip():
                    try:
                        corpus = df_client["task"].astype(str).tolist() + [fb]
                        vect = TfidfVectorizer().fit_transform(corpus)
                        sims = cosine_similarity(vect[-1], vect[:-1]).flatten()
                        top_idx = np.argsort(sims)[::-1][:3]
                        st.markdown("### 🔍 Similar past tasks:")
                        for i in top_idx:
                            st.write(f"- {df_client.iloc[i]['task']} ({round(sims[i]*100,2)}% similar)")
                    except Exception as e:
                        st.error(f"Similarity check failed: {e}")

# ============================================================
# HR ADMIN DASHBOARD
# ============================================================
elif role == "HR Administrator":
    st.header("HR Analytics Dashboard")
    df = fetch_all()
    if df.empty: st.info("No data.")
    else:
        df["completion"] = pd.to_numeric(df.get("completion", 0), errors="coerce").fillna(0)
        df["marks"] = pd.to_numeric(df.get("marks", 0), errors="coerce").fillna(0)
        tasks = df[df["type"] == "Task"]
        leaves = df[df["type"] == "Leave"]

        tabs = st.tabs(["Performance Clusters", "Leave Tracker"])

        # --- Cluster graph ---
        with tabs[0]:
            if len(tasks) >= 2:
                km = KMeans(n_clusters=min(3, len(tasks)), random_state=42, n_init=10)
                tasks["cluster"] = km.fit_predict(tasks[["completion","marks"]])
                fig = px.scatter(tasks, x="completion", y="marks", color="cluster", hover_data=["employee","task"],
                                 title="Employee Performance Clusters")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(tasks[["employee","task","completion","marks","cluster"]])
            else:
                st.info("Not enough task data for clustering.")

        # --- Leave analytics ---
        with tabs[1]:
            if leaves.empty:
                st.info("No leave data.")
            else:
                counts = leaves["status"].value_counts().reset_index()
                counts.columns = ["Status","Count"]
                fig1 = px.pie(counts, names="Status", values="Count", title="Leave Status Distribution")
                st.plotly_chart(fig1, use_container_width=True)
                emp_leave = leaves.groupby("employee").size().reset_index(name="Total Leaves")
                fig2 = px.bar(emp_leave, x="employee", y="Total Leaves", title="Leaves per Employee")
                st.plotly_chart(fig2, use_container_width=True)
