# ============================================================
# main.py — Enterprise Workforce Intelligence System (Final 360° Version)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import json, uuid, time
from datetime import datetime, date
from sklearn.linear_model import LinearRegression, LogisticRegression
import plotly.express as px

# ============================================================
# Optional integrations
# ============================================================
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except Exception:
    PINECONE_AVAILABLE = False

# ============================================================
# Streamlit config
# ============================================================
st.set_page_config(page_title="Enterprise Workforce Intelligence System", layout="wide")
st.title("SmartWorkAI")

# ============================================================
# Pinecone setup
# ============================================================
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
INDEX_NAME = "task"
DIMENSION = 1024

pc, index = None, None
if PINECONE_AVAILABLE and PINECONE_API_KEY:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        existing = [i["name"] for i in pc.list_indexes()]
        if INDEX_NAME not in existing:
            INDEX_NAME = existing[0]
        index = pc.Index(INDEX_NAME)
    except Exception as e:
        st.warning(f"Pinecone connection failed — running in local mode. ({e})")
else:
    st.warning("Pinecone API key missing — using local memory mode.")

# ============================================================
# Helper Functions
# ============================================================
def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def rand_vec():
    np.random.seed(int(time.time()) % 10000)
    return np.random.rand(DIMENSION).tolist()

def safe_meta(md):
    if isinstance(md, pd.Series):
        md = md.to_dict()
    elif not isinstance(md, dict):
        md = dict(md)
    clean = {}
    for k, v in md.items():
        try:
            if isinstance(v, (pd.Series, np.ndarray, list, dict)):
                clean[str(k)] = json.dumps(np.array(v).tolist(), ensure_ascii=False)
            elif pd.isna(v):
                clean[str(k)] = ""
            elif isinstance(v, (np.generic,)):
                clean[str(k)] = v.item()
            else:
                clean[str(k)] = str(v)
        except Exception:
            clean[str(k)] = str(v)
    return clean

def upsert_data(id_, md):
    id_ = str(id_)
    if isinstance(md, pd.Series):
        md = md.to_dict()
    elif not isinstance(md, dict):
        md = dict(md)
    if not index:
        local = st.session_state.setdefault("LOCAL_DATA", {})
        local[id_] = safe_meta(md)
        return True
    try:
        index.upsert(vectors=[{"id": id_, "values": rand_vec(), "metadata": safe_meta(md)}])
        return True
    except Exception as e:
        st.error(f"Pinecone upsert failed: {e}")
        return False

def fetch_all():
    if not index:
        local = st.session_state.get("LOCAL_DATA", {})
        if not local:
            return pd.DataFrame()
        return pd.DataFrame([{"_id": k, **v} for k, v in local.items()])
    try:
        stats = index.describe_index_stats()
        if stats.get("total_vector_count", 0) == 0:
            return pd.DataFrame()
        res = index.query(vector=rand_vec(), top_k=1000, include_metadata=True)
        rows = []
        for m in res["matches"]:
            md = m["metadata"]
            md["_id"] = m["id"]
            rows.append(md)
        return pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"Fetch failed: {e}")
        return pd.DataFrame()

# ============================================================
# AI Models
# ============================================================
lin_reg = LinearRegression().fit([[0], [50], [100]], [0, 2.5, 5])
log_reg = LogisticRegression().fit([[0], [50], [100]], [0, 1, 1])

# ============================================================
# Role selection
# ============================================================
role = st.sidebar.selectbox("Access As", ["Manager", "Team Member", "Client", "HR Administrator"])

# ============================================================
# MANAGER
# ============================================================
if role == "Manager":
    st.header("👨‍💼 Manager Command Center — 360° Overview")
    tabs = st.tabs(["Task Management", "Feedback", "Leave Management", "Meeting Management", "360° Overview"])

    # ---------------- Task Management ----------------
    with tabs[0]:
        st.subheader("Assign New Task")
        with st.form("assign_task"):
            company = st.text_input("Company Name")
            dept = st.text_input("Department")
            emp = st.text_input("Employee Name")
            task = st.text_input("Task Title")
            desc = st.text_area("Task Description")
            deadline = st.date_input("Deadline", value=date.today())
            submit = st.form_submit_button("Assign Task")
            if submit and emp and task:
                tid = str(uuid.uuid4())
                md = {
                    "type": "Task", "company": company, "department": dept,
                    "employee": emp, "task": task, "description": desc,
                    "deadline": str(deadline), "completion": 0, "marks": 0,
                    "status": "Assigned", "created": now()
                }
                upsert_data(tid, md)
                st.success(f"Task '{task}' assigned to {emp}")

    # ---------------- Feedback ----------------
    with tabs[1]:
        st.subheader("Manager Feedback & Evaluation")
        df_all = fetch_all()
        if df_all.empty:
            st.info("No tasks found.")
        else:
            search_query = st.text_input("Search by Employee or Task").strip().lower()
            if not search_query:
                st.warning("Enter employee name or task title to review.")
            else:
                df_tasks = df_all[df_all["type"].astype(str).str.lower() == "task"]
                results = df_tasks[
                    df_tasks["employee"].astype(str).str.lower().str.contains(search_query, na=False)
                    | df_tasks["task"].astype(str).str.lower().str.contains(search_query, na=False)
                ]
                results = results[results["completion"].astype(float) >= 75]
                if results.empty:
                    st.error("No matching completed tasks found.")
                else:
                    rec = results.iloc[0].to_dict()
                    emp = rec.get("employee", "?")
                    task = rec.get("task", "?")
                    curr_comp = int(float(rec.get("completion", 0)))
                    st.markdown(f"### **{emp} — {task}**")
                    new_comp = st.slider("Completion %", 0, 100, curr_comp)
                    comment = st.text_area("Manager Feedback", rec.get("manager_feedback", ""))
                    if st.button("Save Review"):
                        predicted_marks = float(lin_reg.predict([[new_comp]])[0])
                        status_val = log_reg.predict([[new_comp]])[0]
                        status_text = "On Track" if status_val == 1 else "Delayed"
                        updated = {
                            **rec,
                            "completion": new_comp,
                            "marks": predicted_marks,
                            "status": status_text,
                            "manager_feedback": comment,
                            "manager_reviewed_on": now()
                        }
                        upsert_data(rec.get("_id", str(uuid.uuid4())), updated)
                        st.success(f"Feedback saved for {emp} — {task}")

    # ---------------- Leave Management ----------------
    with tabs[2]:
        st.subheader("Leave Management")
        df_all = fetch_all()
        if df_all.empty:
            st.info("No leave requests.")
        else:
            leaves = df_all[df_all["type"].astype(str).str.lower() == "leave"].drop_duplicates(subset=["employee", "from", "to"], keep="last")
            pending = leaves[leaves["status"].astype(str).str.lower() == "pending"]
            if pending.empty:
                st.success("No pending leave requests.")
            else:
                for i, r in pending.iterrows():
                    emp = r.get("employee", "")
                    frm, to = r.get("from", ""), r.get("to", "")
                    reason = r.get("reason", "-")
                    st.markdown(f"**{emp}** → {frm} to {to} — _{reason}_")
                    decision = st.radio(
                        f"Decision for {emp}", ["Approve", "Reject"], key=f"dec_{i}", horizontal=True
                    )
                    if st.button(f"Submit Decision for {emp}", key=f"btn_{i}"):
                        r["status"] = "Approved" if decision == "Approve" else "Rejected"
                        r["approved_on"] = now()
                        upsert_data(r.get("_id", str(uuid.uuid4())), r)
                        st.success(f"Leave {r['status']} for {emp}")
                        st.experimental_rerun()

    # ---------------- Meeting Management ----------------
    with tabs[3]:
        st.subheader("Meeting Management")
        with st.form("add_meeting"):
            title = st.text_input("Meeting Title")
            m_date = st.date_input("Meeting Date", date.today())
            m_time = st.text_input("Time", "10:00 AM")
            attendees = st.text_area("Attendees (comma separated)")
            notes = st.text_area("Meeting Notes (optional)")
            add = st.form_submit_button("Schedule Meeting")
            if add and title:
                mid = str(uuid.uuid4())
                md = {
                    "type": "Meeting", "meeting_title": title,
                    "meeting_date": str(m_date), "meeting_time": m_time,
                    "attendees": attendees, "notes": notes, "created": now()
                }
                upsert_data(mid, md)
                st.success(f" Meeting '{title}' scheduled successfully!")

        df_meet = fetch_all()
        df_meet = df_meet[df_meet["type"].astype(str).str.lower() == "meeting"] if not df_meet.empty else pd.DataFrame()
        if not df_meet.empty:
            st.markdown("### 📅 Upcoming Meetings")
            st.dataframe(df_meet[["meeting_title", "meeting_date", "meeting_time", "attendees"]], use_container_width=True)

    # ---------------- 360° Overview ----------------
    with tabs[4]:
        st.subheader("360° Overview — Performance, Meetings & Leave Insights")
        df_all = fetch_all()
        if df_all.empty:
            st.info("No data available.")
        else:
            tasks = df_all[df_all["type"].astype(str).str.lower() == "task"]
            leaves = df_all[df_all["type"].astype(str).str.lower() == "leave"]
            meets = df_all[df_all["type"].astype(str).str.lower() == "meeting"]

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Tasks", len(tasks))
            c2.metric("Meetings", len(meets))
            c3.metric("Leave Requests", len(leaves))

            if not tasks.empty:
                fig1 = px.bar(tasks, x="employee", y="completion", color="status", title="Task Completion by Employee")
                st.plotly_chart(fig1, use_container_width=True)
                if "department" in tasks.columns:
                    fig2 = px.box(tasks, x="department", y="completion", title="Department Performance Distribution")
                    st.plotly_chart(fig2, use_container_width=True)
            if not leaves.empty:
                leaves["status"] = leaves["status"].astype(str).str.title()
                fig3 = px.histogram(leaves, x="status", color="status", title="Leave Status Overview")
                st.plotly_chart(fig3, use_container_width=True)

# ============================================================
# TEAM MEMBER
# ============================================================
elif role == "Team Member":
    st.header("Team Member Workspace")
    name = st.text_input("Enter your name")
    if name:
        df = fetch_all()
        if not df.empty:
            my = df[(df["type"] == "Task") & (df["employee"].str.lower() == name.lower())]
            for _, r in my.iterrows():
                st.markdown(f"**{r['task']}** — {r['status']}")
                val = st.slider("Progress %", 0, 100, int(float(r["completion"])), key=r["_id"])
                if st.button(f"Update {r['task']}", key=f"upd_{r['_id']}"):
                    r["completion"] = val
                    r["marks"] = float(lin_reg.predict([[val]])[0])
                    r["status"] = "In Progress" if val < 100 else "Completed"
                    upsert_data(r["_id"], r)
                    st.success("Task updated successfully.")

# ============================================================
# CLIENT
# ============================================================
elif role == "Client":
    st.header("Client Review Portal")
    company = st.text_input("Enter Company Name")
    if company:
        df = fetch_all()
        df_c = df[df["company"].astype(str).str.lower() == company.lower()] if not df.empty else pd.DataFrame()
        if df_c.empty:
            st.warning("No company data found.")
        else:
            st.dataframe(df_c[df_c["type"] == "Task"][["employee", "task", "completion", "marks", "status"]], use_container_width=True)

# ============================================================
# HR ADMIN
# ============================================================
elif role == "HR Administrator":
    st.header("HR Analytics & Leave Intelligence")
    df_all = fetch_all()
    if df_all.empty:
        st.info("No data available.")
    else:
        tasks = df_all[df_all["type"].astype(str).str.lower() == "task"]
        leaves = df_all[df_all["type"].astype(str).str.lower() == "leave"]
        if not tasks.empty:
            fig = px.scatter(tasks, x="completion", y="marks", color="status", hover_data=["employee"], title="Employee Performance Overview")
            st.plotly_chart(fig, use_container_width=True)
        if not leaves.empty:
            leaves["status"] = leaves["status"].astype(str).str.title()
            st.dataframe(leaves[["employee", "from", "to", "reason", "status"]].fillna(""), use_container_width=True)
