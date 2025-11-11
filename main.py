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
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
except Exception:
    PINECONE_AVAILABLE = False

# ============================================================
# Streamlit setup
# ============================================================
st.set_page_config(page_title="Enterprise Workforce Intelligence System", layout="wide")
st.title("🏢 Enterprise Workforce Intelligence System")
st.caption("AI-Driven Workforce Analytics • Productivity Intelligence • 360° Performance Overview")

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
        indexes = [i["name"] for i in pc.list_indexes()]
        if INDEX_NAME not in indexes:
            INDEX_NAME = indexes[0]
        index = pc.Index(INDEX_NAME)
        st.success(f"✅ Connected to Pinecone index: {INDEX_NAME}")
    except Exception as e:
        st.warning(f"Pinecone connection failed — running in local mode ({e})")
else:
    st.warning("⚠️ Pinecone key missing — using local mode.")

# ============================================================
# Utility functions
# ============================================================
def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def rand_vec():
    np.random.seed(int(time.time()) % 10000)
    return np.random.rand(DIMENSION).tolist()

def safe_meta(md):
    """Ensure metadata is JSON safe before upserting to Pinecone."""
    if isinstance(md, pd.Series):
        md = md.to_dict()
    elif not isinstance(md, dict):
        md = dict(md)
    clean = {}
    for k, v in md.items():
        try:
            if isinstance(v, (dict, list, np.ndarray)):
                clean[str(k)] = json.dumps(v, ensure_ascii=False)
            elif pd.isna(v):
                clean[str(k)] = ""
            else:
                clean[str(k)] = str(v)
        except Exception:
            clean[str(k)] = str(v)
    return clean

def upsert_data(id_, md):
    id_ = str(id_)
    md = safe_meta(md)
    if not index:
        local = st.session_state.setdefault("LOCAL_DATA", {})
        local[id_] = md
        return True
    try:
        index.upsert(vectors=[{"id": id_, "values": rand_vec(), "metadata": md}])
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
# ML helpers
# ============================================================
lin_reg = LinearRegression().fit([[0], [50], [100]], [0, 2.5, 5])
log_reg = LogisticRegression().fit([[0], [50], [100]], [0, 1, 1])

# ============================================================
# Role selector
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
                st.success(f"✅ Task '{task}' assigned to {emp}")

    # ---------------- Feedback ----------------
    with tabs[1]:
        st.subheader("Manager Feedback & Evaluation")
        df = fetch_all()
        if df.empty:
            st.info("No tasks found.")
        else:
            search = st.text_input("Search by Employee or Task").strip().lower()
            if not search:
                st.warning("Enter an employee or task name to review.")
            else:
                df_task = df[df["type"].astype(str).str.lower() == "task"]
                res = df_task[df_task["employee"].astype(str).str.lower().str.contains(search, na=False)
                              | df_task["task"].astype(str).str.lower().str.contains(search, na=False)]
                if res.empty:
                    st.error("No completed tasks found.")
                else:
                    rec = res.iloc[0].to_dict()
                    emp = rec.get("employee", "")
                    task = rec.get("task", "")
                    comp = int(float(rec.get("completion", 0)))
                    st.markdown(f"### **{emp} — {task}**")
                    new_comp = st.slider("Completion %", 0, 100, comp)
                    comment = st.text_area("Manager Feedback", rec.get("manager_feedback", ""))
                    if st.button("Save Review"):
                        marks = float(lin_reg.predict([[new_comp]])[0])
                        status_val = log_reg.predict([[new_comp]])[0]
                        updated = {
                            **rec,
                            "completion": new_comp,
                            "marks": marks,
                            "status": "On Track" if status_val == 1 else "Delayed",
                            "manager_feedback": comment,
                            "manager_reviewed_on": now()
                        }
                        upsert_data(rec.get("_id", str(uuid.uuid4())), updated)
                        st.success("✅ Review saved successfully.")

    # ---------------- Leave Management (search + preview) ----------------
    with tabs[2]:
        st.subheader("Leave Management — Search & Approve Requests")
        df = fetch_all()
        leaves = df[df["type"].astype(str).str.lower() == "leave"]
        if leaves.empty:
            st.info("No leave data.")
        else:
            query = st.text_input("Search by Employee Name").strip().lower()
            if not query:
                st.warning("Enter employee name to view leave requests.")
            else:
                match = leaves[leaves["employee"].astype(str).str.lower().str.contains(query, na=False)]
                if match.empty:
                    st.error("No matching leave requests.")
                else:
                    st.dataframe(match[["employee", "leave_type", "from", "to", "reason", "status"]], use_container_width=True)
                    pending = match[match["status"].astype(str).str.lower() == "pending"]
                    if pending.empty:
                        st.success("✅ All requests already processed.")
                    else:
                        for i, row in pending.iterrows():
                            emp = row["employee"]
                            reason = row.get("reason", "-")
                            st.markdown(f"**{emp}** → {row['from']} to {row['to']} — _{reason}_")
                            decision = st.radio(f"Decision for {emp}", ["Approve", "Reject"], key=f"dec_{i}", horizontal=True)
                            if st.button(f"Submit Decision for {emp}", key=f"btn_{i}"):
                                row["status"] = "Approved" if decision == "Approve" else "Rejected"
                                row["approved_on"] = now()
                                upsert_data(row["_id"], row)
                                st.success(f"Leave {row['status']} for {emp}")
                                st.experimental_rerun()

    # ---------------- Meeting Management ----------------
    with tabs[3]:
        st.subheader("Meeting Management")
        with st.form("add_meeting"):
            title = st.text_input("Meeting Title")
            m_date = st.date_input("Meeting Date", date.today())
            m_time = st.text_input("Meeting Time", "10:00 AM")
            attendees = st.text_area("Attendees (comma-separated)")
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
                st.success(f"✅ Meeting '{title}' added successfully.")

        df_meet = fetch_all()
        df_meet = df_meet[df_meet["type"].astype(str).str.lower() == "meeting"]
        if not df_meet.empty:
            st.dataframe(df_meet[["meeting_title", "meeting_date", "meeting_time", "attendees"]], use_container_width=True)

    # ---------------- 360° Overview ----------------
    with tabs[4]:
        st.subheader("360° Overview — Team & Department Insights")
        df_all = fetch_all()
        if df_all.empty:
            st.info("No data available.")
        else:
            tasks = df_all[df_all["type"].astype(str).str.lower() == "task"]
            leaves = df_all[df_all["type"].astype(str).str.lower() == "leave"]
            meets = df_all[df_all["type"].astype(str).str.lower() == "meeting"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Tasks", len(tasks))
            c2.metric("Leaves", len(leaves))
            c3.metric("Meetings", len(meets))
            if not tasks.empty:
                fig1 = px.bar(tasks, x="employee", y="completion", color="status", title="Task Completion by Employee")
                st.plotly_chart(fig1, use_container_width=True)

# ============================================================
# TEAM MEMBER (Enhanced)
# ============================================================
elif role == "Team Member":
    st.header("🧑‍💻 Team Member Dashboard")
    name = st.text_input("Enter your Name")
    if name:
        df_all = fetch_all()
        if df_all.empty:
            st.info("No records found.")
        else:
            tabs_tm = st.tabs(["My Tasks", "My Leave", "My Meetings"])

            # --- My Tasks ---
            with tabs_tm[0]:
                st.subheader("Assigned Tasks")
                my_tasks = df_all[(df_all["type"] == "Task") & (df_all["employee"].str.lower() == name.lower())]
                if my_tasks.empty:
                    st.info("No tasks assigned to you.")
                else:
                    for _, r in my_tasks.iterrows():
                        st.markdown(f"**{r['task']}** — _{r.get('status', 'Unknown')}_")
                        val = st.slider("Progress %", 0, 100, int(float(r.get("completion", 0))), key=r["_id"])
                        if st.button(f"Update {r['task']}", key=f"upd_{r['_id']}"):
                            r["completion"] = val
                            r["marks"] = float(lin_reg.predict([[val]])[0])
                            r["status"] = "In Progress" if val < 100 else "Completed"
                            upsert_data(r["_id"], r)
                            st.success("✅ Progress updated.")

            # --- My Leave ---
            with tabs_tm[1]:
                st.subheader("Leave Management")
                tabs_l = st.tabs(["Apply Leave", "My Leave History"])
                # Apply leave
                with tabs_l[0]:
                    leave_type = st.selectbox("Leave Type", ["Casual", "Sick", "Earned"])
                    f_date = st.date_input("From Date", min_value=date.today())
                    t_date = st.date_input("To Date", min_value=f_date)
                    reason = st.text_area("Reason")
                    if st.button("Submit Leave"):
                        lid = str(uuid.uuid4())
                        md = {
                            "type": "Leave", "employee": name,
                            "leave_type": leave_type, "from": str(f_date),
                            "to": str(t_date), "reason": reason,
                            "status": "Pending", "submitted": now()
                        }
                        upsert_data(lid, md)
                        st.success("✅ Leave request submitted.")
                # Leave history
                with tabs_l[1]:
                    my_leaves = df_all[(df_all["type"].astype(str).str.lower() == "leave") &
                                       (df_all["employee"].astype(str).str.lower() == name.lower())]
                    if my_leaves.empty:
                        st.info("No leave history.")
                    else:
                        st.dataframe(my_leaves[["leave_type", "from", "to", "reason", "status"]].fillna(""),
                                     use_container_width=True)
                        for i, row in my_leaves.iterrows():
                            if row.get("status", "").lower() == "pending":
                                if st.button(f"Cancel Leave ({row['from']} - {row['to']})", key=f"cancel_{i}"):
                                    row["status"] = "Cancelled"
                                    row["cancelled_on"] = now()
                                    upsert_data(row["_id"], row)
                                    st.warning("❌ Leave cancelled.")
                                    st.experimental_rerun()

            # --- My Meetings ---
            with tabs_tm[2]:
                st.subheader("My Meetings")
                df_meet = df_all[df_all["type"].astype(str).str.lower() == "meeting"]
                if df_meet.empty:
                    st.info("No meetings scheduled.")
                else:
                    def is_attendee(row):
                        return name.lower() in str(row.get("attendees", "")).lower()
                    my_meet = df_meet[df_meet.apply(is_attendee, axis=1)]
                    if my_meet.empty:
                        st.info("You have no meetings.")
                    else:
                        st.dataframe(my_meet[["meeting_title", "meeting_date", "meeting_time", "attendees", "notes"]],
                                     use_container_width=True)
