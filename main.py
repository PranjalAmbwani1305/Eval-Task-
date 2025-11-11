# ============================================================
# main.py — Enterprise Workforce Intelligence System (Final Full Version)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import json, uuid, time
from datetime import datetime, date
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import plotly.express as px

# ============================================================
# Optional Pinecone setup
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
# Pinecone config
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
# ML Models
# ============================================================
lin_reg = LinearRegression().fit([[0], [50], [100]], [0, 2.5, 5])
log_reg = LogisticRegression().fit([[0], [50], [100]], [0, 1, 1])

vectorizer = CountVectorizer()
train_texts = ["excellent work", "great job", "poor performance", "needs improvement", "bad result", "amazing effort"]
train_labels = [1, 1, 0, 0, 0, 1]
X_train = vectorizer.fit_transform(train_texts)
svm_clf = SVC(kernel="linear").fit(X_train, train_labels)

# ============================================================
# Role selector
# ============================================================
role = st.sidebar.selectbox("Access As", ["Manager", "Team Member"])

# ============================================================
# MANAGER SECTION
# ============================================================
if role == "Manager":
    st.header("👨‍💼 Manager Dashboard — Task, Leave, Feedback & Meetings")
    df_all = fetch_all()
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
        if df_all.empty:
            st.info("No task data available.")
        else:
            search = st.text_input("Search by Employee Name or Task Title").strip().lower()
            if search:
                df_tasks = df_all[df_all["type"].astype(str).str.lower() == "task"]
                matched = df_tasks[
                    df_tasks["employee"].astype(str).str.lower().str.contains(search) |
                    df_tasks["task"].astype(str).str.lower().str.contains(search)
                ]
                completed = matched[matched["completion"].astype(float) >= 100]
                if completed.empty:
                    st.warning("No completed tasks for review.")
                else:
                    for _, task in completed.iterrows():
                        st.markdown(f"### {task['employee']} — {task['task']}")
                        marks = st.slider("Final Marks", 0.0, 5.0, float(task.get("marks", 0)), key=f"mark_{task['_id']}")
                        fb = st.text_area("Feedback", task.get("manager_feedback", ""), key=f"fb_{task['_id']}")
                        if st.button(f"Save Review for {task['employee']}", key=f"btn_{task['_id']}"):
                            updated = dict(task)
                            updated["marks"] = marks
                            updated["manager_feedback"] = fb
                            updated["status"] = "Reviewed"
                            updated["manager_reviewed_on"] = now()
                            upsert_data(task["_id"], updated)
                            st.success(f"✅ Feedback saved for {task['employee']}")
                            st.experimental_rerun()

    # ---------------- Leave Management ----------------
    with tabs[2]:
        st.subheader("Leave Approval Center")
        df_leave = df_all[df_all["type"].astype(str).str.lower() == "leave"]
        if df_leave.empty:
            st.info("No leave data.")
        else:
            query = st.text_input("Search Employee for Leave").strip().lower()
            if query:
                matched = df_leave[df_leave["employee"].astype(str).str.lower().str.contains(query)]
                pending = matched[matched["status"].astype(str).str.lower() == "pending"]
                if pending.empty:
                    st.success("✅ No pending requests.")
                else:
                    for i, row in pending.iterrows():
                        st.markdown(f"**{row['employee']}** — {row['from']} → {row['to']} — {row['reason']}")
                        decision = st.radio(f"Decision for {row['employee']}", ["Approve", "Reject"], key=f"dec_{i}", horizontal=True)
                        if st.button(f"Finalize Decision for {row['employee']}", key=f"btn_{i}"):
                            row["status"] = "Approved" if decision == "Approve" else "Rejected"
                            row["approved_on"] = now()
                            upsert_data(row["_id"], row)
                            st.success(f"Leave {row['status']} for {row['employee']}")
                            st.experimental_rerun()

    # ---------------- Meeting Management ----------------
    with tabs[3]:
        st.subheader("Schedule and Manage Meetings")
        with st.form("add_meeting"):
            title = st.text_input("Meeting Title")
            m_date = st.date_input("Meeting Date", date.today())
            m_time = st.text_input("Meeting Time", "10:00 AM")
            attendees = st.text_area("Attendees (comma-separated)")
            notes = st.text_area("Notes (optional)")
            add = st.form_submit_button("Add Meeting")
            if add:
                mid = str(uuid.uuid4())
                md = {
                    "type": "Meeting", "meeting_title": title, "meeting_date": str(m_date),
                    "meeting_time": m_time, "attendees": attendees, "notes": notes, "created": now()
                }
                upsert_data(mid, md)
                st.success("✅ Meeting scheduled.")

    # ---------------- 360 Overview ----------------
    with tabs[4]:
        st.subheader("Team Performance Overview")
        if not df_all.empty:
            tasks = df_all[df_all["type"].astype(str).str.lower() == "task"]
            if not tasks.empty:
                fig = px.bar(tasks, x="employee", y="completion", color="status", title="Task Completion by Employee")
                st.plotly_chart(fig, use_container_width=True)
            leaves = df_all[df_all["type"].astype(str).str.lower() == "leave"]
            if not leaves.empty:
                fig2 = px.histogram(leaves, x="status", title="Leave Status Overview")
                st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# TEAM MEMBER SECTION
# ============================================================
elif role == "Team Member":
    st.header("🧑‍💻 Team Member Dashboard — Tasks, Leaves, Meetings & Feedback")
    name = st.text_input("Enter your Name")
    if name:
        df_all = fetch_all()
        if df_all.empty:
            st.info("No records found.")
        else:
            tabs_tm = st.tabs(["My Tasks", "My Leave", "My Meetings", "My Feedback"])

            # --- My Tasks ---
            with tabs_tm[0]:
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
                tabs_l = st.tabs(["Apply Leave", "My Leave History"])
                with tabs_l[0]:
                    leave_type = st.selectbox("Leave Type", ["Casual", "Sick", "Earned"])
                    f_date = st.date_input("From Date", min_value=date.today())
                    t_date = st.date_input("To Date", min_value=f_date)
                    reason = st.text_area("Reason")
                    if st.button("Submit Leave"):
                        lid = str(uuid.uuid4())
                        md = {"type": "Leave", "employee": name, "leave_type": leave_type,
                              "from": str(f_date), "to": str(t_date), "reason": reason,
                              "status": "Pending", "submitted": now()}
                        upsert_data(lid, md)
                        st.success("✅ Leave request submitted.")
                with tabs_l[1]:
                    my_leaves = df_all[(df_all["type"] == "Leave") &
                                       (df_all["employee"].str.lower() == name.lower())]
                    if my_leaves.empty:
                        st.info("No leave history.")
                    else:
                        st.dataframe(my_leaves[["leave_type", "from", "to", "reason", "status"]], use_container_width=True)
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

            # --- My Feedback + Sentiment ---
            with tabs_tm[3]:
                my_tasks = df_all[(df_all["type"] == "Task") & (df_all["employee"].str.lower() == name.lower())]
                if my_tasks.empty:
                    st.info("No feedback available.")
                else:
                    for _, task in my_tasks.iterrows():
                        st.markdown(f"### 🧩 Task: {task.get('task', 'N/A')}")
                        st.write(f"**Completion:** {task.get('completion', '0')}%")
                        st.write(f"**Marks:** {task.get('marks', '0')}")
                        st.write(f"**Manager Feedback:** {task.get('manager_feedback', 'N/A')}")
                        fb_text = task.get("manager_feedback", "")
                        if fb_text.strip():
                            vec = vectorizer.transform([fb_text])
                            sentiment = svm_clf.predict(vec)[0]
                            st.markdown(f"**Sentiment Analysis:** {'😊 Positive' if sentiment==1 else '😞 Negative'}")
                        st.divider()
