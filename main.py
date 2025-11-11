# ============================================================
# main.py — Enterprise Workforce Intelligence System (Stable Final)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import json, uuid, time
from datetime import datetime, date
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
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
        existing = [i["name"] for i in pc.list_indexes()]
        if INDEX_NAME not in existing and existing:
            INDEX_NAME = existing[0]
        index = pc.Index(INDEX_NAME)
        st.success(f"✅ Connected to Pinecone index: {INDEX_NAME}")
    except Exception as e:
        st.warning(f"Pinecone connection failed — running in local mode. ({e})")
else:
    st.warning("⚠️ Pinecone key missing — using local session storage.")

# ============================================================
# Utility Functions
# ============================================================
def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def rand_vec():
    np.random.seed(int(time.time()) % 10000)
    return np.random.rand(DIMENSION).tolist()

def safe_meta(md):
    """Normalize metadata before upsert."""
    if isinstance(md, pd.Series):
        md = md.to_dict()
    elif not isinstance(md, dict):
        try:
            md = dict(md)
        except Exception:
            md = {"value": str(md)}
    clean = {}
    for k, v in md.items():
        try:
            if isinstance(v, (dict, list, np.ndarray)):
                clean[str(k)] = json.dumps(v, ensure_ascii=False)
            elif pd.isna(v):
                clean[str(k)] = ""
            else:
                if isinstance(v, (np.generic,)):
                    v = v.item()
                clean[str(k)] = str(v)
        except Exception:
            clean[str(k)] = str(v)
    return clean

def upsert_data(id_, md):
    """Safely upsert to Pinecone or local storage."""
    id_ = str(id_)
    md_clean = safe_meta(md)
    if not index:
        local = st.session_state.setdefault("LOCAL_DATA", {})
        local[id_] = md_clean
        return True
    try:
        index.upsert(vectors=[{"id": id_, "values": rand_vec(), "metadata": md_clean}])
        return True
    except Exception as e:
        st.error(f"Pinecone upsert failed: {e}")
        return False

def fetch_all():
    """Fetch all Pinecone data or from session."""
    if not index:
        local = st.session_state.get("LOCAL_DATA", {})
        if not local:
            return pd.DataFrame()
        rows = []
        for k, v in local.items():
            rec = dict(v)
            rec["_id"] = k
            rows.append(rec)
        return pd.DataFrame(rows)
    try:
        res = index.query(vector=rand_vec(), top_k=1000, include_metadata=True)
        matches = res.get("matches", [])
        rows = []
        for m in matches:
            md = m.get("metadata", {})
            md["_id"] = m.get("id")
            rows.append(md)
        return pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"Fetch failed: {e}")
        return pd.DataFrame()

# ============================================================
# Machine Learning models
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
role = st.sidebar.selectbox("Access As", ["Manager", "Team Member", "Client", "HR Administrator"])

# ============================================================
# MANAGER DASHBOARD
# ============================================================
if role == "Manager":
    st.header("👨‍💼 Manager Dashboard — Task, Leave, Feedback & Meetings")
    df_all = fetch_all()
    tabs = st.tabs(["Task Management", "Feedback", "Leave Management", "Meeting Management", "360° Overview"])

    # ---------------- Task Management ----------------
    with tabs[0]:
        st.subheader("Assign New Task")
        with st.form("assign_task"):
            emp = st.text_input("Employee Name")
            dept = st.text_input("Department")
            task = st.text_input("Task Title")
            desc = st.text_area("Description")
            deadline = st.date_input("Deadline", value=date.today())
            submit = st.form_submit_button("Assign Task")
            if submit:
                tid = str(uuid.uuid4())
                md = {
                    "type": "Task", "employee": emp, "department": dept, "task": task,
                    "description": desc, "deadline": str(deadline),
                    "completion": 0, "marks": 0, "status": "Assigned", "created": now()
                }
                upsert_data(tid, md)
                st.success(f"✅ Task '{task}' assigned to {emp}")

    # ---------------- Feedback ----------------
    with tabs[1]:
        st.subheader("Manager Feedback & Evaluation")
        df_all = fetch_all()
        if df_all.empty:
            st.info("No data available.")
        else:
            search = st.text_input("Search by Employee Name or Task Title").strip().lower()
            if not search:
                st.info("Enter a search term to see tasks.")
            else:
                df_tasks = df_all[df_all["type"].astype(str).str.lower() == "task"]
                df_tasks["completion"] = pd.to_numeric(df_tasks["completion"], errors="coerce").fillna(0)
                matched = df_tasks[
                    df_tasks["employee"].astype(str).str.lower().str.contains(search, na=False)
                    | df_tasks["task"].astype(str).str.lower().str.contains(search, na=False)
                ]
                completed = matched[matched["completion"] >= 75]
                if completed.empty:
                    st.warning("No matching completed tasks.")
                else:
                    for _, task in completed.iterrows():
                        emp = task["employee"]
                        st.markdown(f"### {emp} — {task['task']}")
                        marks = st.slider("Final Marks", 0.0, 5.0, float(task.get("marks", 0)), key=f"mark_{task['_id']}")
                        fb = st.text_area("Feedback", task.get("manager_feedback", ""), key=f"fb_{task['_id']}")
                        if st.button(f"Save Review for {emp}", key=f"btn_{task['_id']}"):
                            updated = dict(task)
                            updated["marks"] = marks
                            updated["manager_feedback"] = fb
                            updated["status"] = "Under Client Review"
                            updated["manager_reviewed_on"] = now()
                            ok = upsert_data(task["_id"], updated)
                            if ok:
                                st.success(f"✅ Feedback saved for {emp}")
                                time.sleep(0.5)
                                st.experimental_rerun()
                            else:
                                st.error("Save failed.")

    # ---------------- Leave Management ----------------
    with tabs[2]:
        st.subheader("Leave Approval Center")
        df_leave = df_all[df_all["type"].astype(str).str.lower() == "leave"]
        if df_leave.empty:
            st.info("No leave requests.")
        else:
            query = st.text_input("Search by Employee Name").strip().lower()
            if query:
                matched = df_leave[df_leave["employee"].astype(str).str.lower().str.contains(query, na=False)]
                pending = matched[matched["status"].astype(str).str.lower() == "pending"]
                if pending.empty:
                    st.success("✅ No pending leave requests.")
                else:
                    for i, row in pending.iterrows():
                        st.markdown(f"**{row['employee']}** → {row['from']} to {row['to']} — {row['reason']}")
                        dec = st.radio(f"Decision for {row['employee']}", ["Approve", "Reject"], key=f"dec_{i}", horizontal=True)
                        if st.button(f"Submit Decision for {row['employee']}", key=f"btn_{i}"):
                            row["status"] = "Approved" if dec == "Approve" else "Rejected"
                            row["approved_on"] = now()
                            ok = upsert_data(row["_id"], row)
                            if ok:
                                st.success(f"Leave {row['status']} for {row['employee']}")
                                time.sleep(0.5)
                                st.experimental_rerun()

    # ---------------- Meeting Management ----------------
    with tabs[3]:
        st.subheader("Meeting Scheduler")
        with st.form("schedule_meet"):
            title = st.text_input("Meeting Title")
            date_meet = st.date_input("Date", date.today())
            time_meet = st.text_input("Time", "10:00 AM")
            attendees = st.text_area("Attendees (comma-separated)")
            notes = st.text_area("Notes (optional)")
            submit = st.form_submit_button("Schedule")
            if submit:
                mid = str(uuid.uuid4())
                md = {
                    "type": "Meeting", "meeting_title": title, "meeting_date": str(date_meet),
                    "meeting_time": time_meet, "attendees": attendees, "notes": notes, "created": now()
                }
                upsert_data(mid, md)
                st.success("✅ Meeting scheduled.")

    # ---------------- 360° Overview ----------------
    with tabs[4]:
        st.subheader("360° Overview")
        df_all = fetch_all()
        if df_all.empty:
            st.info("No data available.")
        else:
            tasks = df_all[df_all["type"].astype(str).str.lower() == "task"]
            leaves = df_all[df_all["type"].astype(str).str.lower() == "leave"]
            meets = df_all[df_all["type"].astype(str).str.lower() == "meeting"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Tasks", len(tasks))
            c2.metric("Meetings", len(meets))
            c3.metric("Leaves", len(leaves))
            if not tasks.empty:
                tasks["completion"] = pd.to_numeric(tasks["completion"], errors="coerce").fillna(0)
                fig = px.bar(tasks, x="employee", y="completion", color="status", title="Task Completion")
                st.plotly_chart(fig, use_container_width=True)
            if not leaves.empty:
                fig2 = px.histogram(leaves, x="status", title="Leave Status")
                st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# TEAM MEMBER DASHBOARD
# ============================================================
elif role == "Team Member":
    st.header("🧑‍💻 Team Member Dashboard — Tasks, Leaves, Meetings & Feedback")
    name = st.text_input("Enter your name")
    if name:
        df_all = fetch_all()
        tabs_tm = st.tabs(["My Tasks", "My Leave", "My Meetings", "My Feedback"])

        # Tasks
        with tabs_tm[0]:
            my_tasks = df_all[(df_all["type"] == "Task") & (df_all["employee"].str.lower() == name.lower())]
            for _, r in my_tasks.iterrows():
                st.markdown(f"**{r['task']}** — _{r['status']}_")
                val = st.slider("Progress %", 0, 100, int(float(r["completion"])), key=r["_id"])
                if st.button(f"Update {r['task']}", key=f"upd_{r['_id']}"):
                    r["completion"] = val
                    r["marks"] = float(lin_reg.predict([[val]])[0])
                    r["status"] = "In Progress" if val < 100 else "Completed"
                    ok = upsert_data(r["_id"], r)
                    if ok:
                        st.success("✅ Progress updated.")
                        time.sleep(0.5)
                        st.experimental_rerun()

        # Leaves
        with tabs_tm[1]:
            leave_tabs = st.tabs(["Apply Leave", "My Leave History"])
            with leave_tabs[0]:
                leave_type = st.selectbox("Leave Type", ["Casual", "Sick", "Earned"])
                f_date = st.date_input("From Date", min_value=date.today())
                t_date = st.date_input("To Date", min_value=f_date)
                reason = st.text_area("Reason")
                if st.button("Submit Leave"):
                    lid = str(uuid.uuid4())
                    md = {
                        "type": "Leave", "employee": name, "leave_type": leave_type,
                        "from": str(f_date), "to": str(t_date), "reason": reason,
                        "status": "Pending", "submitted": now()
                    }
                    upsert_data(lid, md)
                    st.success("✅ Leave request submitted.")
                    time.sleep(0.5)
                    st.experimental_rerun()

            with leave_tabs[1]:
                my_leaves = df_all[(df_all["type"] == "Leave") & (df_all["employee"].str.lower() == name.lower())]
                if my_leaves.empty:
                    st.info("No leave history.")
                else:
                    st.dataframe(my_leaves[["leave_type", "from", "to", "reason", "status"]])

        # Meetings
        with tabs_tm[2]:
            meets = df_all[df_all["type"] == "Meeting"]
            my_meets = meets[meets["attendees"].astype(str).str.lower().str.contains(name.lower(), na=False)]
            if my_meets.empty:
                st.info("No meetings.")
            else:
                st.dataframe(my_meets[["meeting_title", "meeting_date", "meeting_time", "attendees", "notes"]])

        # Feedback
        with tabs_tm[3]:
            my_tasks = df_all[(df_all["type"] == "Task") & (df_all["employee"].str.lower() == name.lower())]
            if my_tasks.empty:
                st.info("No feedback.")
            else:
                for _, t in my_tasks.iterrows():
                    st.markdown(f"### {t['task']}")
                    st.write("Completion:", t["completion"])
                    st.write("Marks:", t["marks"])
                    st.write("Manager Feedback:", t.get("manager_feedback", "N/A"))
                    fb_text = t.get("manager_feedback", "")
                    if fb_text:
                        vec = vectorizer.transform([fb_text])
                        sentiment = svm_clf.predict(vec)[0]
                        st.markdown(f"Sentiment: {'😊 Positive' if sentiment==1 else '😞 Negative'}")

# ============================================================
# CLIENT DASHBOARD
# ============================================================
elif role == "Client":
    st.header("🏢 Client Review Portal")
    company = st.text_input("Enter company name")
    if company:
        df_all = fetch_all()
        df_tasks = df_all[df_all["type"] == "Task"]
        df_company = df_tasks[df_tasks["company"].astype(str).str.lower() == company.lower()]
        pending = df_company[df_company["status"] == "Under Client Review"]
        if pending.empty:
            st.info("No tasks pending client review.")
        else:
            for _, t in pending.iterrows():
                fb = st.text_area(f"Feedback for {t['task']}", key=f"cf_{t['_id']}")
                rating = st.slider("Rating", 1, 5, 3, key=f"cr_{t['_id']}")
                if st.button(f"Submit Review ({t['task']})", key=f"sb_{t['_id']}"):
                    t["client_feedback"] = fb
                    t["client_rating"] = rating
                    t["status"] = "Client Approved" if rating >= 3 else "Client Requires Changes"
                    t["client_reviewed_on"] = now()
                    ok = upsert_data(t["_id"], t)
                    if ok:
                        st.success("✅ Client feedback submitted.")
                        time.sleep(0.5)
                        st.experimental_rerun()

# ============================================================
# HR ADMIN DASHBOARD
# ============================================================
elif role == "HR Administrator":
    st.header("👩‍💼 HR Analytics — Performance & Leave Insights")
    df_all = fetch_all()
    if df_all.empty:
        st.info("No records found.")
    else:
        tabs_hr = st.tabs(["Employee Stats", "Clusters", "Leaves"])
        with tabs_hr[0]:
            tasks = df_all[df_all["type"] == "Task"]
            if not tasks.empty:
                by_emp = tasks.groupby("employee").agg({"completion":"mean","marks":"mean"}).reset_index()
                st.dataframe(by_emp)
                fig = px.bar(by_emp, x="employee", y="completion", title="Avg Completion by Employee")
                st.plotly_chart(fig)
        with tabs_hr[1]:
            tasks = df_all[df_all["type"] == "Task"]
            if len(tasks) >= 2:
                tasks["completion"] = pd.to_numeric(tasks["completion"], errors="coerce").fillna(0)
                tasks["marks"] = pd.to_numeric(tasks["marks"], errors="coerce").fillna(0)
                km = KMeans(n_clusters=min(3, len(tasks)), n_init=10)
                tasks["cluster"] = km.fit_predict(tasks[["completion","marks"]])
                fig2 = px.scatter(tasks, x="completion", y="marks", color="cluster", hover_data=["employee"])
                st.plotly_chart(fig2)
        with tabs_hr[2]:
            leaves = df_all[df_all["type"] == "Leave"]
            if not leaves.empty:
                leaves["status"] = leaves["status"].astype(str).str.title()
                fig3 = px.histogram(leaves, x="status", title="Leave Status Overview")
                st.plotly_chart(fig3)
