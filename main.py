# ============================================================
# main.py — Enterprise Workforce Intelligence System (Final Stable Build v2)
# (patched: safer text handling for vectorizer and string ops)
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
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# ============================================================
# Optional Pinecone setup
# ============================================================
try:
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
except Exception:
    PINECONE_AVAILABLE = False

st.set_page_config(page_title="Enterprise Workforce Intelligence System", layout="wide")
st.title("SmartWorkAI")
st.caption("AI-Driven Workforce Analytics • Productivity Intelligence • 360° Performance Overview")

# ============================================================
# Pinecone configuration
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
        st.warning(f"Pinecone connection failed — local mode active ({e})")
else:
    st.warning("⚠️ Pinecone key missing — using local session storage.")

# ============================================================
# Utility functions
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
        md = {"data": str(md)}
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
        return pd.DataFrame([{"_id": k, **v} for k, v in local.items()])
    try:
        res = index.query(vector=rand_vec(), top_k=1000, include_metadata=True)
        rows = []
        for m in res.get("matches", []):
            md = m.get("metadata", {})
            md["_id"] = m.get("id")
            rows.append(md)
        return pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"Fetch failed: {e}")
        return pd.DataFrame()

# ---------- Safe text helper ----------
def safe_text(x):
    """
    Return a safe string for vectorizer and string operations.
    If x is NaN/None, returns empty string.
    Otherwise returns str(x).
    """
    try:
        if x is None:
            return ""
        # pandas NA check
        if isinstance(x, (float,)) and pd.isna(x):
            return ""
        if pd.isna(x):
            return ""
    except Exception:
        pass
    # if it is a list/dict/ndarray, convert to json-ish string
    if isinstance(x, (list, dict, np.ndarray)):
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)
    return str(x)

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
role = st.sidebar.selectbox("Access As", ["Manager", "Team Member", "Client", "HR Administrator"])

# ============================================================
# MANAGER
# ============================================================
if role == "Manager":
    st.header("Manager Dashboard — Tasks, Feedback, Leave, Meetings, Overview")
    df_all = fetch_all()
    tabs = st.tabs(["Task Management", "Feedback", "Leave Management", "Meeting Management", "360° Overview"])

    # --- Task Management
    with tabs[0]:
        st.subheader("Assign Task")
        with st.form("assign_task"):
            emp = st.text_input("Employee Name")
            company = st.text_input("Company Name")
            dept = st.text_input("Department")
            task = st.text_input("Task Title")
            desc = st.text_area("Description")
            deadline = st.date_input("Deadline", value=date.today())
            if st.form_submit_button("Assign Task") and emp and task:
                tid = str(uuid.uuid4())
                md = {"type": "Task", "employee": emp, "company": company, "department": dept, "task": task,
                      "description": desc, "deadline": str(deadline),
                      "completion": 0, "marks": 0, "status": "Assigned", "created": now()}
                upsert_data(tid, md)
                st.success(f"Task '{task}' assigned to {emp}")

    # --- Feedback
    with tabs[1]:
        st.subheader("Manager Feedback & Evaluation")
        if df_all.empty:
            st.info("No data available.")
        else:
            search = st.text_input("Search by Employee or Task").strip().lower()
            if search:
                df_tasks = df_all[df_all.get("type", "") == "Task"] if "type" in df_all.columns else df_all.copy()
                # ensure columns exist and safe types
                df_tasks = df_all[df_all["type"] == "Task"].copy() if "type" in df_all.columns else pd.DataFrame()
                if "completion" in df_tasks.columns:
                    df_tasks["completion"] = pd.to_numeric(df_tasks["completion"], errors="coerce").fillna(0)
                # safe filtering
                matched = df_tasks[df_tasks["employee"].astype(str).str.lower().str.contains(search, na=False) |
                                   df_tasks["task"].astype(str).str.lower().str.contains(search, na=False)]
                for _, task in matched.iterrows():
                    emp = task.get("employee", "")
                    st.markdown(f"### {emp} — {task.get('task','')}")
                    marks = st.slider("Marks", 0.0, 5.0, float(task.get("marks", 0)), key=f"m_{task.get('_id','')}")
                    fb = st.text_area("Feedback", safe_text(task.get("manager_feedback", "")), key=f"f_{task.get('_id','')}")
                    if st.button(f"Save Feedback ({emp})", key=f"s_{task.get('_id','')}"):
                        task["marks"] = marks
                        task["manager_feedback"] = fb
                        task["status"] = "Under Client Review"
                        task["manager_reviewed_on"] = now()
                        upsert_data(task.get("_id", str(uuid.uuid4())), task)
                        st.success(f"Feedback saved for {emp}")

    # --- Leave Management
    with tabs[2]:
        st.subheader("Leave Approvals")
        df_leave = df_all[df_all["type"] == "Leave"] if "type" in df_all.columns else pd.DataFrame()
        if df_leave.empty:
            st.info("No leave data.")
        else:
            st.dataframe(df_leave[["employee", "leave_type", "from", "to", "reason", "status"]], use_container_width=True)
            search = st.text_input("Search by Employee").strip().lower()
            if search:
                pending = df_leave[df_leave["employee"].astype(str).str.lower().str.contains(search, na=False)].copy()
                pending = pending[pending["status"].astype(str).str.lower() == "pending"]
                for i, row in pending.iterrows():
                    st.markdown(f"**{row.get('employee','')}** → {row.get('from','')} to {row.get('to','')} — {row.get('reason','')}")
                    dec = st.radio(f"Decision for {row.get('employee','')}", ["Approve", "Reject"], key=f"d_{i}", horizontal=True)
                    if st.button(f"Submit ({row.get('employee','')})", key=f"b_{i}"):
                        row["status"] = "Approved" if dec == "Approve" else "Rejected"
                        row["approved_on"] = now()
                        upsert_data(row.get("_id", str(uuid.uuid4())), row)
                        st.success(f"Leave {row['status']} for {row.get('employee','')}")

    # --- Meeting
    with tabs[3]:
        with st.form("schedule_meeting"):
            title = st.text_input("Title")
            m_date = st.date_input("Date", date.today())
            m_time = st.text_input("Time", "10:00 AM")
            attendees = st.text_area("Attendees (comma-separated)")
            notes = st.text_area("Notes")
            if st.form_submit_button("Schedule"):
                mid = str(uuid.uuid4())
                md = {"type": "Meeting", "meeting_title": title, "meeting_date": str(m_date),
                      "meeting_time": m_time, "attendees": attendees, "notes": notes, "created": now()}
                upsert_data(mid, md)
                st.success("Meeting scheduled.")

    # --- Overview
    with tabs[4]:
        st.subheader("360° Overview")
        if not df_all.empty:
            tasks = df_all[df_all["type"] == "Task"] if "type" in df_all.columns else pd.DataFrame()
            leaves = df_all[df_all["type"] == "Leave"] if "type" in df_all.columns else pd.DataFrame()
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Tasks", len(tasks))
            c2.metric("Total Leaves", len(leaves))
            avg_completion = pd.to_numeric(tasks["completion"], errors="coerce").mean() if "completion" in tasks.columns else 0
            c3.metric("Avg Completion", f"{avg_completion:.1f}%")
            if not tasks.empty and "employee" in tasks.columns:
                fig = px.bar(tasks, x="employee", y="completion", color="status", title="Task Completion by Employee")
                st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TEAM MEMBER
# ============================================================
elif role == "Team Member":
    st.header("Team Member Dashboard")
    name = st.text_input("Enter Your Name")
    if name:
        df_all = fetch_all()
        # normalize name for comparison
        name_safe = name.strip().lower()
        tabs_tm = st.tabs(["My Tasks", "My Leave", "My Meetings", "My Feedback"])

        # --- Tasks
        with tabs_tm[0]:
            if df_all.empty or "type" not in df_all.columns or "employee" not in df_all.columns:
                st.info("No assigned tasks.")
            else:
                my = df_all[(df_all["type"] == "Task") & (df_all["employee"].astype(str).str.lower() == name_safe)].copy()
                if my.empty:
                    st.info("No assigned tasks.")
                else:
                    for _, t in my.iterrows():
                        st.markdown(f"**{t.get('task','')}**")
                        try:
                            current_completion = int(float(t.get("completion", 0)))
                        except Exception:
                            current_completion = 0
                        val = st.slider("Progress", 0, 100, current_completion, key=t.get("_id", str(uuid.uuid4())))
                        if st.button(f"Update {t.get('task','')}", key=f"u_{t.get('_id','')}"):
                            t["completion"] = val
                            # predict marks via linear regression
                            try:
                                t["marks"] = float(lin_reg.predict([[val]])[0])
                            except Exception:
                                t["marks"] = 0.0
                            t["status"] = "Completed" if val >= 100 else "In Progress"
                            upsert_data(t.get("_id", str(uuid.uuid4())), t)
                            st.success("Progress updated.")

        # --- Feedback (Manager + Client)
        with tabs_tm[3]:
            if df_all.empty or "type" not in df_all.columns or "employee" not in df_all.columns:
                st.info("No feedback yet.")
            else:
                df_tasks = df_all[(df_all["type"] == "Task") & (df_all["employee"].astype(str).str.lower() == name_safe)].copy()
                if df_tasks.empty:
                    st.info("No feedback yet.")
                else:
                    for _, t in df_tasks.iterrows():
                        st.markdown(f"### {t.get('task','')}")
                        mgr_fb = safe_text(t.get("manager_feedback", ""))
                        client_fb = safe_text(t.get("client_feedback", ""))
                        st.write(f"**Manager Feedback:** {mgr_fb if mgr_fb else 'N/A'}")
                        if mgr_fb:
                            try:
                                s_vec = vectorizer.transform([mgr_fb])
                                s = svm_clf.predict(s_vec)[0]
                                st.write(f"Sentiment (Manager): {'Positive' if s == 1 else 'Negative'}")
                            except Exception as e:
                                st.write("Sentiment (Manager): N/A")
                        st.write(f"**Client Feedback:** {client_fb if client_fb else 'N/A'}")
                        if client_fb:
                            try:
                                s_vec = vectorizer.transform([client_fb])
                                s = svm_clf.predict(s_vec)[0]
                                st.write(f"Sentiment (Client): {'Positive' if s == 1 else 'Negative'}")
                            except Exception:
                                st.write("Sentiment (Client): N/A")

        # --- Leave
        with tabs_tm[1]:
            leave_tabs = st.tabs(["Apply Leave", "My Leave History"])
            with leave_tabs[0]:
                l_type = st.selectbox("Leave Type", ["Casual", "Sick", "Earned"])
                f = st.date_input("From", min_value=date.today())
                t = st.date_input("To", min_value=f)
                r = st.text_area("Reason")
                if st.button("Submit Leave"):
                    lid = str(uuid.uuid4())
                    md = {"type": "Leave", "employee": name, "leave_type": l_type, "from": str(f),
                          "to": str(t), "reason": r, "status": "Pending", "submitted": now()}
                    upsert_data(lid, md)
                    st.success(" Leave Submitted.")

            with leave_tabs[1]:
                if df_all.empty or "type" not in df_all.columns or "employee" not in df_all.columns:
                    st.info("No leave history.")
                else:
                    df = df_all[(df_all["type"] == "Leave") & (df_all["employee"].astype(str).str.lower() == name_safe)].copy()
                    if df.empty:
                        st.info("No leave history.")
                    else:
                        st.dataframe(df[["leave_type", "from", "to", "reason", "status"]], use_container_width=True)

        # --- Meetings
        with tabs_tm[2]:
            if df_all.empty or "type" not in df_all.columns:
                st.info("No meetings.")
            else:
                dfm = df_all[df_all["type"] == "Meeting"].copy()
                mine = dfm[dfm["attendees"].astype(str).str.lower().str.contains(name_safe, na=False)]
                if mine.empty:
                    st.info("No meetings.")
                else:
                    st.dataframe(mine[["meeting_title", "meeting_date", "meeting_time", "attendees", "notes"]])

# ============================================================
# CLIENT
# ============================================================
elif role == "Client":
    st.header("Client Review Portal")
    df = fetch_all()
    company = st.text_input("Enter Company Name").strip().lower()
    if company:
        if df.empty or "type" not in df.columns or "company" not in df.columns:
            st.info("No tasks pending for review.")
        else:
            df_tasks = df[(df["type"] == "Task") & (df["company"].astype(str).str.lower() == company)].copy()
            pending = df_tasks[df_tasks["status"] == "Under Client Review"] if not df_tasks.empty else pd.DataFrame()
            if pending.empty:
                st.info("No tasks pending for review.")
            else:
                st.subheader("Review Completed Tasks")
                for _, t in pending.iterrows():
                    fb_key = f"cf_{t.get('_id','')}"
                    fb = st.text_area(f"Client Feedback for {t.get('employee','')} - {t.get('task','')}", key=fb_key)
                    rating = st.slider("Rating (1–5)", 1, 5, 3, key=f"cr_{t.get('_id','')}")
                    if st.button(f"Submit Review ({t.get('task','')})", key=f"sb_{t.get('_id','')}"):
                        t["client_feedback"] = safe_text(fb)
                        t["client_rating"] = rating
                        t["status"] = "Client Approved" if rating >= 3 else "Client Requires Changes"
                        t["client_reviewed_on"] = now()

                        # Similarity check between client and manager feedback
                        mfb = safe_text(t.get("manager_feedback", ""))
                        cfb = safe_text(fb)
                        if mfb and cfb:
                            try:
                                vecs = vectorizer.transform([mfb, cfb])
                                sim = cosine_similarity(vecs)[0][1]
                                t["feedback_similarity"] = round(float(sim), 2)
                            except Exception:
                                t["feedback_similarity"] = "N/A"
                        else:
                            t["feedback_similarity"] = "N/A"

                        upsert_data(t.get("_id", str(uuid.uuid4())), t)
                        st.success(f" Review saved for {t.get('task','')}")

# ============================================================
# HR ANALYTICS
# ============================================================
elif role == "HR Administrator":
    st.header(" HR Analytics — Performance Insights")
    df = fetch_all()
    if df.empty:
        st.info("No data available.")
    else:
        tasks = df[df["type"] == "Task"].copy() if "type" in df.columns else pd.DataFrame()
        if not tasks.empty:
            if "completion" in tasks.columns:
                tasks["completion"] = pd.to_numeric(tasks["completion"], errors="coerce").fillna(0)
            if "marks" in tasks.columns:
                tasks["marks"] = pd.to_numeric(tasks["marks"], errors="coerce").fillna(0)

        st.subheader("Employee Performance Statistics & Clusters")

        if len(tasks) > 1:
            try:
                kmeans = KMeans(n_clusters=min(3, len(tasks)), n_init=10)
                tasks["cluster"] = kmeans.fit_predict(tasks[["completion", "marks"]])
                fig = px.scatter(tasks, x="completion", y="marks", color="cluster",
                                 hover_data=["employee", "task"], title="Employee Performance Clusters")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Clustering failed: {e}")

        if not tasks.empty:
            by_emp = tasks.groupby("employee", dropna=True).agg({"completion": "mean", "marks": "mean"}).reset_index()
            st.dataframe(by_emp, use_container_width=True)
            fig2 = px.bar(by_emp, x="employee", y="completion", color="marks",
                          title="Average Completion & Marks by Employee")
            st.plotly_chart(fig2, use_container_width=True)
