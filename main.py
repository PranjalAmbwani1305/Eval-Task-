# app.py
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import pandas as pd
import uuid
from datetime import datetime, date
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import plotly.express as px

# Optional advanced NLP
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except Exception:
    TEXTBLOB_AVAILABLE = False

# -----------------------------
# CONFIG & INIT
# -----------------------------
st.set_page_config(page_title="AI Task System — Master Dashboard", layout="wide")
st.title("AI-Powered Task Management — MASTER")

# Pinecone config (expects secret)
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "task"
DIMENSION = 1024

# Create index if not exists
if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(INDEX_NAME)

# -----------------------------
# HELPERS
# -----------------------------
def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def stable_vec(seed_text):
    """Deterministic vector based on seed_text (so upserts for same task id stay near same spot)."""
    rng = np.random.default_rng(abs(hash(seed_text)) % (2**32))
    return rng.random(DIMENSION).tolist()

def safe_meta(md):
    clean = {}
    for k, v in md.items():
        if v is None:
            v = ""
        elif isinstance(v, (datetime, date)):
            v = v.isoformat()
        clean[k] = v
    return clean

def to_df_from_matches(matches):
    rows = []
    for m in matches:
        md = m.metadata or {}
        md["_id"] = m.id
        rows.append(md)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # normalize columns lower-case for consistent access
    df.columns = [c.lower() for c in df.columns]
    return df

def fetch_filtered(filter_dict=None, top_k=10000):
    """Fetch filtered results from Pinecone deterministically."""
    try:
        res = index.query(
            vector=np.zeros(DIMENSION).tolist(),
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict or {}
        )
        return to_df_from_matches(res.matches or [])
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def fetch_all(top_k=10000):
    """Admin-only: fetch all records without filters."""
    try:
        res = index.query(
            vector=np.zeros(DIMENSION).tolist(),
            top_k=top_k,
            include_metadata=True
        )
        return to_df_from_matches(res.matches or [])
    except Exception as e:
        st.error(f"Error fetching all data: {e}")
        return pd.DataFrame()

# -----------------------------
# SIMPLE AI MODELS (fallbacks)
# -----------------------------
lin_reg = LinearRegression()
lin_reg.fit([[0], [50], [100]], [0, 2.5, 5])

log_reg = LogisticRegression()
log_reg.fit([[0], [40], [80], [100]], [0, 0, 1, 1])

# Small dataset to train fallback SVM sentiment
comments = ["excellent work", "needs improvement", "bad performance", "great job", "average"]
sentiments = [1, 0, 0, 1, 0]
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(comments)
svm_clf = SVC()
svm_clf.fit(X_train, sentiments)

rf = RandomForestClassifier()
X_rf = np.array([[10, 2], [50, 1], [90, 0], [100, 0]])
y_rf = [0, 1, 0, 0]
rf.fit(X_rf, y_rf)

# -----------------------------
# SENTIMENT + FEEDBACK UTILS
# -----------------------------
def analyze_comment_advanced(text):
    """
    Returns dict: {
        polarity: float (-1..1),
        subjectivity: float (0..1),
        sentiment_label: "Excellent"/"Needs Improvement"/"Neutral",
        emoji: str,
        feedback_level: str,
        short_summary: str
    }
    Uses TextBlob if available; otherwise falls back to SVM toy model.
    """
    text = (text or "").strip()
    if not text:
        return {"polarity": 0.0, "subjectivity": 0.0,
                "sentiment_label": "Neutral", "emoji": "🟡",
                "feedback_level": "No comments", "short_summary": "No comment provided."}
    if TEXTBLOB_AVAILABLE:
        tb = TextBlob(text)
        polarity = tb.sentiment.polarity
        subjectivity = tb.sentiment.subjectivity
        if polarity > 0.4:
            label = "Excellent"
            emoji = "🟢"
            level = "Outstanding performance"
        elif polarity < -0.25:
            label = "Needs Improvement"
            emoji = "🔴"
            level = "Below expectations"
        elif polarity > 0.1:
            label = "Good"
            emoji = "🟢"
            level = "Good performance"
        else:
            label = "Neutral"
            emoji = "🟡"
            level = "Satisfactory but can improve"
        short = f"{label} — polarity={polarity:.2f}, subj={subjectivity:.2f}"
        return {"polarity": polarity, "subjectivity": subjectivity,
                "sentiment_label": label, "emoji": emoji,
                "feedback_level": level, "short_summary": short}
    else:
        # fallback: SVM classifier
        try:
            pred = int(svm_clf.predict(vectorizer.transform([text]))[0])
            if pred == 1:
                return {"polarity": 0.6, "subjectivity": 0.5,
                        "sentiment_label": "Excellent", "emoji": "🟢",
                        "feedback_level": "Positive feedback (fallback model)",
                        "short_summary": "Positive (fallback SVM)"}
            else:
                return {"polarity": -0.3, "subjectivity": 0.6,
                        "sentiment_label": "Needs Improvement", "emoji": "🔴",
                        "feedback_level": "Negative feedback (fallback model)",
                        "short_summary": "Negative (fallback SVM)"}
        except Exception:
            return {"polarity": 0.0, "subjectivity": 0.0,
                    "sentiment_label": "Neutral", "emoji": "🟡",
                    "feedback_level": "Unknown", "short_summary": "Could not analyze"}

# -----------------------------
# ROLE SELECTION
# -----------------------------
role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client", "Admin"])
current_month = datetime.now().strftime("%B %Y")

# Shared quick-stats box
def show_quick_stats(df):
    if df.empty:
        st.info("No tasks found for the current selection.")
        return
    avg_marks = pd.to_numeric(df.get("marks", 0)).mean()
    avg_completion = pd.to_numeric(df.get("completion", 0)).mean()
    counts = len(df)
    cols = st.columns(3)
    cols[0].metric("Tasks", counts)
    cols[1].metric("Avg Completion %", f"{avg_completion:.1f}%")
    cols[2].metric("Avg Marks", f"{avg_marks:.2f}")

# -----------------------------
# MANAGER (BOSS)
# -----------------------------
if role == "Manager":
    st.header("Manager (Boss) — Master Dashboard")
    tab1, tab2, tab3 = st.tabs(["Assign Task", "Boss Review & Adjustment", "Manager Actions & 360°"])

    # Assign Task
    with tab1:
        st.subheader("Assign New Task")
        with st.form("assign"):
            company = st.text_input("Company Name")
            employee = st.text_input("Employee Name")
            task = st.text_input("Task Title")
            desc = st.text_area("Description")
            deadline = st.date_input("Deadline", value=date.today())
            month = st.text_input("Month", value=current_month)
            submit = st.form_submit_button("Assign Task")
            if submit:
                if not (company and employee and task):
                    st.warning("Please fill company, employee and task.")
                else:
                    tid = str(uuid.uuid4())
                    md = safe_meta({
                        "company": company,
                        "employee": employee,
                        "task": task,
                        "description": desc,
                        "deadline": deadline.isoformat(),
                        "month": month,
                        "completion": 0,
                        "marks": 0,
                        "status": "Assigned",
                        "reviewed": False,
                        "client_reviewed": False,
                        "assigned_on": now(),
                        "escalated": False,
                        "approvals": {},  # e.g. {"deliverable": True}
                        "manager_actions": []  # list of dicts: {type, note, by, on}
                    })
                    index.upsert([{"id": tid, "values": stable_vec(task), "metadata": md}])
                    st.success(f"Task '{task}' assigned to {employee} for {company}.")

    # Boss Review & Adjustment
    with tab2:
        st.subheader("Boss Review & Adjustment")
        company_name = st.text_input("Filter by Company Name (optional)", "")
        if company_name:
            df = fetch_filtered({"company": {"$eq": company_name}})
        else:
            df = pd.DataFrame()

        show_quick_stats(df)

        if df.empty:
            st.info("No tasks to review for this company selection.")
        else:
            # Sentiment distribution if available
            if "manager_comments" in df.columns:
                try:
                    fig = px.histogram(df, x="manager_comments", title="Recent Manager Comments (sample)", nbins=10)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass

            for _, r in df.iterrows():
                st.markdown("----")
                st.markdown(f"### {r.get('task')}")
                left, right = st.columns([2,1])
                with left:
                    st.write(f"**Employee:** {r.get('employee', 'Unknown')}")
                    st.write(f"**Reported Completion:** {r.get('completion', 0)}%")
                    st.write(f"**Current Marks:** {float(r.get('marks',0)):.2f}")
                    st.write(f"**Deadline Risk:** {r.get('deadline_risk','N/A')}")
                    st.write(f"**Status:** {r.get('status','N/A')}")
                    st.write(f"**Escalated:** {r.get('escalated', False)}")
                    st.write(f"**Assigned On:** {r.get('assigned_on','')}")
                    st.write(f"**Reviewed:** {r.get('reviewed', False)}")
                    st.write(f"**Manager Actions:**")
                    for act in r.get("manager_actions", []) or []:
                        st.write(f"- {act.get('type')} by {act.get('by')} on {act.get('on')}: {act.get('note')}")
                with right:
                    # Adjustment controls
                    adjusted_completion = st.slider(
                        f"Adjust Completion % for {r.get('task')}",
                        0, 100, int(r.get("completion", 0)), key=f"adj_{r.get('_id')}"
                    )
                    adjusted_marks = float(lin_reg.predict([[adjusted_completion]])[0])
                    comments = st.text_area(f"Boss Comments for {r.get('task')}", key=f"boss_cmt_{r.get('_id')}")
                    approve = st.selectbox(f"Approve Task {r.get('task')}?", ["No", "Yes"], key=f"boss_app_{r.get('_id')}")
                    # Quick actions
                    st.markdown("**Quick Actions**")
                    reassign_to = st.text_input(f"Reassign to (employee) for {r.get('task')}", key=f"reassign_{r.get('_id')}")
                    escalate = st.checkbox("Escalate this task", key=f"esc_{r.get('_id')}")
                    approve_deliverable = st.checkbox("Approve deliverable", key=f"appd_{r.get('_id')}")
                    send_note_type = st.selectbox("Send Note Type", ["None","Appreciation","Warning","Suggestion"], key=f"note_type_{r.get('_id')}")
                    note_text = st.text_area("Note text (optional)", key=f"note_text_{r.get('_id')}", height=80)
                    if st.button(f"Finalize Review for {r.get('task')}", key=f"final_{r.get('_id')}"):
                        # sentiment analysis & save
                        sentiment_info = analyze_comment_advanced(comments)
                        sentiment_label = sentiment_info["sentiment_label"]
                        # update metadata
                        md = dict(r)
                        md.update({
                            "completion": int(adjusted_completion),
                            "marks": float(adjusted_marks),
                            "manager_comments": comments,
                            "reviewed": True,
                            "sentiment": sentiment_label,
                            "sentiment_pol": float(sentiment_info.get("polarity",0.0)),
                            "approved_by_boss": (approve == "Yes"),
                            "reviewed_on": now()
                        })
                        # handle reassign
                        if reassign_to:
                            old_emp = md.get("employee")
                            md["employee"] = reassign_to
                            md.setdefault("manager_actions", []).append({
                                "type": "Reassigned",
                                "note": f"Reassigned from {old_emp} to {reassign_to}",
                                "by": "Manager",
                                "on": now()
                            })
                        # escalate
                        if escalate:
                            md["escalated"] = True
                            md.setdefault("manager_actions", []).append({
                                "type": "Escalated",
                                "note": note_text or "Escalated by manager",
                                "by": "Manager",
                                "on": now()
                            })
                        # approval flags
                        appr = md.get("approvals", {})
                        if approve_deliverable:
                            appr["deliverable"] = True
                            md["approvals"] = appr
                            md.setdefault("manager_actions", []).append({
                                "type": "Deliverable Approved",
                                "note": f"Deliverable approved",
                                "by": "Manager",
                                "on": now()
                            })
                        # send note
                        if send_note_type and send_note_type != "None":
                            md.setdefault("manager_actions", []).append({
                                "type": send_note_type,
                                "note": note_text or send_note_type,
                                "by": "Manager",
                                "on": now()
                            })
                        # upsert
                        index.upsert([{"id": md["_id"], "values": stable_vec(md.get("task","")), "metadata": safe_meta(md)}])
                        # nice feedback
                        if sentiment_info["sentiment_label"] in ["Excellent","Good"]:
                            st.success(f"✅ Review finalized for **{md.get('task')}** — {sentiment_info['emoji']} {sentiment_info['feedback_level']}.")
                        elif sentiment_info["sentiment_label"] == "Neutral":
                            st.info(f"ℹ️ Review finalized for **{md.get('task')}** — Neutral tone.")
                        else:
                            st.warning(f"⚠️ Review finalized for **{md.get('task')}** — {sentiment_info['emoji']} {sentiment_info['feedback_level']}.")
                        st.experimental_rerun()

    # Manager Actions & 360° Page
    with tab3:
        st.subheader("Quick Managerial Actions")
        st.write("Quick search by task ID or employee to take actions.")
        task_id = st.text_input("Task ID (optional)")
        emp_search = st.text_input("Employee name (optional)")
        if st.button("Load for Action"):
            if task_id:
                df_load = fetch_filtered({"_id": {"$eq": task_id}})
            elif emp_search:
                df_load = fetch_filtered({"employee": {"$eq": emp_search}})
            else:
                st.warning("Provide task id or employee to load.")
                df_load = pd.DataFrame()
            if df_load.empty:
                st.info("No matching tasks found.")
            else:
                st.session_state["action_tasks"] = df_load.to_dict("records")
                st.success(f"Loaded {len(df_load)} tasks for action.")
        for r in st.session_state.get("action_tasks", []):
            st.markdown("----")
            st.markdown(f"### {r.get('task')} (ID: {r.get('_id')})")
            st.write(f"Employee: {r.get('employee')}, Company: {r.get('company')}")
            col1, col2, col3 = st.columns(3)
            with col1:
                new_emp = st.text_input(f"Reassign {r.get('_id')}", key=f"reassign2_{r.get('_id')}")
                if st.button(f"Reassign Now {r.get('_id')}", key=f"reassign_btn_{r.get('_id')}"):
                    md = dict(r)
                    old = md.get("employee")
                    md["employee"] = new_emp
                    md.setdefault("manager_actions", []).append({"type":"Reassigned","note":f"From {old} to {new_emp}","by":"Manager","on":now()})
                    index.upsert([{"id": md["_id"], "values": stable_vec(md.get("task","")), "metadata": safe_meta(md)}])
                    st.success(f"Reassigned {md.get('task')} -> {new_emp}")
                    st.experimental_rerun()
            with col2:
                if st.button(f"Escalate {r.get('_id')}", key=f"escalate2_{r.get('_id')}"):
                    md = dict(r)
                    md["escalated"] = True
                    md.setdefault("manager_actions", []).append({"type":"Escalated","note":"Escalated by Manager","by":"Manager","on":now()})
                    index.upsert([{"id": md["_id"], "values": stable_vec(md.get("task","")), "metadata": safe_meta(md)}])
                    st.success("Escalated.")
                    st.experimental_rerun()
            with col3:
                note_type = st.selectbox("Note type", ["Appreciation","Warning","Suggestion"], key=f"note3_{r.get('_id')}")
                note_text = st.text_input("Note text", key=f"note3text_{r.get('_id')}")
                if st.button(f"Send Note {r.get('_id')}", key=f"note3btn_{r.get('_id')}"):
                    md = dict(r)
                    md.setdefault("manager_actions", []).append({"type":note_type,"note":note_text or note_type,"by":"Manager","on":now()})
                    index.upsert([{"id": md["_id"], "values": stable_vec(md.get("task","")), "metadata": safe_meta(md)}])
                    st.success("Note sent and recorded.")
                    st.experimental_rerun()

        st.markdown("----")
        st.subheader("Generate 360° Performance Summaries")
        month = st.text_input("Month/Period (e.g., 'October 2025')", value=current_month)
        company_for_360 = st.text_input("Company for 360° (optional)")
        if st.button("Generate 360° Summaries"):
            # fetch tasks (filter by company if provided)
            if company_for_360:
                df_all = fetch_filtered({"company": {"$eq": company_for_360}})
            else:
                df_all = fetch_all()
            if df_all.empty:
                st.info("No data found for 360° generation.")
            else:
                # compute metrics per employee
                df_all["marks"] = pd.to_numeric(df_all.get("marks",0), errors="coerce")
                df_all["completion"] = pd.to_numeric(df_all.get("completion",0), errors="coerce")
                summaries = []
                for emp, group in df_all.groupby("employee"):
                    avg_marks = float(group["marks"].mean() or 0)
                    avg_comp = float(group["completion"].mean() or 0)
                    completed = len(group[group["completion"]>=90])
                    reviewed = len(group[group.get("reviewed","") == True])
                    pos_count = len(group[group.get("sentiment","") == "Excellent"])
                    neg_count = len(group[group.get("sentiment","") == "Needs Improvement"])
                    # heuristic textual summary
                    tone = "Balanced"
                    if pos_count > neg_count + 1:
                        tone = "Strong positive feedback"
                    elif neg_count > pos_count + 1:
                        tone = "Concerns flagged"
                    text = (f"{emp} — Avg Marks: {avg_marks:.2f}, Avg Completion: {avg_comp:.1f}%. "
                            f"Completed High-Quality Tasks: {completed}. Reviewed items: {reviewed}. Feedback tone: {tone}.")
                    summaries.append({"employee": emp, "avg_marks": avg_marks, "avg_completion": avg_comp,
                                      "completed_high": completed, "reviewed_count": reviewed,
                                      "pos_feedback": pos_count, "neg_feedback": neg_count,
                                      "summary": text})
                s_df = pd.DataFrame(summaries).sort_values("avg_marks", ascending=False)
                st.dataframe(s_df)
                # Downloadable CSV
                st.download_button("Download 360° Summaries CSV", s_df.to_csv(index=False), "360_summaries.csv", "text/csv")

# -----------------------------
# TEAM MEMBER
# -----------------------------
elif role == "Team Member":
    st.header("Team Member — Progress & Submit")
    company = st.text_input("Company")
    employee = st.text_input("Your Name")

    if st.button("Load Tasks"):
        if not company or not employee:
            st.warning("Please enter both company and your name.")
        else:
            df = fetch_filtered({"company": {"$eq": company}, "employee": {"$eq": employee}})
            if df.empty:
                st.info("No tasks found for you.")
                st.session_state.pop("tasks", None)
            else:
                st.session_state["tasks"] = df.to_dict("records")
                st.success(f"Loaded {len(df)} tasks.")

    for r in st.session_state.get("tasks", []):
        st.markdown("----")
        st.subheader(r.get("task"))
        st.write(r.get("description"))
        st.write(f"Assigned on: {r.get('assigned_on','')}")
        curr = float(r.get("completion", 0) or 0)
        new = st.slider(f"Completion for {r.get('task')}", 0, 100, int(curr), key=f"tm_{r.get('_id')}")
        hours_over = st.number_input(f"Overtime hours for {r.get('task')}", min_value=0.0, value=float(r.get('overtime',0) or 0), step=0.5, key=f"ot_{r.get('_id')}")
        leave_flag = st.checkbox(f"Request leave related to {r.get('task')}", key=f"leave_{r.get('_id')}")
        if st.button(f"Submit Update {r.get('task')}", key=f"sub_{r.get('_id')}"):
            marks = float(lin_reg.predict([[new]])[0])
            track = "On Track" if log_reg.predict([[new]])[0] == 1 else "Delayed"
            miss = rf.predict([[new, 0]])[0]
            md2 = dict(r)
            md2.update({
                "completion": int(new),
                "marks": float(marks),
                "status": track,
                "deadline_risk": "High" if miss else "Low",
                "submitted_on": now(),
                "client_reviewed": False,
                "overtime": float(hours_over),
                "leave_requested": bool(leave_flag)
            })
            index.upsert([{"id": md2["_id"], "values": stable_vec(md2.get("task","")), "metadata": safe_meta(md2)}])
            st.success(f"Updated {md2.get('task')} ({track}, Risk={md2['deadline_risk']})")
            st.experimental_rerun()

# -----------------------------
# CLIENT
# -----------------------------
elif role == "Client":
    st.header("Client — Review Deliverables")
    company = st.text_input("Company Name")
    if st.button("Load Reviewed Tasks"):
        if not company:
            st.warning("Enter company name.")
        else:
            df = fetch_filtered({"company": {"$eq": company}, "reviewed": {"$eq": True}})
            if df.empty:
                st.info("No reviewed tasks yet.")
                st.session_state.pop("ctasks", None)
            else:
                st.session_state["ctasks"] = df.to_dict("records")
                st.success(f"Loaded {len(df)} reviewed tasks.")

    for r in st.session_state.get("ctasks", []):
        st.markdown("----")
        st.subheader(r.get("task"))
        st.write(f"Employee: {r.get('employee')}")
        st.write(f"Final Completion: {r.get('completion')}%")
        st.write(f"Manager Marks: {r.get('marks')}")
        client_comment = st.text_area(f"Feedback for {r.get('task')}", key=f"client_cmt_{r.get('_id')}")
        approve_type = st.selectbox("Approve type", ["Approve Deliverable","Approve with changes","Reject"], key=f"client_app_{r.get('_id')}")
        if st.button(f"Submit Client Decision for {r.get('task')}", key=f"client_sub_{r.get('_id')}"):
            md2 = dict(r)
            md2.update({
                "client_reviewed": True,
                "client_comments": client_comment,
                "client_approved_on": now(),
                "client_decision": approve_type
            })
            index.upsert([{"id": md2["_id"], "values": stable_vec(md2.get("task","")), "metadata": safe_meta(md2)}])
            st.success(f"Client decision submitted: {approve_type}")
            st.experimental_rerun()

# -----------------------------
# ADMIN
# -----------------------------
elif role == "Admin":
    st.header("Admin Dashboard — Full Access")
    df = fetch_all()
    if df.empty:
        st.warning("No data found.")
    else:
        st.subheader("Quick Metrics")
        show_quick_stats(df)
        st.subheader("Top Employees by Avg Marks")
        df["marks"] = pd.to_numeric(df.get("marks",0), errors="coerce")
        top = df.groupby("employee")["marks"].mean().reset_index().sort_values("marks", ascending=False).head(10)
        st.dataframe(top)

        st.subheader("K-Means Clustering (Performance)")
        if len(df) > 2:
            n_clusters = min(3, len(df))
            km = KMeans(n_clusters=n_clusters, n_init=10).fit(df[["completion","marks"]].fillna(0))
            df["cluster"] = km.labels_
            st.plotly_chart(px.scatter(df, x="completion", y="marks", color=df["cluster"].astype(str),
                                       hover_data=["employee","task"], title="Task Clusters"))
        else:
            st.info("Not enough data for clustering.")

        st.subheader("Sentiment Distribution (Manager Comments)")
        if "manager_comments" in df.columns:
            df["sentiment_label"] = df.get("sentiment","").fillna("Unknown")
            try:
                fig = px.pie(df, names="sentiment_label", title="Manager Comment Sentiment")
                st.plotly_chart(fig)
            except Exception:
                st.info("No sentiment chart available.")

        st.subheader("Data Table (All Tasks)")
        st.dataframe(df.sort_values("assigned_on", ascending=False))

        st.download_button("Download All Tasks (CSV)", df.to_csv(index=False), "tasks_full.csv", "text/csv")

# -----------------------------
# END
# -----------------------------
