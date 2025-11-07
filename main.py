# ai_enterprise_secure.py
"""
Enterprise-grade Streamlit app (single file) for:
- Multi-company / multi-project / multi-department / multi-team task & leave management
- Manager, Team Member, Department Head, Client roles
- Pinecone backend (new SDK) with index 'task' and dimension 1024
- Leave module, reassignment, comments, file upload stubs
- Simple ML helpers (marks, on-track, risk, sentiment)
- Secure secrets usage via st.secrets (no credentials in code)
- Notifications via Email/Twilio (only if secrets provided)
IMPORTANT: Put your credentials in .streamlit/secrets.toml and rotate any keys you leaked.
"""

import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import plotly.express as px
import uuid
import base64
import os
import io
import traceback

# -----------------------
# Page config & constants
# -----------------------
st.set_page_config(page_title="AI Enterprise Workforce System (Secure)", layout="wide")
st.title("🏢 AI Enterprise Workforce & Task Management — Secure")

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

INDEX_NAME = "task"
DIMENSION = 1024

# -----------------------
# Load secrets safely
# -----------------------
def secret_ok(path_list):
    """Check nested st.secrets keys presence."""
    cur = st.secrets
    try:
        for k in path_list:
            cur = cur[k] if isinstance(cur, dict) else cur.get(k)
        return cur is not None and cur != ""
    except Exception:
        return False

# Read Pinecone key (top-level key expected)
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
# Optional grouped secrets for email and twilio (example TOML uses [email], [twilio])
EMAIL = st.secrets.get("email", {}) if isinstance(st.secrets.get("email", {}), dict) else {}
TWILIO = st.secrets.get("twilio", {}) if isinstance(st.secrets.get("twilio", {}), dict) else {}

# If secrets were accidentally pasted into chat — warn user (do not display secret values)
if not PINECONE_API_KEY:
    st.warning("Pinecone API key not found in Streamlit secrets. Pinecone features will be disabled until you add the key to .streamlit/secrets.toml.")
    st.info("If you accidentally shared credentials publicly, rotate them immediately (Pinecone, Google app password, Twilio).")

# -----------------------
# Pinecone init (safe)
# -----------------------
@st.cache_resource
def init_pinecone_client(api_key: str):
    if not api_key:
        return None
    try:
        pc = Pinecone(api_key=api_key)
        # create index if missing
        names = {i["name"] for i in pc.list_indexes()}
        if INDEX_NAME not in names:
            pc.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        idx = pc.Index(INDEX_NAME)
        return idx
    except Exception as e:
        st.error(f"Pinecone init failed: {e}")
        return None

index = init_pinecone_client(PINECONE_API_KEY)

# -----------------------
# Utility helpers
# -----------------------
def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def rand_vec():
    return np.random.rand(DIMENSION).tolist()

def safe_upsert(md: dict):
    """Upsert metadata record into Pinecone; uses placeholder random vector if no embedding pipeline."""
    if index is None:
        st.warning("Pinecone index not available — record not persisted.")
        return False
    try:
        index.upsert([{
            "id": str(md.get("_id", uuid.uuid4())),
            "values": rand_vec(),
            "metadata": md
        }])
        return True
    except Exception as e:
        st.error(f"Pinecone upsert error: {e}")
        return False

def safe_query(filter_dict=None, top_k=1000):
    """Query Pinecone index and return DataFrame of metadata items."""
    if index is None:
        return pd.DataFrame()
    try:
        res = index.query(vector=rand_vec(), top_k=top_k, include_metadata=True, filter=filter_dict or {})
        rows = []
        for m in (res.matches or []):
            md = m.metadata or {}
            md["_id"] = m.id
            rows.append(md)
        return pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"Pinecone query failed: {e}")
        return pd.DataFrame()

def download_df(df: pd.DataFrame, name="export.csv"):
    """Return download button for dataframe."""
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, file_name=name)

# -----------------------
# Notification helpers
# -----------------------
def send_email(to, subject, body):
    """Send email using SMTP if email secrets provided."""
    if not EMAIL or not EMAIL.get("SENDER_EMAIL") or not EMAIL.get("APP_PASSWORD"):
        st.info("Email not configured (add email credentials to st.secrets['email']).")
        return False
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        sender = EMAIL["SENDER_EMAIL"]
        password = EMAIL["APP_PASSWORD"]
        smtp_server = EMAIL.get("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(EMAIL.get("SMTP_PORT", 587))

        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = to
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
        return True
    except Exception as e:
        st.warning(f"Email send failed: {e}")
        return False

def send_sms(to, body):
    """Send SMS using Twilio if configured."""
    if not TWILIO or not TWILIO.get("ACCOUNT_SID") or not TWILIO.get("AUTH_TOKEN") or not TWILIO.get("PHONE_NUMBER"):
        st.info("Twilio not configured (add Twilio credentials to st.secrets['twilio']).")
        return False
    try:
        from twilio.rest import Client
        client = Client(TWILIO["ACCOUNT_SID"], TWILIO["AUTH_TOKEN"])
        client.messages.create(body=body, from_=TWILIO["PHONE_NUMBER"], to=to)
        return True
    except Exception as e:
        st.warning(f"SMS send failed: {e}")
        return False

# -----------------------
# Simple ML helpers (extend as needed)
# -----------------------
lin_reg = LinearRegression().fit([[0], [50], [100]], [0, 2.5, 5])
log_reg = LogisticRegression().fit([[0], [40], [80], [100]], [0, 0, 1, 1])
rf = RandomForestClassifier().fit(np.array([[10, 2], [50, 1], [90, 0], [100, 0]]), [0, 1, 0, 0])

# small text model (TF-IDF + SVM for demo)
tfidf = TfidfVectorizer(max_features=2000)
sample_texts = ["excellent work", "needs improvement", "bad performance", "great job", "average"]
tfidf.fit(sample_texts)
svm = SVC().fit(tfidf.transform(sample_texts), [1,0,0,1,0])

# -----------------------
# Role selection (can be forced to Team Member via secrets FORCE_TEAM_MEMBER)
# -----------------------
FORCE_TEAM_MEMBER = str(st.secrets.get("FORCE_TEAM_MEMBER", "false")).lower() == "true"
if FORCE_TEAM_MEMBER:
    role = "Team Member"
    st.caption("Auto-loaded Team Member view (login bypass enabled in secrets).")
else:
    role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Department Head", "Client"])

current_month = datetime.now().strftime("%B %Y")

# -----------------------
# Helper: file uploader storage (in-memory / temp)
# -----------------------
def handle_file_upload(uploaded_file):
    """Return metadata entry for uploaded file (for demo: save to local /tmp and return link)."""
    if uploaded_file is None:
        return None
    try:
        folder = "uploads"
        os.makedirs(folder, exist_ok=True)
        fname = f"{uuid.uuid4()}_{uploaded_file.name}"
        path = os.path.join(folder, fname)
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return {"filename": uploaded_file.name, "path": path, "uploaded_on": now()}
    except Exception as e:
        st.warning(f"File save failed: {e}")
        return None

# -----------------------
# Manager UI
# -----------------------
if role == "Manager":
    st.header("👨‍💼 Manager Dashboard — Enterprise")
    tab_assign, tab_review, tab_leave, tab_analytics = st.tabs(["Assign / Reassign", "Review Tasks", "Leave Requests", "Analytics & Leaderboard"])

    # Assign / Reassign
    with tab_assign:
        st.subheader("➕ Assign or Reassign Task")
        with st.form("assign_form"):
            company = st.text_input("Company")
            project = st.text_input("Project")
            department = st.selectbox("Department", ["IT", "HR", "Finance", "Marketing", "Operations"])
            team = st.text_input("Team (e.g., Team Alpha)")
            employee = st.text_input("Employee")
            title = st.text_input("Task Title")
            description = st.text_area("Description")
            deadline = st.date_input("Deadline", value=date.today() + timedelta(days=7))
            files = st.file_uploader("Attach Proof / Files (optional)", accept_multiple_files=True)
            submit = st.form_submit_button("Assign Task")

            if submit:
                file_entries = []
                for f in files:
                    meta = handle_file_upload(f)
                    if meta:
                        file_entries.append(meta)
                md = {
                    "_id": str(uuid.uuid4()),
                    "company": company, "project": project, "department": department,
                    "team": team, "employee": employee, "task": title, "description": description,
                    "deadline": deadline.isoformat(), "completion": 0, "marks": 0,
                    "status": "Assigned", "files": file_entries, "assigned_on": now()
                }
                ok = safe_upsert(md)
                if ok:
                    st.success(f"Task '{title}' assigned to {employee}.")
                    # optional notifications
                    if st.checkbox("Notify employee by email/sms?"):
                        if employee:
                            body = f"You have a new task: {title}. Deadline: {deadline}."
                            if EMAIL.get("SENDER_EMAIL"): send_email(st.text_input("Employee email to notify", value=""), f"New Task: {title}", body)
                            if TWILIO.get("PHONE_NUMBER"): send_sms(st.text_input("Employee phone to notify", value=""), body)

    # Review Tasks
    with tab_review:
        st.subheader("✅ Review & Finalize Tasks")
        filter_company = st.text_input("Filter by Company (optional)")
        filter_department = st.selectbox("Filter by Department (optional)", ["", "IT", "HR", "Finance", "Marketing", "Operations"])
        # Build filter for Pinecone
        fdict = {}
        if filter_company: fdict["company"] = {"$eq": filter_company}
        if filter_department: fdict["department"] = {"$eq": filter_department}
        df_tasks = safe_query(filter_dict=fdict)
        if df_tasks:
            df = pd.DataFrame(df_tasks)
            st.dataframe(df[["company","project","department","team","employee","task","completion","marks","status"]].head(50))
            # Iterate for granular review
            for idx, row in df.iterrows():
                st.markdown(f"### {row.get('task')} — {row.get('employee')} ({row.get('team')})")
                adj = st.slider(f"Adjust Completion (%)", 0, 100, int(row.get("completion", 0)), key=f"adj_{idx}")
                marks_pred = float(lin_reg.predict([[adj]])[0])
                comments = st.text_area("Manager Comments", key=f"mgrc_{idx}")
                approve = st.radio("Approve Task?", ["Yes","No"], index=0, key=f"apv_{idx}")
                if st.button(f"Finalize Review: {row.get('task')}", key=f"fin_{idx}"):
                    sentiment_val = int(svm.predict(tfidf.transform([comments]))[0]) if comments else None
                    sentiment = "Positive" if sentiment_val == 1 else ("Negative" if sentiment_val == 0 else "Neutral")
                    new_md = dict(row)
                    new_md.update({
                        "completion": adj, "marks": marks_pred, "manager_comments": comments,
                        "sentiment": sentiment, "approved": approve == "Yes", "reviewed_on": now()
                    })
                    safe_upsert(new_md)
                    st.success(f"Saved review for {row.get('task')} — Sentiment: {sentiment}")
        else:
            st.info("No tasks found (or Pinecone not connected).")

    # Leave requests
    with tab_leave:
        st.subheader("🏖 Leave Requests")
        leaves = safe_query(filter_dict={"status": {"$eq": "Leave Applied"}})
        if leaves:
            for i, lv in enumerate(leaves):
                st.markdown(f"**{lv.get('employee')}** — {lv.get('leave_type')} from {lv.get('from_date')} to {lv.get('to_date')}")
                decision = st.radio("Decision", ["Approve", "Reject"], key=f"lv_{i}")
                if st.button(f"Submit Leave Decision ({lv.get('employee')})", key=f"dec_{i}"):
                    lv["status"] = "Leave Approved" if decision == "Approve" else "Leave Rejected"
                    lv["decision_on"] = now()
                    safe_upsert(lv)
                    st.success(f"Leave {decision}d for {lv.get('employee')}")
        else:
            st.info("No pending leave requests.")

    # Analytics & Leaderboard
    with tab_analytics:
        st.subheader("📊 Analytics & Leaderboard")
        df = safe_query()
        if df:
            df = pd.DataFrame(df)
            numeric_cols = []
            if "marks" in df.columns:
                df["marks"] = pd.to_numeric(df["marks"], errors="coerce")
                numeric_cols.append("marks")
            if "completion" in df.columns:
                df["completion"] = pd.to_numeric(df["completion"], errors="coerce")
                numeric_cols.append("completion")
            if numeric_cols:
                st.metric("Avg Marks", f"{df['marks'].mean():.2f}" if "marks" in df.columns else "N/A")
                st.metric("Avg Completion", f"{df['completion'].mean():.2f}" if "completion" in df.columns else "N/A")
            # Leaderboard
            if "marks" in df.columns:
                top = df.sort_values(by="marks", ascending=False).head(10)
                fig = px.bar(top, x="employee", y="marks", color="department", title="Top Performers (by marks)")
                st.plotly_chart(fig, use_container_width=True)
            # Export
            if st.button("Export All Metadata (CSV)"):
                download_df(df, "all_tasks_export.csv")
        else:
            st.info("No data to analyze.")

# -----------------------
# Team Member UI
# -----------------------
elif role == "Team Member":
    st.header("👷 Team Member Portal — Simple & Fast")
    st.info("Load your tasks, update completion, apply for leave, and upload files as evidence.")
    t_company = st.text_input("Company")
    t_employee = st.text_input("Your Name")
    if st.button("Load My Tasks"):
        fdict = {"company": {"$eq": t_company}, "employee": {"$eq": t_employee}}
        tasks = safe_query(filter_dict=fdict)
        st.session_state["my_tasks"] = tasks
        st.success(f"Loaded {len(tasks)} tasks." if tasks is not None else "0 tasks loaded.")

    tasks_list = st.session_state.get("my_tasks", [])
    if tasks_list:
        for i, t in enumerate(tasks_list):
            st.subheader(f"{t.get('task')} — {t.get('project','')}")
            curr = int(t.get("completion", 0))
            new = st.slider("Update Completion (%)", 0, 100, curr, key=f"tm_{i}")
            files = st.file_uploader(f"Upload proof for {t.get('task')}", key=f"file_{i}")
            if st.button(f"Submit Update: {t.get('task')}", key=f"subtm_{i}"):
                # handle file
                fmeta = handle_file_upload(files) if files else []
                t["completion"] = new
                t["marks"] = float(lin_reg.predict([[new]])[0])
                t["status"] = "On Track" if log_reg.predict([[new]])[0] == 1 else "Delayed"
                t.setdefault("files", []).extend([fmeta] if fmeta else [])
                t["updated_on"] = now()
                safe_upsert(t)
                st.success(f"Updated task '{t.get('task')}' — {t.get('status')}")
                st.experimental_rerun()

    st.markdown("---")
    st.subheader("🏖 Apply for Leave")
    leave_employee = st.text_input("Employee Name (for leave)")
    leave_type = st.selectbox("Leave Type", ["Casual", "Sick", "Paid", "Unpaid"])
    from_date = st.date_input("From Date", value=date.today())
    to_date = st.date_input("To Date", value=date.today()+timedelta(days=1))
    reason = st.text_area("Reason")
    if st.button("Apply Leave"):
        md = {
            "_id": str(uuid.uuid4()), "employee": leave_employee, "leave_type": leave_type,
            "from_date": from_date.isoformat(), "to_date": to_date.isoformat(),
            "reason": reason, "status": "Leave Applied", "applied_on": now()
        }
        safe_upsert(md)
        st.success("Leave applied successfully.")

    st.markdown("---")
    st.subheader("📈 My Performance Summary")
    if st.button("Load My Performance"):
        df = safe_query(filter_dict={"employee": {"$eq": t_employee}})
        if df:
            df = pd.DataFrame(df)
            if "marks" in df.columns:
                df["marks"] = pd.to_numeric(df["marks"], errors="coerce")
                fig = px.line(df, x="task", y="marks", title="Marks over Tasks")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df[["task","marks","completion","status"]])
            else:
                st.info("No marks data found for you.")
        else:
            st.info("No tasks found.")

# -----------------------
# Department Head UI
# -----------------------
elif role == "Department Head":
    st.header("🏢 Department Head Panel")
    dept = st.selectbox("Department", ["IT","HR","Finance","Marketing","Operations"])
    if st.button("Load Department Data"):
        df = safe_query(filter_dict={"department": {"$eq": dept}})
        if df:
            df = pd.DataFrame(df)
            if "marks" in df.columns:
                df["marks"] = pd.to_numeric(df["marks"], errors="coerce")
            st.metric("Avg Marks", f"{df['marks'].mean():.2f}" if "marks" in df.columns else "N/A")
            fig = px.box(df, x="team", y="marks", color="team", title=f"{dept} team marks distribution")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df[["employee","task","marks","completion","status","team"]])
        else:
            st.info("No tasks for this department.")

# -----------------------
# Client UI
# -----------------------
elif role == "Client":
    st.header("🧾 Client Project Portal")
    st.info("Clients can view projects, approve completed tasks, rate and give feedback.")
    c_company = st.text_input("Company")
    c_project = st.text_input("Project")

    if st.button("Load Project Overview"):
        df = safe_query(filter_dict={"company": {"$eq": c_company}, "project": {"$eq": c_project}})
        if df:
            df = pd.DataFrame(df)
            st.metric("Avg Completion", f"{df['completion'].astype(float).mean():.2f}" if "completion" in df.columns else "N/A")
            st.dataframe(df[["task","employee","completion","marks","status"]])
            # Approve completed
            completed = df[df["completion"].astype(float) >= 100] if "completion" in df.columns else pd.DataFrame()
            if not completed.empty:
                st.subheader("Approve Completed Tasks")
                for i, row in completed.iterrows():
                    st.markdown(f"**{row['task']}** by {row['employee']}")
                    comment = st.text_area("Feedback", key=f"cmt_{i}")
                    rating = st.slider("Rating (1-5)", 1, 5, 4, key=f"rate_{i}")
                    if st.button(f"Submit Client Review for {row['task']}", key=f"client_rev_{i}"):
                        row["client_comment"] = comment
                        row["client_rating"] = rating
                        row["client_approved"] = True
                        row["client_reviewed_on"] = now()
                        safe_upsert(dict(row))
                        st.success(f"Saved client review for {row['task']}")
            else:
                st.info("No fully completed tasks yet.")
        else:
            st.info("No project data found (or Pinecone disconnected).")

# -----------------------
# Footer + security reminder
# -----------------------
st.markdown("---")
st.caption("Security reminder: keep credentials in .streamlit/secrets.toml and rotate keys if they were exposed.")
if st.button("Show security checklist"):
    st.info("""
    1) Do NOT paste API keys or passwords in chats or public code.  
    2) Store secrets in .streamlit/secrets.toml or use platform secrets.  
    3) If a key was shared, rotate (revoke/create new) immediately.  
    4) Limit keys' permissions and use principle of least privilege.  
    5) Monitor usage and billing for unexpected activity.
    """)
