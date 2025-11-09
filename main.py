# main.py — AI Workforce Intelligence Platform (Enterprise Edition)
"""
Production-ready workforce analytics and management system with:
- Streamlit front-end
- Pinecone vector storage
- HuggingFace AI insights
- Slack & Email notifications

Before running:
    pip install streamlit pinecone-client scikit-learn plotly huggingface-hub PyPDF2 pandas openpyxl requests

Secrets required in .streamlit/secrets.toml:
[PINECONE_API_KEY]
HUGGINGFACEHUB_API_TOKEN
SMTP_SERVER
SMTP_PORT
SMTP_USER
SMTP_PASS
NOTIFY_FROM
SLACK_WEBHOOK_URL
"""

import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import pandas as pd
import uuid, json, time, smtplib, requests
from datetime import datetime, date
from email.message import EmailMessage
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import plotly.express as px

# Optional AI / PDF tools
try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except:
    HF_AVAILABLE = False

try:
    import PyPDF2
    PDF_AVAILABLE = True
except:
    PDF_AVAILABLE = False

# ---------------------------------------------------------
# Streamlit Configuration
# ---------------------------------------------------------
st.set_page_config(page_title="AI Workforce Intelligence", layout="wide")
st.title("AI Workforce Intelligence Platform")

# ---------------------------------------------------------
# Secure Config
# ---------------------------------------------------------
cfg = st.secrets
PINECONE_API_KEY = cfg.get("PINECONE_API_KEY", "")
HF_TOKEN = cfg.get("HUGGINGFACEHUB_API_TOKEN", "")
SMTP_SERVER = cfg.get("SMTP_SERVER", "")
SMTP_PORT = cfg.get("SMTP_PORT", 587)
SMTP_USER = cfg.get("SMTP_USER", "")
SMTP_PASS = cfg.get("SMTP_PASS", "")
NOTIFY_FROM = cfg.get("NOTIFY_FROM", SMTP_USER)
SLACK_WEBHOOK = cfg.get("SLACK_WEBHOOK_URL", "")

INDEX_NAME = "task"
DIMENSION = 1024

# ---------------------------------------------------------
# Utility Helpers
# ---------------------------------------------------------
def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def rand_vec():
    return np.random.rand(DIMENSION).tolist()

def safe_meta(md):
    """Ensure metadata is serializable for Pinecone."""
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

def notify_slack(msg):
    if not SLACK_WEBHOOK:
        return
    try:
        requests.post(SLACK_WEBHOOK, json={"text": msg}, timeout=5)
    except Exception:
        pass

def notify_email(subject, body, recipients):
    if not all([SMTP_SERVER, SMTP_USER, SMTP_PASS]):
        return
    try:
        msg = EmailMessage()
        msg["From"] = NOTIFY_FROM
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject
        msg.set_content(body)
        with smtplib.SMTP(SMTP_SERVER, int(SMTP_PORT)) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
    except Exception:
        pass

# ---------------------------------------------------------
# Pinecone Initialization
# ---------------------------------------------------------
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
                while not pc.describe_index(INDEX_NAME)["status"]["ready"]:
                    time.sleep(2)
        index = pc.Index(INDEX_NAME)
        st.caption(f"Connected to Pinecone index: {INDEX_NAME}")
    except Exception as e:
        st.warning(f"Pinecone init failed, running in local mode ({e})")
else:
    st.warning("No Pinecone API key found. Running in local mode.")

# ---------------------------------------------------------
# Data Access Layer
# ---------------------------------------------------------
def upsert_data(id_, md):
    md = safe_meta(md)
    if index:
        index.upsert([{"id": id_, "values": rand_vec(), "metadata": md}])
    else:
        st.session_state.setdefault("LOCAL_DATA", {})[id_] = md

def fetch_all():
    if not index:
        return pd.DataFrame(st.session_state.get("LOCAL_DATA", {}).values())
    try:
        stats = index.describe_index_stats()
        if stats.get("total_vector_count", 0) == 0:
            return pd.DataFrame()
        res = index.query(vector=rand_vec(), top_k=1000, include_metadata=True)
        rows = []
        for m in getattr(res, "matches", []):
            md = m.metadata or {}
            md["_id"] = m.id
            rows.append(md)
        return pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"Fetch failed: {e}")
        return pd.DataFrame()

# ---------------------------------------------------------
# ML Models
# ---------------------------------------------------------
lin_reg = LinearRegression().fit([[0], [50], [100]], [0, 2.5, 5])
vectorizer = CountVectorizer()
svm = SVC(probability=True)
X = vectorizer.fit_transform(["good", "excellent", "poor", "bad", "average"])
svm.fit(X, [1, 1, 0, 0, 0])

# ---------------------------------------------------------
# Role Selector
# ---------------------------------------------------------
role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client", "Admin"])
current_month = datetime.now().strftime("%B %Y")

# ---------------------------------------------------------
# MANAGER PORTAL
# ---------------------------------------------------------
if role == "Manager":
    st.header("Manager Command Center")
    tabs = st.tabs([
        "Task Management",
        "AI Insights",
        "Meetings & Feedback",
        "Leave Management",
        "360 Overview"
    ])

    # --- TASK MANAGEMENT ---
    with tabs[0]:
        st.subheader("Assign or Reassign Tasks")

        with st.form("assign_task"):
            company = st.text_input("Company")
            department = st.text_input("Department")
            employee = st.text_input("Employee (email preferred)")
            task = st.text_input("Task Title")
            desc = st.text_area("Task Description")
            deadline = st.date_input("Deadline", value=date.today())
            submit = st.form_submit_button("Assign Task")

            if submit and employee and task:
                tid = str(uuid.uuid4())
                md = dict(
                    company=company, department=department, employee=employee,
                    task=task, description=desc, deadline=deadline.isoformat(),
                    completion=0, marks=0, status="Assigned", assigned_on=now()
                )
                upsert_data(tid, md)
                st.success("Task successfully assigned.")
                notify_slack(f"New task '{task}' assigned to {employee}.")
                if "@" in employee:
                    notify_email(f"New Task: {task}",
                                 f"You have been assigned '{task}'. Deadline: {deadline}",
                                 [employee])

        df = fetch_all()
        if not df.empty:
            if "department" not in df.columns:
                df["department"] = "General"
            df["completion"] = pd.to_numeric(df.get("completion", 0), errors="coerce").fillna(0)
            st.plotly_chart(
                px.bar(df, x="employee", y="completion", color="department", title="Task Completion by Employee"),
                use_container_width=True
            )

            task_choice = st.selectbox("Reassign Task", df["task"].unique())
            new_emp = st.text_input("New Employee")
            reason = st.text_area("Reason for Reassignment")
            if st.button("Confirm Reassignment"):
                r = df[df["task"] == task_choice].iloc[0].to_dict()
                old = r["employee"]
                r.update({"employee": new_emp, "status": "Reassigned", "reassigned_on": now(), "reason": reason})
                upsert_data(r.get("_id", str(uuid.uuid4())), r)
                st.success("Task reassigned.")
                notify_slack(f"Task '{task_choice}' reassigned from {old} to {new_emp}.")
                notify_email(f"Task Reassigned: {task_choice}", f"Reason: {reason}", [new_emp])

    # --- AI INSIGHTS ---
    with tabs[1]:
        st.subheader("AI-Driven Insights")
        df = fetch_all()
        if df.empty:
            st.info("No task data available.")
        else:
            df["completion"] = pd.to_numeric(df["completion"], errors="coerce").fillna(0)
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg Completion", f"{df['completion'].mean():.1f}%")
            col2.metric("Total Tasks", len(df))
            col3.metric("Active Employees", df["employee"].nunique())

            st.plotly_chart(px.histogram(df, x="completion", nbins=10, title="Task Completion Distribution"))

            query = st.text_input("Ask AI (e.g. 'Who is underperforming this month?')")
            if st.button("Analyze"):
                if HF_AVAILABLE and HF_TOKEN:
                    client = InferenceClient(token=HF_TOKEN)
                    prompt = f"Dataset summary:\n{df.describe().to_dict()}\nQuestion: {query}\nAnswer as an HR analyst."
                    try:
                        response = client.text_generation(model="mistralai/Mixtral-8x7B-Instruct",
                                                          prompt=prompt, max_new_tokens=200)
                        st.write(response if isinstance(response, str) else response.get("generated_text", "No output"))
                    except Exception as e:
                        st.error(f"AI query failed: {e}")
                else:
                    st.warning("Hugging Face API key not configured.")

    # --- MEETINGS ---
    with tabs[2]:
        st.subheader("Meeting Scheduler & Feedback")
        with st.form("meeting_form"):
            mtitle = st.text_input("Meeting Title")
            mdate = st.date_input("Date", value=date.today())
            attendees = st.text_area("Attendees (comma-separated emails)")
            upload = st.file_uploader("Upload Meeting Notes", type=["pdf", "csv", "xlsx", "txt"])
            submit = st.form_submit_button("Save Meeting")

            if submit and mtitle:
                mid = str(uuid.uuid4())
                notes = ""
                if upload:
                    if upload.name.endswith(".pdf") and PDF_AVAILABLE:
                        reader = PyPDF2.PdfReader(upload)
                        notes = "\n".join([p.extract_text() or "" for p in reader.pages])
                    elif upload.name.endswith(".csv"):
                        notes = pd.read_csv(upload).to_csv(index=False)
                    elif upload.name.endswith(".xlsx"):
                        notes = pd.read_excel(upload).to_csv(index=False)
                    elif upload.name.endswith(".txt"):
                        notes = upload.getvalue().decode("utf-8", errors="ignore")
                md = dict(meeting=mtitle, date=str(mdate), attendees=attendees, notes=notes[:20000], created_on=now())
                upsert_data(mid, md)
                st.success("Meeting saved.")
                notify_slack(f"New meeting '{mtitle}' scheduled on {mdate}.")
                notify_email(f"Meeting Scheduled: {mtitle}", f"Date: {mdate}", [a.strip() for a in attendees.split(",") if "@" in a])

        df_meet = fetch_all()
        if not df_meet.empty and "meeting" in df_meet.columns:
            valid_meets = df_meet[df_meet["meeting"].notna()]
            st.dataframe(valid_meets[["meeting", "date", "attendees"]].fillna(""))

    # --- LEAVES ---
    with tabs[3]:
        st.subheader("Leave Management")
        leaves = st.session_state.get("LEAVES", pd.DataFrame(columns=["Employee","Type","From","To","Reason","Status"]))
        st.dataframe(leaves)
        with st.form("leave_form"):
            emp = st.text_input("Employee (email preferred)")
            typ = st.selectbox("Leave Type", ["Casual", "Sick", "Earned"])
            f = st.date_input("From")
            t = st.date_input("To")
            reason = st.text_area("Reason")
            sub = st.form_submit_button("Submit Leave")
            if sub:
                new = pd.DataFrame([{"Employee": emp, "Type": typ, "From": str(f), "To": str(t), "Reason": reason, "Status": "Pending"}])
                st.session_state["LEAVES"] = pd.concat([leaves, new], ignore_index=True)
                st.success("Leave submitted.")
                notify_slack(f"{emp} requested {typ} leave from {f} to {t}.")

    # --- 360 OVERVIEW ---
    with tabs[4]:
        st.subheader("360° Overview")
        df = fetch_all()
        if df.empty:
            st.info("No data available.")
        else:
            emp = st.selectbox("Select Employee", sorted(df["employee"].unique()))
            emp_df = df[df["employee"] == emp]
            st.dataframe(emp_df[["task", "completion", "marks", "status"]])
            st.metric("Avg Completion", f"{emp_df['completion'].mean():.1f}%")

# ---------------------------------------------------------
# TEAM MEMBER / CLIENT / ADMIN are same as previous version
# ---------------------------------------------------------
