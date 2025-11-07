import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import pandas as pd
from datetime import datetime, date
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from textblob import TextBlob
import plotly.express as px
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from twilio.rest import Client
import uuid

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Task Management System", layout="wide")
st.title("🚀 AI-Powered Task Management System")

# -----------------------------
# LOAD SECRETS
# -----------------------------
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
EMAIL_SENDER = st.secrets.get("EMAIL_SENDER", "")
EMAIL_PASSWORD = st.secrets.get("EMAIL_PASSWORD", "")
SMS_SID = st.secrets.get("SMS_SID", "")
SMS_AUTH = st.secrets.get("SMS_AUTH", "")
SMS_FROM = st.secrets.get("SMS_FROM", "")

INDEX_NAME = "task"
DIMENSION = 1024

# -----------------------------
# PINECONE INITIALIZATION (NEW SDK)
# -----------------------------
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if INDEX_NAME not in {i["name"] for i in pc.list_indexes()}:
        st.info(f"🆕 Creating Pinecone index '{INDEX_NAME}' (dim={DIMENSION}) ...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    index = pc.Index(INDEX_NAME)
    st.success(f"✅ Connected to Pinecone index: '{INDEX_NAME}'")
except Exception as e:
    st.error(f"❌ Pinecone connection failed: {e}")
    index = None


# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------
def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def rand_vec():
    """Generate a random 1024-dim vector."""
    return np.random.rand(DIMENSION).tolist()

def safe_rerun():
    try:
        st.experimental_rerun()
    except Exception:
        pass

def safe_upsert(md):
    """Safely insert/update into Pinecone."""
    if not index:
        st.error("⚠️ Pinecone index unavailable.")
        return
    try:
        index.upsert([
            {
                "id": str(md.get("_id", uuid.uuid4())),
                "values": rand_vec(),
                "metadata": md
            }
        ])
    except Exception as e:
        st.error(f"⚠️ Upsert failed: {e}")


# -----------------------------
# NOTIFICATION SYSTEM
# -----------------------------
def send_email(to, subject, message):
    if not EMAIL_SENDER or not EMAIL_PASSWORD or not to:
        return
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_SENDER
        msg["To"] = to
        msg["Subject"] = subject
        msg.attach(MIMEText(message, "plain"))
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
    except Exception as e:
        st.warning(f"📧 Email failed: {e}")

def send_sms(to, message):
    if not SMS_SID or not SMS_AUTH or not SMS_FROM or not to:
        return
    try:
        client = Client(SMS_SID, SMS_AUTH)
        client.messages.create(body=message, from_=SMS_FROM, to=to)
    except Exception as e:
        st.warning(f"📱 SMS failed: {e}")

def send_notification(email=None, phone=None, subject="Update", msg="Task update"):
    send_email(email, subject, msg)
    send_sms(phone, msg)


# -----------------------------
# BASIC ML MODELS (SIMULATION)
# -----------------------------
lin_reg = LinearRegression().fit([[0], [50], [100]], [0, 2.5, 5])
log_reg = LogisticRegression().fit([[0], [40], [80], [100]], [0, 0, 1, 1])
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(["excellent work", "needs improvement", "bad performance", "great job", "average"])
svm_clf = SVC()
svm_clf.fit(X_train, [1, 0, 0, 1, 0])
rf = RandomForestClassifier().fit(np.array([[10, 2], [50, 1], [90, 0], [100, 0]]), [0, 1, 0, 0])


# -----------------------------
# FETCH DATA
# -----------------------------
def fetch_all():
    if not index:
        return pd.DataFrame()
    try:
        res = index.query(vector=rand_vec(), top_k=1000, include_metadata=True)
        rows = []
        for m in res.matches:
            md = m.metadata or {}
            md["_id"] = m.id
            rows.append(md)
        return pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"⚠️ Unable to fetch data: {e}")
        return pd.DataFrame()


# -----------------------------
# ROLE SELECTION
# -----------------------------
role = st.sidebar.radio("🔑 Login as", ["Manager", "Team Member", "Client"])
current_month = datetime.now().strftime("%B %Y")

# -----------------------------
# MANAGER DASHBOARD
# -----------------------------
if role == "Manager":
    st.header("👨‍💼 Manager Dashboard")

    tab1, tab2, tab3 = st.tabs(["📝 Assign Task", "✅ Review Tasks", "📊 360° Overview"])

    # Assign Task
    with tab1:
        with st.form("assign_form"):
            company = st.text_input("🏢 Company Name")
            employee = st.text_input("👤 Employee Name")
            email = st.text_input("📧 Employee Email")
            phone = st.text_input("📱 Employee Phone")
            task = st.text_input("🧾 Task Title")
            desc = st.text_area("🖊️ Description")
            deadline = st.date_input("📅 Deadline", value=date.today())
            submit = st.form_submit_button("Assign Task")

            if submit and company and employee and task:
                md = {
                    "_id": str(uuid.uuid4()), "company": company, "employee": employee,
                    "email": email, "phone": phone, "task": task, "description": desc,
                    "deadline": deadline.isoformat(), "month": current_month,
                    "completion": 0, "marks": 0, "status": "Assigned",
                    "reviewed": False, "assigned_on": now()
                }
                safe_upsert(md)
                send_notification(email, phone,
                    subject=f"New Task: {task}",
                    msg=f"Hi {employee}, you have been assigned a new task: '{task}'\nDeadline: {deadline}\nDescription: {desc}")
                st.success(f"✅ Task '{task}' assigned successfully!")

    # Review Tasks
    with tab2:
        company = st.text_input("Filter by Company")
        employee = st.text_input("Filter by Employee")
        if st.button("Load Tasks"):
            res = index.query(vector=rand_vec(), top_k=500, include_metadata=True,
                              filter={"company": {"$eq": company}, "employee": {"$eq": employee}})
            st.session_state["tasks"] = [(m.id, m.metadata) for m in res.matches or []]
            st.success(f"Loaded {len(st.session_state['tasks'])} tasks.")

        for tid, r in st.session_state.get("tasks", []):
            st.subheader(r.get("task"))
            adj = st.slider(f"Completion {r.get('task')}", 0, 100, int(r.get("completion", 0)))
            adj_marks = float(lin_reg.predict([[adj]])[0])
            comments = st.text_area("Manager Comments", key=f"c_{tid}")
            approve = st.radio("Approve?", ["Yes", "No"], key=f"a_{tid}")

            if st.button(f"Finalize {r.get('task')}", key=f"f_{tid}"):
                sentiment_val = int(svm_clf.predict(vectorizer.transform([comments]))[0])
                sentiment = "Positive" if sentiment_val == 1 else "Negative"
                md = {
                    **r, "completion": adj, "marks": adj_marks,
                    "sentiment": sentiment, "approved": approve == "Yes",
                    "reviewed": True, "reviewed_on": now()
                }
                safe_upsert(md)
                send_notification(r.get("email"), r.get("phone"),
                    subject=f"Task Review: {r.get('task')}",
                    msg=f"Task '{r.get('task')}' reviewed.\nCompletion: {adj}%\nMarks: {adj_marks:.2f}\nSentiment: {sentiment}\nApproval: {approve}")
                st.success(f"✅ Review completed for {r.get('task')} ({sentiment})")
                safe_rerun()

    # 360 Overview
    with tab3:
        df = fetch_all()
        if not df.empty and {"completion", "marks"}.issubset(df.columns):
            df["completion"] = pd.to_numeric(df["completion"], errors="coerce")
            df["marks"] = pd.to_numeric(df["marks"], errors="coerce")
            df.dropna(subset=["completion", "marks"], inplace=True)
            if len(df) > 3:
                kmeans = KMeans(n_clusters=3, n_init=10, random_state=42).fit(df[["completion", "marks"]])
                df["cluster"] = kmeans.labels_
                fig = px.scatter(df, x="completion", y="marks",
                                 color=df["cluster"].astype(str),
                                 hover_data=["employee", "task"],
                                 title="Employee Performance Clusters")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Not enough data for visualization.")


# -----------------------------
# TEAM MEMBER
# -----------------------------
elif role == "Team Member":
    st.header("👷 Team Member Portal")
    company = st.text_input("Company")
    employee = st.text_input("Your Name")
    if st.button("Load My Tasks"):
        res = index.query(vector=rand_vec(), top_k=500, include_metadata=True,
                          filter={"company": {"$eq": company}, "employee": {"$eq": employee}})
        st.session_state["tasks"] = [(m.id, m.metadata) for m in res.matches or []]
        st.success(f"Loaded {len(st.session_state['tasks'])} tasks.")

    for tid, md in st.session_state.get("tasks", []):
        st.subheader(md.get("task"))
        curr = float(md.get("completion", 0))
        new = st.slider(f"Completion for {md.get('task')}", 0, 100, int(curr))
        if st.button(f"Submit {md.get('task')}", key=tid):
            marks = float(lin_reg.predict([[new]])[0])
            status = "On Track" if log_reg.predict([[new]])[0] == 1 else "Delayed"
            risk = "High" if rf.predict([[new, 0]])[0] == 1 else "Low"
            md2 = {**md, "completion": new, "marks": marks,
                   "status": status, "risk": risk, "submitted_on": now()}
            safe_upsert(md2)
            send_notification(md.get("email"), md.get("phone"),
                subject=f"Task Update: {md.get('task')}",
                msg=f"Task '{md.get('task')}' updated to {new}% ({status})")
            st.success(f"✅ Updated {md.get('task')} ({status})")
            safe_rerun()

# -----------------------------
# CLIENT PORTAL
# -----------------------------
elif role == "Client":
    st.header("🧾 Client Review")
    company = st.text_input("Company Name")
    if st.button("Load Reviewed Tasks"):
        res = index.query(vector=rand_vec(), top_k=500, include_metadata=True,
                          filter={"company": {"$eq": company}, "reviewed": {"$eq": True}})
        st.session_state["ctasks"] = [(m.id, m.metadata) for m in res.matches or []]
        st.success(f"Loaded {len(st.session_state['ctasks'])} reviewed tasks.")
    for tid, md in st.session_state.get("ctasks", []):
        st.subheader(md.get("task"))
        st.write(f"Employee: {md.get('employee')}")
        st.write(f"Marks: {md.get('marks')}")
        comment = st.text_area("Client Feedback", key=f"cf_{tid}")
        if st.button(f"Approve {md.get('task')}", key=f"app_{tid}"):
            md2 = {**md, "client_reviewed": True, "client_comments": comment, "client_approved_on": now()}
            safe_upsert(md2)
            send_notification(md.get("email"), md.get("phone"),
                subject=f"Client Approval: {md.get('task')}",
                msg=f"Client approved task '{md.get('task')}'. Feedback: {comment}")
            st.success(f"✅ Approved {md.get('task')}")
            safe_rerun()
