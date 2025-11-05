import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import pandas as pd
from datetime import datetime, date
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import plotly.express as px
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from twilio.rest import Client
import uuid
from textblob import TextBlob

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Task Management System", layout="wide")
st.title("🚀 AI-Powered Task Management System")

# -----------------------------
# INITIALIZATION
# -----------------------------
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
EMAIL_SENDER = st.secrets.get("EMAIL_SENDER", "")
EMAIL_PASSWORD = st.secrets.get("EMAIL_PASSWORD", "")
SMS_SID = st.secrets.get("SMS_SID", "")
SMS_AUTH = st.secrets.get("SMS_AUTH", "")
SMS_FROM = st.secrets.get("SMS_FROM", "")

pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "task"
DIMENSION = 1024

if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(INDEX_NAME)

def now(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def rand_vec(): return np.random.rand(DIMENSION).tolist()

# -----------------------------
# SAFETY HELPERS
# -----------------------------
def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

def safe_upsert(index, md):
    try:
        index.upsert([{
            "id": str(md.get("_id", uuid.uuid4())),
            "values": rand_vec(),
            "metadata": md
        }])
    except Exception as e:
        st.error(f"Pinecone upsert failed: {e}")

# -----------------------------
# NOTIFICATION SYSTEM
# -----------------------------
def send_email(to, subject, message):
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        st.warning("⚠️ Email credentials not set in secrets.")
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
        st.warning(f"Email send failed: {e}")

def send_sms(to, message):
    if not SMS_SID or not SMS_AUTH or not SMS_FROM:
        st.info("ℹ️ SMS credentials not set in secrets.")
        return
    try:
        client = Client(SMS_SID, SMS_AUTH)
        client.messages.create(body=message, from_=SMS_FROM, to=to)
    except Exception as e:
        st.warning(f"SMS send failed: {e}")

def send_notification(target_email=None, phone=None, subject="Update", msg="Task update"):
    if target_email:
        send_email(target_email, subject, msg)
    if phone:
        send_sms(phone, msg)

# -----------------------------
# SIMPLE MODELS
# -----------------------------
lin_reg = LinearRegression()
lin_reg.fit([[0], [50], [100]], [0, 2.5, 5])

log_reg = LogisticRegression()
log_reg.fit([[0], [40], [80], [100]], [0, 0, 1, 1])

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(["excellent work", "needs improvement", "bad performance", "great job", "average"])
svm_clf = SVC()
svm_clf.fit(X_train, [1, 0, 0, 1, 0])

rf = RandomForestClassifier()
rf.fit(np.array([[10, 2], [50, 1], [90, 0], [100, 0]]), [0, 1, 0, 0])

# -----------------------------
# AI FEEDBACK SUMMARIZATION
# -----------------------------
def summarize_feedback(feedback_list):
    """
    Summarizes a list of feedback comments into a short, human-readable summary.
    Uses lightweight NLP (TextBlob) for quick sentiment-based summarization.
    """
    if not feedback_list or len(feedback_list) == 0:
        return "No feedback provided yet."
    
    combined_text = " ".join(feedback_list)
    blob = TextBlob(combined_text)
    
    sentences = blob.sentences
    if len(sentences) == 0:
        return combined_text[:100] + "..."
    
    sorted_sentences = sorted(sentences, key=lambda s: abs(s.sentiment.polarity), reverse=True)
    top_sentences = sorted_sentences[:3]
    summary = " ".join(str(s) for s in top_sentences)
    
    return summary if summary else combined_text[:150] + "..."

# -----------------------------
# HELPERS
# -----------------------------
def fetch_all():
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
role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client"])
current_month = datetime.now().strftime("%B %Y")

# -----------------------------
# MANAGER SECTION (unchanged)
# -----------------------------
if role == "Manager":
    st.header("👨‍💼 Manager Dashboard")
    st.info("Use this dashboard to assign, review, and analyze team performance.")
    # (Existing Manager code remains same)

# -----------------------------
# TEAM MEMBER SECTION
# -----------------------------
elif role == "Team Member":
    st.header("👷 Team Member Portal")
    tab1, tab2 = st.tabs(["My Tasks", "AI Feedback Summarization"])

    # --- Tab 1: Task Updates ---
    with tab1:
        company = st.text_input("Company")
        employee = st.text_input("Your Name")
        if st.button("Load Tasks"):
            res = index.query(vector=rand_vec(), top_k=500, include_metadata=True,
                              filter={"company": {"$eq": company}, "employee": {"$eq": employee}})
            st.session_state["tasks"] = [(m.id, m.metadata) for m in res.matches or []]
            st.success(f"Loaded {len(st.session_state['tasks'])} tasks.")
        for tid, md in st.session_state.get("tasks", []):
            st.subheader(md.get("task"))
            curr = float(md.get("completion", 0))
            new = st.slider(f"Completion {md.get('task')}", 0, 100, int(curr))
            if st.button(f"Submit {md.get('task')}", key=tid):
                marks = float(lin_reg.predict([[new]])[0])
                track = "On Track" if log_reg.predict([[new]])[0] == 1 else "Delayed"
                miss = rf.predict([[new, 0]])[0]
                md2 = {**md, "completion": new, "marks": marks,
                       "status": track, "deadline_risk": "High" if miss else "Low",
                       "submitted_on": now()}
                safe_upsert(index, md2)
                send_notification(md.get("email"), md.get("phone"),
                    subject=f"Task Update: {md.get('task')}",
                    msg=f"Task '{md.get('task')}' updated to {new}% ({track})")
                st.success(f"✅ Updated {md.get('task')} ({track})")
                safe_rerun()

    # --- Tab 2: AI Feedback Summarization ---
    with tab2:
        st.subheader("🧠 AI Feedback Summarization")
        company = st.text_input("Company Name (for summary)")
        employee = st.text_input("Your Name (for summary)")

        if st.button("Generate AI Summary"):
            res = index.query(vector=rand_vec(), top_k=500, include_metadata=True,
                              filter={"company": {"$eq": company}, "employee": {"$eq": employee}})
            feedbacks = []
            for m in res.matches:
                md = m.metadata
                if "client_comments" in md:
                    feedbacks.append(md["client_comments"])
                if "manager_feedback" in md:
                    feedbacks.append(md["manager_feedback"])
            summary = summarize_feedback(feedbacks)
            st.markdown(f"### 🧾 AI Summary of Your Feedback:")
            st.info(summary)

# -----------------------------
# CLIENT SECTION
# -----------------------------
elif role == "Client":
    st.header("🧾 Client Review")
    company = st.text_input("Company Name")
    if st.button("Load Completed"):
        res = index.query(vector=rand_vec(), top_k=500, include_metadata=True,
                          filter={"company": {"$eq": company}, "reviewed": {"$eq": True}})
        st.session_state["ctasks"] = [(m.id, m.metadata) for m in res.matches or []]
        st.success(f"Loaded {len(st.session_state['ctasks'])} tasks.")
    for tid, md in st.session_state.get("ctasks", []):
        st.subheader(md.get("task"))
        st.write(f"Employee: {md.get('employee')}")
        st.write(f"Final Completion: {md.get('completion')}%")
        st.write(f"Marks: {md.get('marks')}")
        comment = st.text_area(f"Client Feedback ({md.get('task')})", key=f"cf_{tid}")
        if st.button(f"Approve {md.get('task')}", key=f"app_{tid}"):

            md2 = {**md, "client_reviewed": True, "client_comments": comment, "client_approved_on": now()}
            safe_upsert(index, md2)
            send_notification(md.get("email"), md.get("phone"),
                subject=f"Client Approval: {md.get('task')}",
                msg=f"Client has approved task '{md.get('task')}'. Feedback: {comment}")
            st.success(f"✅ Approved {md.get('task')}")
            safe_rerun()
