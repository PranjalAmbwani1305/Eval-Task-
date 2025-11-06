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
    try:
        st.rerun()
    except Exception:
        pass

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
# MANAGER DASHBOARD
# -----------------------------
if role == "Manager":
    st.header("👨‍💼 Manager Dashboard")
    st.info("Use this dashboard to assign, review, and analyze team performance.")
    tab1, tab2, tab3, tab4 = st.tabs(["Assign Task", "Review Tasks", "360° Overview", "Managerial Actions"])

    # --- Assign Task ---
    with tab1:
        with st.form("assign"):
            company = st.text_input("Company Name")
            employee = st.text_input("Employee Name")
            employee_email = st.text_input("Employee Email (optional)")
            employee_phone = st.text_input("Employee Phone (optional)")
            task = st.text_input("Task Title")
            desc = st.text_area("Description")
            deadline = st.date_input("Deadline", value=date.today())
            submit = st.form_submit_button("Assign Task")

            if submit and company and employee and task:
                tid = str(uuid.uuid4())
                md = {
                    "_id": tid, "company": company, "employee": employee,
                    "email": employee_email, "phone": employee_phone,
                    "task": task, "description": desc,
                    "deadline": deadline.isoformat(),
                    "month": current_month, "completion": 0,
                    "marks": 0, "status": "Assigned",
                    "reviewed": False, "assigned_on": now()
                }
                safe_upsert(index, md)
                send_notification(employee_email, employee_phone,
                    subject=f"New Task Assigned: {task}",
                    msg=f"Hello {employee}, you have been assigned a new task: {task}\nDeadline: {deadline}\nDescription: {desc}")
                st.success(f"✅ Task '{task}' assigned to {employee}")

    # --- Review Tasks (Filter by Company & Employee) ---
    with tab2:
        st.subheader("🔍 Review Tasks by Employee and Company")
        company = st.text_input("Enter Company Name to Review")
        employee = st.text_input("Enter Employee Name to Review")

        if st.button("Load Tasks"):
            if company and employee:
                try:
                    res = index.query(
                        vector=rand_vec(),
                        top_k=500,
                        include_metadata=True,
                        filter={"company": {"$eq": company}, "employee": {"$eq": employee}}
                    )
                    st.session_state["review_tasks"] = [(m.id, m.metadata) for m in res.matches or []]
                    st.success(f"Loaded {len(st.session_state['review_tasks'])} tasks for {employee} at {company}.")
                except Exception as e:
                    st.warning(f"⚠️ Error fetching tasks: {e}")
            else:
                st.warning("Please enter both Company and Employee names to load tasks.")

        for tid, r in st.session_state.get("review_tasks", []):
            st.markdown(f"### 📝 {r.get('task', 'Unnamed Task')}")
            adj = st.slider(f"Adjust Completion ({r.get('employee', '')})", 0, 100, int(r.get("completion", 0)))
            adj_marks = float(lin_reg.predict([[adj]])[0])
            comments = st.text_area(f"Manager Comments ({r.get('task', '')})", key=f"c_{tid}")
            approve = st.radio(f"Approve {r.get('task', '')}?", ["Yes", "No"], key=f"a_{tid}")

            if st.button(f"Finalize Review {r.get('task', '')}", key=f"f_{tid}"):
                sentiment_val = int(svm_clf.predict(vectorizer.transform([comments]))[0])
                sentiment = "Positive" if sentiment_val == 1 else "Negative"
                md = {
                    **r,
                    "completion": adj,
                    "marks": adj_marks,
                    "reviewed": True,
                    "sentiment": sentiment,
                    "approved_by_boss": approve == "Yes",
                    "reviewed_on": now()
                }
                safe_upsert(index, md)
                send_notification(
                    r.get("email"),
                    r.get("phone"),
                    subject=f"Task Review: {r.get('task')}",
                    msg=(
                        f"Hello {r.get('employee')},\n\n"
                        f"Your task '{r.get('task')}' has been reviewed.\n"
                        f"Completion: {adj}%\nMarks: {adj_marks:.2f}\n"
                        f"Sentiment: {sentiment}\nApproval: {approve}\n\n"
                        f"Regards,\n{company} Management"
                    )
                )
                st.success(f"✅ Review finalized for {r.get('task')} ({sentiment})")
                safe_rerun()

    # --- 360° Overview ---
    with tab3:
        st.subheader("📊 360° Performance Overview")
        df = fetch_all()
        if not df.empty:
            df["marks"] = pd.to_numeric(df["marks"], errors="coerce")
            df["completion"] = pd.to_numeric(df["completion"], errors="coerce")
            df = df.dropna(subset=["marks", "completion"])

            if len(df) >= 3:
                kmeans = KMeans(n_clusters=3, n_init=10, random_state=42).fit(df[["completion", "marks"]])
                df["cluster"] = kmeans.labels_
                fig = px.scatter(df, x="completion", y="marks",
                                 color=df["cluster"].astype(str),
                                 hover_data=["employee", "task"],
                                 title="Employee Clusters",
                                 color_discrete_sequence=px.colors.qualitative.Dark24)
                st.plotly_chart(fig, use_container_width=True)
            avg_marks = df["marks"].mean()
            st.info(f"📈 Avg Marks: {avg_marks:.2f}")

    # --- Managerial Actions ---
    with tab4:
        st.subheader("⚙️ Managerial Actions & Approvals")
        st.markdown("""
        - Task reassignments or escalations  
        - Approve leave / overtime / deliverables  
        - Send appreciation / warning / suggestion notes  
        - Generate 360° performance summaries for appraisal cycles  
        """)

# -----------------------------
# TEAM MEMBER PORTAL
# -----------------------------
elif role == "Team Member":
    st.header("👷 Team Member Portal")
    tab1, tab2 = st.tabs(["My Tasks", "AI Feedback Summarization"])

    # --- My Tasks ---
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

    # --- AI Feedback Summarization ---
    with tab2:
        st.subheader("🧠 AI Feedback Summarization")
        company = st.text_input("🏢 Company Name (for summary)")
        employee = st.text_input("👤 Your Name (for summary)")
        feedback_text = st.text_area("💬 Paste feedback text below:")

        if st.button("🔍 Analyze Feedback"):
            if feedback_text.strip():
                blob = TextBlob(feedback_text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity

                if polarity > 0.1:
                    sentiment = "✅ Overall Positive Feedback"
                    emoji = "😄"
                    advice = "Great job! Keep maintaining consistency and communication."
                elif polarity < -0.1:
                    sentiment = "⚠️ Overall Negative Feedback"
                    emoji = "😟"
                    advice = "Consider addressing pain points or communication gaps with your manager."
                else:
                    sentiment = "💬 Neutral Feedback"
                    emoji = "😐"
                    advice = "Feedback seems balanced. Continue monitoring improvements."

                st.markdown("---")
                st.markdown(f"### {emoji} Sentiment Analysis Result")
                st.markdown(f"**Sentiment:** {sentiment}")
                st.markdown(f"**Polarity Score:** `{polarity:.3f}`  |  **Subjectivity:** `{subjectivity:.3f}`")
                st.progress((polarity + 1) / 2)

                keywords = list(blob.noun_phrases)
                if keywords:
                    st.markdown(f"**Key Themes:** {', '.join(keywords)}")
                else:
                    st.markdown("**Key Themes:** No specific keywords detected.")

                st.info(f"💡 Suggestion: {advice}")
                summary = " ".join(keywords[:5]) if keywords else "General performance feedback."
                st.success(f"📘 Summary for {employee or 'Employee'} ({company or 'Company'}): {summary}")
            else:
                st.warning("⚠️ Please paste some feedback text to analyze.")

# -----------------------------
# CLIENT REVIEW
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
