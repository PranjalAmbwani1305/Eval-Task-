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
st.set_page_config(page_title="🤖 AI-Driven Employee Management System", layout="wide")
st.title("🤖 AI-Driven Employee Management System")

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

# Create index if not exists
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
# HELPERS
# -----------------------------
def safe_upsert(md):
    try:
        index.upsert([{
            "id": str(md.get("_id", uuid.uuid4())),
            "values": rand_vec(),
            "metadata": md
        }])
    except Exception as e:
        st.error(f"Pinecone upsert failed: {e}")

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

def safe_rerun():
    try: st.rerun()
    except Exception: pass

# -----------------------------
# NOTIFICATIONS
# -----------------------------
def send_email(to, subject, message):
    if not EMAIL_SENDER or not EMAIL_PASSWORD: return
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
        st.warning(f"Email failed: {e}")

def send_sms(to, message):
    if not SMS_SID or not SMS_AUTH or not SMS_FROM: return
    try:
        client = Client(SMS_SID, SMS_AUTH)
        client.messages.create(body=message, from_=SMS_FROM, to=to)
    except Exception as e:
        st.warning(f"SMS failed: {e}")

def send_notification(email=None, phone=None, subject="Update", msg="Task update"):
    if email: send_email(email, subject, msg)
    if phone: send_sms(phone, msg)

# -----------------------------
# SIMPLE MODELS
# -----------------------------
lin_reg = LinearRegression().fit([[0], [50], [100]], [0, 2.5, 5])
log_reg = LogisticRegression().fit([[0], [40], [80], [100]], [0, 0, 1, 1])
rf = RandomForestClassifier().fit(np.array([[10, 2], [50, 1], [90, 0], [100, 0]]), [0, 1, 0, 0])
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(["excellent work", "needs improvement", "bad performance", "great job", "average"])
svm_clf = SVC().fit(X_train, [1, 0, 0, 1, 0])

# -----------------------------
# ROLE SELECTION
# -----------------------------
role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client", "Admin"])
current_month = datetime.now().strftime("%B %Y")

# -----------------------------
# MANAGER DASHBOARD
# -----------------------------
if role == "Manager":
    st.header("👨‍💼 Manager Dashboard")
    tab1, tab2, tab3 = st.tabs(["Assign Task", "Review Tasks", "360° Overview"])

    # --- Assign Task ---
    with tab1:
        with st.form("assign_task"):
            company = st.text_input("Company")
            employee = st.text_input("Employee Name")
            email = st.text_input("Employee Email")
            phone = st.text_input("Employee Phone")
            task = st.text_input("Task Title")
            desc = st.text_area("Description")
            deadline = st.date_input("Deadline", date.today())
            if st.form_submit_button("Assign"):
                md = {
                    "_id": str(uuid.uuid4()), "type": "task", "company": company, "employee": employee,
                    "email": email, "phone": phone, "task": task, "description": desc,
                    "completion": 0, "marks": 0, "month": current_month,
                    "deadline": str(deadline), "status": "Assigned", "assigned_on": now()
                }
                safe_upsert(md)
                send_notification(email, phone, f"Task Assigned: {task}", f"You have a new task: {task}")
                st.success(f"✅ Task '{task}' assigned to {employee}")

    # --- Review Tasks ---
    with tab2:
        st.subheader("📋 Review Employee Tasks")
        company = st.text_input("Company for Review")
        employee = st.text_input("Employee Name for Review")
        if st.button("Load Tasks"):
            res = index.query(vector=rand_vec(), top_k=100, include_metadata=True,
                              filter={"company": {"$eq": company}, "employee": {"$eq": employee}})
            st.session_state["review"] = [(m.id, m.metadata) for m in res.matches]
        for tid, r in st.session_state.get("review", []):
            st.markdown(f"### 🧾 {r.get('task')}")
            adj = st.slider(f"Completion for {r.get('task')}", 0, 100, int(r.get("completion", 0)))
            marks = float(lin_reg.predict([[adj]])[0])
            comments = st.text_area("Manager Comments", key=f"c_{tid}")
            approve = st.radio("Approve Task?", ["Yes", "No"], key=f"a_{tid}")
            if st.button(f"Finalize Review {r.get('task')}", key=f"f_{tid}"):
                sentiment = "Positive" if svm_clf.predict(vectorizer.transform([comments]))[0] else "Negative"
                md = {**r, "completion": adj, "marks": marks, "sentiment": sentiment,
                      "approved_by_boss": approve == "Yes", "reviewed": True, "reviewed_on": now()}
                safe_upsert(md)
                send_notification(r.get("email"), r.get("phone"), "Task Reviewed",
                                  f"Your task {r.get('task')} reviewed. Marks: {marks:.2f}, Sentiment: {sentiment}")
                st.success(f"✅ Reviewed {r.get('task')} ({sentiment})")
                safe_rerun()

    # --- 360° Overview ---
    with tab3:
        st.subheader("📊 360° Performance Overview")
        df = fetch_all()
        if not df.empty:
            for col in ["marks", "completion"]:
                if col not in df.columns: df[col] = np.nan
            df["marks"] = pd.to_numeric(df["marks"], errors="coerce")
            df["completion"] = pd.to_numeric(df["completion"], errors="coerce")
            df = df.dropna(subset=["marks", "completion"])
            if len(df) >= 3:
                kmeans = KMeans(n_clusters=3, n_init=10, random_state=42).fit(df[["completion", "marks"]])
                df["cluster"] = kmeans.labels_
                fig = px.scatter(df, x="completion", y="marks", color=df["cluster"].astype(str),
                                 hover_data=["employee", "task"], title="Employee Performance Clusters")
                st.plotly_chart(fig, use_container_width=True)
            st.info(f"Average Marks: {df['marks'].mean():.2f}")

# -----------------------------
# TEAM MEMBER PORTAL
# -----------------------------
elif role == "Team Member":
    st.header("👷 Team Member Portal")
    tab1, tab2, tab3 = st.tabs(["My Tasks", "AI Feedback", "Submit Leave"])

    # --- My Tasks ---
    with tab1:
        company = st.text_input("Company Name")
        employee = st.text_input("Your Name")
        if st.button("Load Tasks"):
            res = index.query(vector=rand_vec(), top_k=200, include_metadata=True,
                              filter={"company": {"$eq": company}, "employee": {"$eq": employee}})
            st.session_state["tasks"] = [(m.id, m.metadata) for m in res.matches]
        for tid, md in st.session_state.get("tasks", []):
            st.markdown(f"### 🧾 {md.get('task')}")
            prog = st.slider("Update Completion", 0, 100, int(md.get("completion", 0)), key=tid)
            if st.button(f"Submit {md.get('task')}", key=f"s_{tid}"):
                marks = float(lin_reg.predict([[prog]])[0])
                track = "On Track" if log_reg.predict([[prog]])[0] else "Delayed"
                md2 = {**md, "completion": prog, "marks": marks, "status": track, "submitted_on": now()}
                safe_upsert(md2)
                st.success(f"✅ Updated {md.get('task')} to {prog}% ({track})")
                safe_rerun()

    # --- AI Feedback ---
    with tab2:
        st.subheader("🧠 AI Feedback Summarization")
        company = st.text_input("Company Name")
        employee = st.text_input("Employee Name")
        feedback = st.text_area("Feedback Text")
        if st.button("Analyze Feedback"):
            blob = TextBlob(feedback)
            pol = blob.sentiment.polarity
            if pol > 0.1:
                st.success("Positive Feedback 😊")
            elif pol < -0.1:
                st.error("Negative Feedback 😞")
            else:
                st.info("Neutral Feedback 😐")
            st.write(f"Polarity: {pol:.2f}")
            st.write(f"Keywords: {', '.join(blob.noun_phrases)}")

    # --- Submit Leave ---
    with tab3:
        st.subheader("🏖️ Submit Leave Request")
        company = st.text_input("Company")
        employee = st.text_input("Your Name")
        start = st.date_input("Start Date")
        end = st.date_input("End Date")
        reason = st.text_area("Reason")
        if st.button("Submit Leave"):
            md = {"_id": str(uuid.uuid4()), "type": "leave", "company": company,
                  "employee": employee, "start_date": str(start), "end_date": str(end),
                  "reason": reason, "status": "Pending", "requested_on": now()}
            safe_upsert(md)
            st.success(f"✅ Leave submitted from {start} to {end}")

# -----------------------------
# CLIENT REVIEW
# -----------------------------
elif role == "Client":
    st.header("🧾 Client Review")
    company = st.text_input("Company Name")
    if st.button("Load Completed Tasks"):
        res = index.query(vector=rand_vec(), top_k=500, include_metadata=True,
                          filter={"company": {"$eq": company}, "reviewed": {"$eq": True}})
        st.session_state["client_tasks"] = [(m.id, m.metadata) for m in res.matches]
    for tid, md in st.session_state.get("client_tasks", []):
        st.subheader(md.get("task"))
        comment = st.text_area(f"Client Feedback ({md.get('task')})", key=f"cf_{tid}")
        if st.button(f"Approve {md.get('task')}", key=f"app_{tid}"):
            md2 = {**md, "client_reviewed": True, "client_comments": comment, "client_approved_on": now()}
            safe_upsert(md2)
            st.success(f"✅ Approved {md.get('task')}")

# -----------------------------
# ADMIN DASHBOARD
# -----------------------------
elif role == "Admin":
    st.header("🏢 Admin Dashboard")
    df = fetch_all()
    if df.empty:
        st.info("No data found yet. Assign tasks and submit updates first.")
    else:
        df["marks"] = pd.to_numeric(df.get("marks", 0), errors="coerce")
        df["completion"] = pd.to_numeric(df.get("completion", 0), errors="coerce")

        st.subheader("📊 Department-wise Performance")
        if "department" not in df.columns:
            df["department"] = np.random.choice(["Tech", "Design", "HR", "Sales"], size=len(df))
        dep_avg = df.groupby("department")["marks"].mean().reset_index()
        st.plotly_chart(px.bar(dep_avg, x="department", y="marks", title="Average Marks by Department"))

        st.subheader("🏅 Top Employees")
        top = df.sort_values("marks", ascending=False).head(10)
        st.dataframe(top[["employee", "company", "marks", "completion"]])

        st.subheader("🏖️ Leave Overview")
        leaves = df[df.get("type") == "leave"]
        if not leaves.empty:
            st.dataframe(leaves[["employee", "start_date", "end_date", "status"]])
        else:
            st.write("No leave requests found.")
