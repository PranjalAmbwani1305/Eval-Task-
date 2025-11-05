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
import re
import uuid
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
# from twilio.rest import Client   # Optional: uncomment if SMS configured

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Task Management System", layout="wide")
st.title("🚀 AI-Powered Task Management System")

# -----------------------------
# INITIALIZATION
# -----------------------------
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
EMAIL_SENDER = st.secrets.get("EMAIL_SENDER", "")
EMAIL_PASSWORD = st.secrets.get("EMAIL_PASSWORD", "")

pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "task"
DIMENSION = 1024

if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(INDEX_NAME)


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def rand_vec():
    return np.random.rand(DIMENSION).tolist()


# -----------------------------
# SAFE HELPERS
# -----------------------------
def safe_upsert(index, md):
    try:
        index.upsert(
            [
                {
                    "id": str(md.get("_id", uuid.uuid4())),
                    "values": rand_vec(),
                    "metadata": md,
                }
            ]
        )
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
    try:
        st.rerun()
    except:
        st.experimental_rerun()


# -----------------------------
# EMAIL NOTIFICATION
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


# -----------------------------
# SIMPLE ML MODELS
# -----------------------------
lin_reg = LinearRegression().fit([[0], [50], [100]], [0, 2.5, 5])
log_reg = LogisticRegression().fit([[0], [40], [80], [100]], [0, 0, 1, 1])
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(
    ["excellent work", "needs improvement", "bad performance", "great job", "average"]
)
svm_clf = SVC().fit(X_train, [1, 0, 0, 1, 0])
rf = RandomForestClassifier().fit(np.array([[10, 2], [50, 1], [90, 0], [100, 0]]), [0, 1, 0, 0])

# -----------------------------
# NLP: SAFE FEEDBACK SUMMARIZATION
# -----------------------------
def summarize_feedback(feedbacks):
    """Summarize feedback text safely without requiring NLTK corpora."""
    if not feedbacks:
        return "No feedback available to summarize."

    text = " ".join(feedbacks)
    # Simple sentence segmentation without punkt
    sentences = re.split(r'(?<=[.!?]) +', text)
    blob = TextBlob(text)

    # Compute sentiment manually
    polarity = blob.sentiment.polarity
    if polarity > 0.2:
        sentiment = "😊 Overall Positive Feedback"
    elif polarity < -0.2:
        sentiment = "⚠️ Overall Negative Feedback"
    else:
        sentiment = "😐 Neutral Feedback"

    # Frequency-based summarization
    word_freq = {}
    for word in re.findall(r'\w+', text.lower()):
        word_freq[word] = word_freq.get(word, 0) + 1
    top_sentences = sorted(
        sentences,
        key=lambda s: sum(word_freq.get(w.lower(), 0) for w in re.findall(r'\w+', s)),
        reverse=True,
    )[:3]

    summary = " ".join(top_sentences)
    return f"{sentiment}\n\n**Summary:** {summary}"


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
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Assign Task", "Review Tasks", "360° & Clustering", "Managerial Actions"]
    )

    # --- Assign Task ---
    with tab1:
        with st.form("assign"):
            company = st.text_input("Company Name")
            employee = st.text_input("Employee Name")
            employee_email = st.text_input("Employee Email (optional)")
            task = st.text_input("Task Title")
            desc = st.text_area("Description")
            deadline = st.date_input("Deadline", value=date.today())
            submit = st.form_submit_button("Assign Task")

            if submit and company and employee and task:
                tid = str(uuid.uuid4())
                md = {
                    "_id": tid,
                    "company": company,
                    "employee": employee,
                    "email": employee_email,
                    "task": task,
                    "description": desc,
                    "deadline": deadline.isoformat(),
                    "month": current_month,
                    "completion": 0,
                    "marks": 0,
                    "status": "Assigned",
                    "reviewed": False,
                    "assigned_on": now(),
                }
                safe_upsert(index, md)
                if employee_email:
                    send_email(employee_email, f"New Task: {task}", f"Hello {employee}, new task assigned.")
                st.success(f"✅ Task '{task}' assigned to {employee}")

    # --- Review Tasks ---
    with tab2:
        df = fetch_all()
        if df.empty:
            st.warning("No tasks found.")
        else:
            for _, r in df.iterrows():
                st.subheader(f"📋 {r.get('task', 'Unnamed Task')}")
                adj = st.slider(
                    f"Adjust Completion ({r.get('employee', '')})", 0, 100, int(r.get("completion", 0))
                )
                comments = st.text_area(f"Manager Comments ({r.get('task', '')})", key=f"c_{r['_id']}")
                approve = st.radio(f"Approve {r.get('task', '')}?", ["Yes", "No"], key=f"a_{r['_id']}")
                if st.button(f"Finalize Review {r.get('task', '')}", key=f"f_{r['_id']}"):
                    marks = float(lin_reg.predict([[adj]])[0])
                    sentiment_val = int(svm_clf.predict(vectorizer.transform([comments]))[0])
                    sentiment = "Positive" if sentiment_val == 1 else "Negative"
                    md = {
                        **r,
                        "completion": adj,
                        "marks": marks,
                        "reviewed": True,
                        "sentiment": sentiment,
                        "approved": approve == "Yes",
                        "reviewed_on": now(),
                    }
                    safe_upsert(index, md)
                    st.success(f"✅ Reviewed with sentiment: {sentiment}")
                    safe_rerun()

    # --- 360° Performance & Clustering ---
    with tab3:
        st.subheader("📊 360° Performance Overview")
        df = fetch_all()
        if not df.empty:
            df["marks"] = pd.to_numeric(df["marks"], errors="coerce")
            df["completion"] = pd.to_numeric(df["completion"], errors="coerce")
            if len(df) >= 3:
                kmeans = KMeans(n_clusters=3, n_init=10).fit(df[["completion", "marks"]].fillna(0))
                df["cluster"] = kmeans.labels_
                st.scatter_chart(df[["completion", "marks"]])
            st.dataframe(df[["employee", "task", "completion", "marks", "sentiment"]])

            reassign_task = st.selectbox("Select task to reassign", df["task"].unique())
            new_emp = st.text_input("Reassign to (employee name)")
            if st.button("🔁 Reassign Task"):
                md = df[df["task"] == reassign_task].iloc[0].to_dict()
                md["employee"] = new_emp
                md["reassigned_on"] = now()
                safe_upsert(index, md)
                st.success(f"Task '{reassign_task}' reassigned to {new_emp}")

    # --- Managerial Actions ---
    with tab4:
        st.subheader("⚙️ Managerial Actions & Approvals")
        st.markdown(
            """
            **Actions:**
            - Task reassignments or escalations  
            - Approve leave / overtime / deliverables  
            - Send appreciation / warning / suggestion notes  
            - Generate 360° performance summaries for appraisals  
            """
        )
        st.text_area("Write a note to team (AI suggestion or appreciation)")
        st.button("Send Managerial Note")


# -----------------------------
# TEAM MEMBER
# -----------------------------
elif role == "Team Member":
    st.header("👷 Team Member Portal")
    tab1, tab2 = st.tabs(["My Tasks", "AI Feedback Summarization"])

    with tab1:
        company = st.text_input("Company")
        employee = st.text_input("Your Name")
        if st.button("Load Tasks"):
            res = index.query(
                vector=rand_vec(),
                top_k=500,
                include_metadata=True,
                filter={"company": {"$eq": company}, "employee": {"$eq": employee}},
            )
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
                md2 = {
                    **md,
                    "completion": new,
                    "marks": marks,
                    "status": track,
                    "deadline_risk": "High" if miss else "Low",
                    "submitted_on": now(),
                }
                safe_upsert(index, md2)
                st.success(f"✅ Updated {md.get('task')} ({track})")
                safe_rerun()

    with tab2:
        st.subheader("🧠 AI Feedback Summarization")
        company = st.text_input("Company Name (for summary)")
        employee = st.text_input("Your Name (for summary)")
        if st.button("Summarize Feedback"):
            res = index.query(
                vector=rand_vec(),
                top_k=200,
                include_metadata=True,
                filter={"company": {"$eq": company}, "employee": {"$eq": employee}, "reviewed": {"$eq": True}},
            )
            feedbacks = [m.metadata.get("sentiment", "") + " " + m.metadata.get("description", "") for m in res.matches]
            summary = summarize_feedback(feedbacks)
            st.info(summary)


# -----------------------------
# CLIENT
# -----------------------------
elif role == "Client":
    st.header("🧾 Client Review")
    company = st.text_input("Company Name")
    if st.button("Load Completed"):
        res = index.query(
            vector=rand_vec(),
            top_k=500,
            include_metadata=True,
            filter={"company": {"$eq": company}, "reviewed": {"$eq": True}},
        )
        st.session_state["ctasks"] = [(m.id, m.metadata) for m in res.matches or []]
        st.success(f"Loaded {len(st.session_state['ctasks'])} tasks.")
    for tid, md in st.session_state.get("ctasks", []):
        st.subheader(md.get("task"))
        st.write(f"Employee: {md.get('employee')}")
        st.write(f"Completion: {md.get('completion')}% | Marks: {md.get('marks')}")
        comment = st.text_area(f"Client Feedback ({md.get('task')})", key=f"cf_{tid}")
        if st.button(f"Approve {md.get('task')}", key=f"app_{tid}"):
            md2 = {
                **md,
                "client_reviewed": True,
                "client_comments": comment,
                "client_approved_on": now(),
            }
            safe_upsert(index, md2)
            st.success(f"✅ Approved {md.get('task')}")
            safe_rerun()
