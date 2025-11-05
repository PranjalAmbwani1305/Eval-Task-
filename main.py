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
import smtplib
from email.mime.text import MIMEText

# ----------------------------------
# CONFIG & SAFE RERUN HANDLER
# ----------------------------------
st.set_page_config(page_title="AI Task System", layout="wide")
st.title("🚀 AI-Powered Task Management System")

def safe_rerun():
    """Compatible rerun for all Streamlit versions."""
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

# ----------------------------------
# PINECONE INIT
# ----------------------------------
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
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

# ----------------------------------
# HELPERS
# ----------------------------------
def now(): 
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def stable_vec(text: str):
    np.random.seed(abs(hash(text)) % (2**32))
    return np.random.rand(DIMENSION).tolist()

def safe_meta(md):
    clean = {}
    for k, v in md.items():
        if isinstance(v, (datetime, date)):
            v = v.isoformat()
        elif v is None:
            v = ""
        clean[k] = v
    return clean

def safe_upsert(index, md):
    """Handles Pinecone upsert safely"""
    try:
        if "_id" not in md or not md["_id"]:
            md["_id"] = str(uuid.uuid4())
        vec = stable_vec(md.get("task", "default"))
        if len(vec) != DIMENSION:
            st.error("Vector dimension mismatch.")
            return
        index.upsert([{"id": str(md["_id"]), "values": vec, "metadata": safe_meta(md)}])
    except Exception as e:
        st.error(f"Pinecone upsert failed: {e}")

def fetch_all():
    try:
        res = index.query(vector=stable_vec("all"), top_k=1000, include_metadata=True)
        if not res.matches:
            return pd.DataFrame()
        rows = []
        for m in res.matches:
            md = m.metadata or {}
            md["_id"] = m.id
            rows.append(md)
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Error fetching from Pinecone: {e}")
        return pd.DataFrame()

# ----------------------------------
# EMAIL UTILITY
# ----------------------------------
def send_email(to_email, subject, body):
    try:
        sender_email = "your_email@example.com"
        sender_password = st.secrets["EMAIL_PASSWORD"]
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = to_email

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, to_email, msg.as_string())
        st.success(f"Notification sent to {to_email}")
    except Exception as e:
        st.warning(f"Email failed: {e}")

# ----------------------------------
# MODELS
# ----------------------------------
lin_reg = LinearRegression()
lin_reg.fit([[0], [50], [100]], [0, 2.5, 5])
log_reg = LogisticRegression()
log_reg.fit([[0], [40], [80], [100]], [0, 0, 1, 1])
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

# ----------------------------------
# ROLE SELECTION
# ----------------------------------
role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client", "Admin"])
current_month = datetime.now().strftime("%B %Y")

# ----------------------------------
# MANAGER
# ----------------------------------
if role == "Manager":
    st.header("👔 Manager Dashboard")
    tab1, tab2, tab3 = st.tabs(["Assign Task", "Boss Review", "Manager Actions & 360°"])

    # --- Assign Task ---
    with tab1:
        with st.form("assign"):
            company = st.text_input("Company Name")
            employee = st.text_input("Employee Name")
            task = st.text_input("Task Title")
            desc = st.text_area("Description")
            deadline = st.date_input("Deadline", value=date.today())
            month = st.text_input("Month", value=current_month)
            email = st.text_input("Employee Email (optional)")
            submit = st.form_submit_button("Assign Task")

            if submit and company and employee and task:
                tid = str(uuid.uuid4())
                md = {
                    "_id": tid,
                    "company": company,
                    "employee": employee,
                    "task": task,
                    "description": desc,
                    "deadline": deadline.isoformat(),
                    "month": month,
                    "completion": 0,
                    "marks": 0,
                    "status": "Assigned",
                    "assigned_on": now(),
                    "email": email
                }
                safe_upsert(index, md)
                st.success(f"Task '{task}' assigned to {employee}")
                if email:
                    send_email(email, f"New Task Assigned: {task}",
                               f"Dear {employee},\n\nYou have been assigned a new task: {task}.\n\nBest,\nManagement")
                safe_rerun()

    # --- Boss Review ---
    with tab2:
        df = fetch_all()
        if df.empty:
            st.warning("No tasks found.")
        else:
            df["completion"] = pd.to_numeric(df["completion"], errors="coerce").fillna(0)
            for _, r in df.iterrows():
                st.subheader(r["task"])
                st.write(f"Employee: {r.get('employee', 'N/A')} | Completion: {r.get('completion', 0)}%")
                adj = st.slider(f"Adjust Completion for {r['task']}", 0, 100, int(r["completion"]), key=f"adj_{r['_id']}")
                comments = st.text_area(f"Boss Comments", key=f"cmt_{r['_id']}")
                approve = st.radio(f"Approve {r['task']}?", ["Yes", "No"], key=f"app_{r['_id']}")
                if st.button(f"Finalize {r['task']}", key=f"fin_{r['_id']}"):
                    sentiment_val = int(svm_clf.predict(vectorizer.transform([comments]))[0])
                    sentiment = "Positive" if sentiment_val == 1 else "Negative"
                    md = {**r, "completion": adj, "marks": float(lin_reg.predict([[adj]])[0]),
                          "manager_comments": comments, "sentiment": sentiment,
                          "approved_by_boss": approve == "Yes", "reviewed_on": now()}
                    safe_upsert(index, md)
                    st.success(f"Task '{r['task']}' reviewed ({sentiment}).")
                    safe_rerun()

    # --- Manager Actions & 360° ---
    with tab3:
        st.subheader("📊 360° Performance Overview")
        search_emp = st.text_input("Filter by Employee")
        df = fetch_all()
        if not df.empty:
            if search_emp:
                df = df[df["employee"].str.contains(search_emp, case=False, na=False)]
            if df.empty:
                st.warning("No records found.")
            else:
                st.dataframe(df[["employee", "task", "completion", "marks", "sentiment", "status"]])
                if st.button("Generate 360° Summary"):
                    avg_perf = df.groupby("employee")[["marks", "completion"]].mean().reset_index()
                    st.plotly_chart(px.bar(avg_perf, x="employee", y="marks", title="Average Marks"))
                    st.plotly_chart(px.line(df, x="employee", y="completion", color="sentiment",
                                            title="Completion vs Sentiment"))
                    st.success("360° Performance Summary Generated ✅")

# ----------------------------------
# TEAM MEMBER
# ----------------------------------
elif role == "Team Member":
    st.header("🧑‍💻 Team Member Progress")
    company = st.text_input("Company")
    employee = st.text_input("Your Name")

    if st.button("Load Tasks"):
        res = index.query(vector=stable_vec(employee), top_k=500, include_metadata=True)
        st.session_state["tasks"] = [(m.id, m.metadata) for m in res.matches or []]
        st.success(f"Loaded {len(st.session_state['tasks'])} tasks.")

    for tid, md in st.session_state.get("tasks", []):
        st.subheader(md.get("task"))
        new = st.slider(f"Completion for {md.get('task')}", 0, 100, int(md.get("completion", 0)))
        if st.button(f"Submit {md.get('task')}", key=tid):
            marks = float(lin_reg.predict([[new]])[0])
            track = "On Track" if log_reg.predict([[new]])[0] == 1 else "Delayed"
            md2 = {**md, "completion": new, "marks": marks, "status": track, "submitted_on": now()}
            safe_upsert(index, md2)
            st.success(f"Updated {md.get('task')} ({track})")
            safe_rerun()

# ----------------------------------
# CLIENT
# ----------------------------------
elif role == "Client":
    st.header("💼 Client Review")
    company = st.text_input("Company Name")
    if st.button("Load Completed"):
        res = index.query(vector=stable_vec(company), top_k=500, include_metadata=True)
        st.session_state["ctasks"] = [(m.id, m.metadata) for m in res.matches or []]
        st.success(f"Loaded {len(st.session_state['ctasks'])} tasks.")
    for tid, md in st.session_state.get("ctasks", []):
        st.subheader(md.get("task"))
        comment = st.text_area(f"Feedback for {md.get('task')}", key=f"c_{tid}")
        if st.button(f"Approve {md.get('task')}", key=f"app_{tid}"):
            md2 = {**md, "client_reviewed": True, "client_comments": comment, "client_approved_on": now()}
            safe_upsert(index, md2)
            st.success(f"Approved {md.get('task')}")
            safe_rerun()
