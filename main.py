import streamlit as st
from datetime import datetime, date
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from textblob import TextBlob
import uuid

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Task Management System", layout="wide")
st.title("AI Task Management System")

# -----------------------------
# LOCAL STORAGE (Simulated Database)
# -----------------------------
if "tasks" not in st.session_state:
    st.session_state["tasks"] = []

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# -----------------------------
# MACHINE LEARNING MODELS
# -----------------------------
lin_reg = LinearRegression()
lin_reg.fit([[0], [50], [100]], [0, 2.5, 5])

log_reg = LogisticRegression()
log_reg.fit([[0], [50], [100]], [0, 0, 1])

rf = RandomForestClassifier()
rf.fit(np.array([[10, 2], [50, 1], [90, 0], [100, 0]]), [0, 1, 0, 0])

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(["great job", "needs improvement", "excellent work", "bad performance", "average"])
svm_clf = SVC()
svm_clf.fit(X_train, [1, 0, 1, 0, 0])

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def fetch_tasks(company=None, employee=None):
    df = pd.DataFrame(st.session_state["tasks"])
    if df.empty:
        return df
    if company:
        df = df[df["company"] == company]
    if employee:
        df = df[df["employee"] == employee]
    return df

def save_task(task):
    df = pd.DataFrame(st.session_state["tasks"])
    existing = df[df["_id"] == task["_id"]]
    if not existing.empty:
        st.session_state["tasks"] = [t if t["_id"] != task["_id"] else task for t in st.session_state["tasks"]]
    else:
        st.session_state["tasks"].append(task)

# -----------------------------
# ROLE SELECTION
# -----------------------------
role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client"])
current_month = datetime.now().strftime("%B %Y")

# -----------------------------
# MANAGER SECTION
# -----------------------------
if role == "Manager":
    st.header("Manager Dashboard")
    tab1, tab2, tab3, tab4 = st.tabs(["Assign Task", "Review Tasks", "360° Analytics", "Managerial Actions & Approvals"])

    # Assign Task
    with tab1:
        with st.form("assign_task"):
            company = st.text_input("Company Name")
            employee = st.text_input("Employee Name")
            task = st.text_input("Task Title")
            desc = st.text_area("Task Description")
            deadline = st.date_input("Deadline", value=date.today())
            submit = st.form_submit_button("Assign Task")
            if submit and company and employee and task:
                tid = str(uuid.uuid4())
                task_data = {
                    "_id": tid,
                    "company": company,
                    "employee": employee,
                    "task": task,
                    "description": desc,
                    "deadline": deadline.isoformat(),
                    "month": current_month,
                    "completion": 0,
                    "marks": 0,
                    "status": "Assigned",
                    "reviewed": False,
                    "assigned_on": now()
                }
                save_task(task_data)
                st.success(f"Task '{task}' assigned to {employee}")

    # Review Tasks
    with tab2:
        df = fetch_tasks()
        if df.empty:
            st.warning("No tasks available.")
        else:
            for _, r in df.iterrows():
                st.markdown(f"### {r['task']}")
                completion = st.slider(f"Update completion for {r['employee']}", 0, 100, int(r["completion"]))
                comments = st.text_area(f"Manager Comments for {r['task']}", key=f"c_{r['_id']}")
                approve = st.radio(f"Approve {r['task']}?", ["Yes", "No"], key=f"a_{r['_id']}")
                if st.button(f"Finalize {r['task']}", key=f"f_{r['_id']}"):
                    sentiment_val = int(svm_clf.predict(vectorizer.transform([comments]))[0])
                    sentiment = "Positive" if sentiment_val == 1 else "Negative"
                    marks = float(lin_reg.predict([[completion]])[0])
                    task_data = {
                        **r,
                        "completion": completion,
                        "marks": marks,
                        "status": "Completed" if approve == "Yes" else "Pending Review",
                        "reviewed": True,
                        "sentiment": sentiment,
                        "reviewed_on": now()
                    }
                    save_task(task_data)
                    st.success(f"Task '{r['task']}' reviewed successfully ({sentiment})")

    # 360° Analytics
    with tab3:
        df = fetch_tasks()
        if not df.empty:
            df["marks"] = pd.to_numeric(df["marks"], errors="coerce")
            df["completion"] = pd.to_numeric(df["completion"], errors="coerce")
            if len(df) >= 3:
                kmeans = KMeans(n_clusters=3, n_init=10).fit(df[["completion", "marks"]].fillna(0))
                df["cluster"] = kmeans.labels_
                st.write("Employee Clustering Summary:")
                st.dataframe(df[["employee", "task", "completion", "marks", "cluster"]])
            avg_marks = df["marks"].mean()
            st.info(f"Average Performance Score: {avg_marks:.2f}")

            # Reassignment Option
            st.subheader("Task Reassignment")
            tsel = st.selectbox("Select Task to Reassign", df["task"].unique())
            new_emp = st.text_input("Reassign to Employee Name")
            if st.button("Reassign Task"):
                df.loc[df["task"] == tsel, "employee"] = new_emp
                st.session_state["tasks"] = df.to_dict("records")
                st.success(f"Task '{tsel}' reassigned to {new_emp}")

    # Managerial Actions & Approvals
    with tab4:
        st.subheader("Managerial Actions & Approvals")
        act = st.selectbox("Select Action", ["Appreciation Note", "Warning", "Approve Overtime", "Generate 360° Summary"])
        emp = st.text_input("Employee Name for Action")
        if st.button("Execute Action"):
            st.success(f"Action '{act}' executed for employee: {emp}")

# -----------------------------
# TEAM MEMBER SECTION
# -----------------------------
elif role == "Team Member":
    st.header("Team Member Portal")
    tab1, tab2 = st.tabs(["My Tasks", "AI Feedback Summarization"])

    # My Tasks
    with tab1:
        company = st.text_input("Company Name")
        employee = st.text_input("Your Name")
        if st.button("Load Tasks"):
            df = fetch_tasks(company, employee)
            st.session_state["my_tasks"] = df.to_dict("records")
            st.success(f"Loaded {len(df)} tasks.")

        for r in st.session_state.get("my_tasks", []):
            st.subheader(r["task"])
            new_progress = st.slider(f"Progress for {r['task']}", 0, 100, int(r["completion"]))
            if st.button(f"Submit {r['task']}", key=f"s_{r['_id']}"):
                marks = float(lin_reg.predict([[new_progress]])[0])
                status = "On Track" if log_reg.predict([[new_progress]])[0] == 1 else "Delayed"
                updated_task = {**r, "completion": new_progress, "marks": marks, "status": status, "updated_on": now()}
                save_task(updated_task)
                st.success(f"Updated {r['task']} ({status})")

    # AI Feedback Summarization
    with tab2:
        st.subheader("AI Feedback Summarization")
        company = st.text_input("Company Name (for summary)")
        employee = st.text_input("Your Name (for summary)")
        feedback_text = st.text_area("Paste feedback text below:")
        if st.button("Analyze Feedback"):
            if feedback_text.strip():
                blob = TextBlob(feedback_text)
                polarity = blob.sentiment.polarity

                if polarity > 0.1:
                    sentiment_label = "Overall Positive Feedback"
                    sentiment_text = "The feedback shows a positive attitude, reflecting good engagement and satisfaction."
                elif polarity < -0.1:
                    sentiment_label = "Overall Negative Feedback"
                    sentiment_text = "The feedback suggests areas of improvement, such as better communication or ERP integration."
                else:
                    sentiment_label = "Neutral Feedback"
                    sentiment_text = "The feedback appears balanced without strong positive or negative tone."

                keywords = list(blob.noun_phrases)
                key_summary = " ".join(keywords).capitalize() if keywords else "No specific keywords found."
                ai_summary = f"{sentiment_text} Key insights include: {key_summary}."

                st.write(sentiment_label)
                st.write("Summary:", ai_summary)

# -----------------------------
# CLIENT SECTION
# -----------------------------
elif role == "Client":
    st.header("Client Review")
    company = st.text_input("Company Name")
    if st.button("Load Reviewed Tasks"):
        df = fetch_tasks(company)
        df = df[df["reviewed"] == True]
        if df.empty:
            st.warning("No reviewed tasks found.")
        else:
            for _, r in df.iterrows():
                st.subheader(r["task"])
                st.write(f"Employee: {r['employee']}")
                st.write(f"Completion: {r['completion']}%")
                st.write(f"Marks: {r['marks']}")
                feedback = st.text_area(f"Client Feedback for {r['task']}", key=f"cf_{r['_id']}")
                if st.button(f"Approve {r['task']}", key=f"a_{r['_id']}"):
                    updated = {**r, "client_reviewed": True, "client_feedback": feedback, "client_approved_on": now()}
                    save_task(updated)
                    st.success(f"Task '{r['task']}' approved.")
