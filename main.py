# ------------------------------
# Command: streamlit run app.py
# ------------------------------
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import uuid
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import pandas as pd

# ------------------------------
# Step 1: Initialize Pinecone DB
# ------------------------------
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index_name = "task"
dimension = 1024

if index_name not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# ------------------------------
# Step 2: ML Models
# ------------------------------
lin_reg = LinearRegression()
lin_reg.fit([[0], [100]], [0, 5])

log_reg = LogisticRegression()
log_reg.fit([[0], [50], [100]], [0, 0, 1])

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(["good work", "excellent", "needs improvement", "bad performance"])
y_train = [1, 1, 0, 0]
svm_clf = SVC()
svm_clf.fit(X_train, y_train)

# ------------------------------
# Step 3: Helper Functions
# ------------------------------
def random_vector(dim=dimension):
    return np.random.rand(dim).tolist()

def safe_metadata(md: dict):
    clean = {}
    for k, v in md.items():
        if isinstance(v, (np.generic,)):
            v = v.item()
        clean[k] = v
    return clean

def assign_task_auto(tasks, employees):
    workload = {e:0 for e in employees}
    assignment = []
    for task in tasks:
        emp = min(workload, key=workload.get)
        task['assigned_to'] = emp
        workload[emp] += 1
        assignment.append(task)
    return assignment

def cluster_tasks(tasks, n_clusters=3):
    X = np.array([[t['completion'], t['marks']] for t in tasks])
    if len(tasks) < n_clusters:
        n_clusters = len(tasks)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    for task, cluster in zip(tasks, clusters):
        task['cluster'] = int(cluster)
    return tasks

def classify_performance(tasks):
    perf = {}
    for task in tasks:
        emp = task['employee']
        score = task['marks']
        if emp not in perf:
            perf[emp] = []
        perf[emp].append(score)
    classification = {}
    for emp, scores in perf.items():
        avg = np.mean(scores)
        if avg >= 4:
            classification[emp] = "High"
        elif avg >= 2.5:
            classification[emp] = "Medium"
        else:
            classification[emp] = "Low"
    return classification

# ------------------------------
# Step 4: Streamlit UI
# ------------------------------
st.set_page_config(page_title="AI Task & Performance Hub", layout="wide")
st.title("💼 AI-Driven Employee Performance & Task Management Suite")

role = st.sidebar.selectbox("Login as", ["Team Member", "Manager", "Client"])

# ------------------------------
# TEAM MEMBER
# ------------------------------
if role == "Team Member":
    st.header("👩‍💻 Team Member Portal")
    company = st.text_input("🏢 Company Name")
    employee = st.text_input("👤 Your Name")
    task = st.text_input("📝 Task Title")
    completion = st.slider("✅ Completion %", 0, 100, 0)

    if st.button("📩 Submit Task"):
        if company and employee and task:
            marks = lin_reg.predict([[completion]])[0]
            status = log_reg.predict([[completion]])[0]
            status_text = "On Track" if status == 1 else "Delayed"
            task_id = str(uuid.uuid4())

            index.upsert(vectors=[{
                "id": task_id,
                "values": random_vector(),
                "metadata": safe_metadata({
                    "company": company,
                    "employee": employee,
                    "task": task,
                    "completion": float(completion),
                    "marks": float(marks),
                    "status": status_text,
                    "reviewed": False
                })
            }])
            st.success(f"✅ Task '{task}' submitted by {employee}")
        else:
            st.error("❌ Fill all fields before submitting")

# ------------------------------
# CLIENT VIEW
# ------------------------------
elif role == "Client":
    st.header("👨‍💼 Client Review Portal")
    company = st.text_input("🏢 Company Name")
    if st.button("🔍 View Approved Tasks") and company:
        res = index.query(
            vector=random_vector(),
            top_k=100,
            include_metadata=True,
            filter={"company": {"$eq": company}, "reviewed": {"$eq": True}}
        )
        if res.matches:
            st.subheader(f"📌 Approved Tasks for {company}")
            for match in res.matches:
                md = match.metadata or {}
                st.write(
                    f"👤 {md.get('employee','?')} | **{md.get('task','?')}** → {md.get('completion',0)}% "
                    f"(Marks: {md.get('marks',0):.2f}) | Status: {md.get('status','?')} | Sentiment: {md.get('sentiment','N/A')}"
                )
        else:
            st.warning("⚠️ No approved tasks found.")

# ------------------------------
# MANAGER DASHBOARD (Merged)
# ------------------------------
elif role == "Manager":
    st.header("🧭 Unified Manager Dashboard")
    st.markdown("Monitor, Manage, and Motivate — All in One Pane")

    # Fetch all task data
    all_res = index.query(vector=random_vector(), top_k=200, include_metadata=True)
    tasks = [m.metadata for m in all_res.matches] if all_res.matches else []
    if not tasks:
        st.warning("⚠️ No task data available yet.")
    else:
        df = pd.DataFrame(tasks)
        if 'employee' in df.columns:
            # Task Summary
            st.subheader("📊 Task Summary")
            total = len(df)
            reviewed = len(df[df["reviewed"] == True])
            pending = total - reviewed
            on_track = len(df[df["status"] == "On Track"])
            delayed = len(df[df["status"] == "Delayed"])
            st.write(f"✅ Total Tasks: {total} | 🕒 Pending: {pending} | 🚀 On Track: {on_track} | ⚠️ Delayed: {delayed}")

            # Team Performance Heatmap
            st.subheader("🔥 Team Performance Heatmap")
            perf = classify_performance(tasks)
            perf_df = pd.DataFrame(list(perf.items()), columns=["Employee", "Category"])
            st.dataframe(perf_df)

            # AI Alerts
            st.subheader("⚡ AI Alerts")
            alerts = []
            if delayed > (total * 0.4):
                alerts.append("High number of delayed tasks! Review workload distribution.")
            if len(set(df['employee'])) < 2:
                alerts.append("Single contributor alert! Possible workload imbalance.")
            if not alerts:
                st.success("✅ No major alerts detected.")
            else:
                for a in alerts:
                    st.warning(a)

            # Goal Tracker
            st.subheader("🎯 Goal Tracker")
            goal = st.slider("Set Monthly Target (in %)", 50, 100, 80)
            avg_completion = df["completion"].mean()
            st.progress(int(avg_completion))
            st.write(f"📈 Current Average: {avg_completion:.2f}% of Target {goal}%")

            # 360° Feedback Section
            st.subheader("💬 360° Feedback & Sentiment")
            feedback = st.text_area("Enter employee feedback (self/peer/manager):")
            if st.button("🔍 Analyze Feedback"):
                if feedback.strip():
                    X_new = vectorizer.transform([feedback])
                    sentiment = svm_clf.predict(X_new)[0]
                    result = "Positive" if sentiment == 1 else "Negative"
                    st.write(f"🧠 AI Sentiment: **{result}**")
                else:
                    st.error("Enter feedback first.")

            # Managerial Actions
            st.subheader("⚙️ Managerial Actions & Approvals")
            st.write("Quick Actions for performance and task control.")
            act = st.selectbox("Choose Action", [
                "Reassign Tasks", "Approve Deliverables", "Send Appreciation", "Issue Warning"
            ])
            emp = st.selectbox("Select Employee", list(set(df["employee"].dropna())))
            msg = st.text_input("Message or Note")

            if st.button("🚀 Execute Action"):
                st.success(f"✅ Action '{act}' executed for {emp}. Message: {msg or 'N/A'}")

            st.success("Managerial Dashboard Updated ✅")
