import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import uuid
from datetime import datetime
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.cluster import KMeans

# ------------------------------------------------------
# Step 1: Initialize Pinecone (Vector Database)
# ------------------------------------------------------
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

# ------------------------------------------------------
# Step 2: Machine Learning Models
# ------------------------------------------------------
lin_reg = LinearRegression()
lin_reg.fit([[0], [100]], [0, 5])

log_reg = LogisticRegression()
log_reg.fit([[0], [50], [100]], [0, 0, 1])

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(["good work", "excellent", "needs improvement", "bad performance"])
y_train = [1, 1, 0, 0]
svm_clf = SVC()
svm_clf.fit(X_train, y_train)

# ------------------------------------------------------
# Step 3: Helper Functions
# ------------------------------------------------------
def random_vector(dim=dimension):
    return np.random.rand(dim).tolist()

def safe_metadata(md: dict):
    clean = {}
    for k, v in md.items():
        if isinstance(v, (np.generic,)):
            v = v.item()
        clean[k] = v
    return clean

def cluster_tasks(tasks, n_clusters=3):
    if not tasks:
        return []
    X = np.array([[t.get('completion', 0), t.get('marks', 0)] for t in tasks])
    n_clusters = min(n_clusters, len(tasks))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    for task, cluster in zip(tasks, clusters):
        task['cluster'] = int(cluster)
    return tasks

# ------------------------------------------------------
# Step 4: Streamlit App
# ------------------------------------------------------
st.title("AI-Powered Task Completion & Review System")
role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client"])

# ------------------------------------------------------
# MANAGER SECTION
# ------------------------------------------------------
if role == "Manager":
    st.header("Manager Section")
    action = st.radio("Select Action", ["Assign New Tasks", "Review Submitted Tasks"])

    # ---------------- Assign New Tasks ----------------
    if action == "Assign New Tasks":
        company = st.text_input("Company Name")
        employee = st.text_input("Employee Name")
        task = st.text_input("Task Title")
        description = st.text_area("Task Description")
        deadline = st.date_input("Deadline")
        month = st.selectbox("Month", [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ])

        if st.button("Assign Task"):
            if not all([company, employee, task, description, deadline, month]):
                st.error("Please fill all fields before assigning a task.")
            else:
                task_id = str(uuid.uuid4())
                index.upsert(vectors=[{
                    "id": task_id,
                    "values": random_vector(),
                    "metadata": safe_metadata({
                        "company": company,
                        "employee": employee,
                        "task": task,
                        "description": description,
                        "deadline": str(deadline),
                        "month": month,
                        "completion": 0.0,
                        "marks": 0.0,
                        "status": "Assigned",
                        "reviewed": False
                    })
                }])
                st.success(f"Task '{task}' assigned to {employee} for {month}.")

    # ---------------- Review Submitted Tasks ----------------
    elif action == "Review Submitted Tasks":
        all_res = index.query(vector=random_vector(), top_k=200, include_metadata=True)
        companies = list(set([m.metadata.get("company", "?") for m in all_res.matches])) if all_res.matches else []
        if companies:
            company = st.selectbox("Select Company", companies)
        else:
            st.warning("No companies found.")
            company = None

        if company and st.button("Load Pending Tasks"):
            res = index.query(
                vector=random_vector(),
                top_k=200,
                include_metadata=True,
                include_values=True,
                filter={"company": {"$eq": company}, "reviewed": {"$eq": False}}
            )
            st.session_state["pending"] = res.matches if res.matches else []

        if "pending" in st.session_state and st.session_state["pending"]:
            tasks = [m.metadata for m in st.session_state["pending"]]
            clustered = cluster_tasks(tasks)
            st.subheader("Task Clusters")
            for t in clustered:
                st.write(f"{t.get('employee')} → {t.get('task')} | Cluster {t.get('cluster')} | Completion: {t.get('completion')}%")

            for match in st.session_state["pending"]:
                md = match.metadata or {}
                emp = md.get("employee", "?")
                task = md.get("task", "?")
                completion = float(md.get("completion", 0))
                st.write(f"Employee: {emp} | Task: {task} | Current Completion: {completion}%")

                manager_completion = st.slider(
                    f"Adjust Completion for {emp} - {task}", 0, 100, int(completion), key=f"adj_{match.id}"
                )

                comments = st.text_area(f"Manager Comments ({emp} - {task})", key=f"c_{match.id}")
                marks = float(lin_reg.predict([[manager_completion]])[0])
                status_pred = log_reg.predict([[manager_completion]])[0]
                status_text = "On Track" if status_pred == 1 else "Delayed"

                sentiment_text = "N/A"
                if comments:
                    try:
                        X_new = vectorizer.transform([comments])
                        sentiment = svm_clf.predict(X_new)[0]
                        sentiment_text = "Positive" if sentiment == 1 else "Negative"
                    except Exception:
                        sentiment_text = "N/A"

                if st.button(f"Save Review ({emp} - {task})", key=f"s_{match.id}"):
                    index.upsert(vectors=[{
                        "id": match.id,
                        "values": match.values if hasattr(match, "values") else random_vector(),
                        "metadata": safe_metadata({
                            **md,
                            "completion": float(manager_completion),
                            "marks": marks,
                            "status": status_text,
                            "reviewed": True,
                            "comments": comments,
                            "sentiment": sentiment_text
                        })
                    }])
                    st.success(f"Review saved for {emp} - {task}")

# ------------------------------------------------------
# TEAM MEMBER SECTION
# ------------------------------------------------------
elif role == "Team Member":
    st.header("Team Member Section")
    company = st.text_input("Company Name")
    employee = st.text_input("Your Name")

    if st.button("Load My Tasks"):
        if not company or not employee:
            st.error("Please enter both company name and your name.")
        else:
            res = index.query(
                vector=random_vector(),
                top_k=300,
                include_metadata=True,
                include_values=True,
                filter={"company": {"$eq": company}, "employee": {"$eq": employee}}
            )

            if not res.matches:
                st.warning("No tasks found for this employee.")
            else:
                tasks_by_month = {}
                for match in res.matches:
                    md = match.metadata or {}
                    month = md.get("month", "Unknown")
                    tasks_by_month.setdefault(month, []).append((match.id, md, match.values))

                months = sorted(tasks_by_month.keys())
                month_selected = st.selectbox("Select Month", months)

                if month_selected:
                    st.subheader(f"Your Tasks for {month_selected}")

                    for match_id, md, match_values in tasks_by_month[month_selected]:
                        task = md.get("task", "?")
                        desc = md.get("description", "No description")
                        deadline = md.get("deadline", "N/A")
                        completion = float(md.get("completion", 0))
                        status = md.get("status", "Assigned")

                        st.write(f"Task: {task}")
                        st.write(f"Description: {desc}")
                        st.write(f"Deadline: {deadline}")
                        st.write(f"Current Status: {status}")
                        st.write(f"Completion: {completion}%")

                        new_completion = st.slider(
                            f"Update Completion for {task}",
                            0, 100, int(completion),
                            key=f"comp_{match_id}"
                        )

                        if st.button(f"Submit Progress ({task})", key=f"submit_{match_id}"):
                            marks = float(lin_reg.predict([[new_completion]])[0])
                            status_pred = log_reg.predict([[new_completion]])[0]
                            status_text = "On Track" if status_pred == 1 else "Delayed"

                            index.upsert(vectors=[{
                                "id": match_id,
                                "values": match_values if match_values else random_vector(),
                                "metadata": safe_metadata({
                                    **md,
                                    "completion": float(new_completion),
                                    "marks": marks,
                                    "status": status_text,
                                    "reviewed": False
                                })
                            }])
                            st.success(f"Progress for '{task}' updated successfully.")

# ------------------------------------------------------
# CLIENT SECTION
# ------------------------------------------------------
elif role == "Client":
    st.header("Client Section")
    company = st.text_input("Company Name")

    if st.button("View All Tasks"):
        if company:
            res = index.query(
                vector=random_vector(),
                top_k=200,
                include_metadata=True,
                filter={"company": {"$eq": company}}
            )

            if res.matches:
                st.subheader(f"All Tasks for {company}")
                for match in res.matches:
                    md = match.metadata or {}
                    reviewed = md.get("reviewed", False)
                    employee = md.get("employee", "?")
                    task = md.get("task", "?")
                    completion = md.get("completion", 0)
                    marks = md.get("marks", 0)
                    status = md.get("status", "Unknown")
                    sentiment = md.get("sentiment", "N/A")

                    if reviewed:
                        st.success(
                            f"{employee} | Task: {task} → {completion}% "
                            f"(Marks: {marks:.2f}) | Status: {status} | Sentiment: {sentiment}"
                        )
                    else:
                        st.warning(
                            f"{employee} | Task: {task} → {completion}% "
                            f"(Marks: {marks:.2f}) | Status: Pending Manager Review"
                        )
            else:
                st.warning("No tasks found for this company.")
        else:
            st.error("Enter company name")
