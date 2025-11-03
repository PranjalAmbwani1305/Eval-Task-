import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import uuid
from datetime import date
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.cluster import KMeans

# ------------------------------------------------------
# Step 1: Initialize Pinecone
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
# Step 2: ML Models
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
    n_clusters = min(len(tasks), n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    for t, c in zip(tasks, clusters):
        t['cluster'] = int(c)
    return tasks

def month_name(dt):
    return dt.strftime("%B %Y")  # e.g. "November 2025"

# ------------------------------------------------------
# Step 4: Streamlit App
# ------------------------------------------------------
st.title("AI-Powered Month-wise Task Management System")

role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client"])

# ------------------------------------------------------
# MANAGER SECTION
# ------------------------------------------------------
if role == "Manager":
    st.header("Manager Section")

    tab_assign, tab_review = st.tabs(["Assign New Tasks", "Review Submitted Tasks"])

    # ---------------------- Assign New Tasks ----------------------
    with tab_assign:
        st.subheader("Assign New Task")
        company = st.text_input("Company Name")
        employee = st.text_input("Employee Name")
        task = st.text_input("Task Title")
        description = st.text_area("Task Description")
        deadline = st.date_input("Deadline", value=date.today())
        month = month_name(deadline)

        if st.button("Assign Task"):
            if company and employee and task:
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
            else:
                st.error("Please fill all required fields.")

    # ---------------------- Review Submitted Tasks ----------------------
    with tab_review:
        st.subheader("Review Team Submissions")

        all_res = index.query(vector=random_vector(), top_k=200, include_metadata=True)
        companies = list(set([m.metadata["company"] for m in all_res.matches])) if all_res.matches else []

        if companies:
            company = st.selectbox("Select Company", companies)
        else:
            st.warning("No companies found.")
            company = None

        if company:
            all_months = list(set([m.metadata.get("month") for m in all_res.matches if m.metadata.get("company") == company]))
            month_selected = st.selectbox("Select Month", sorted(all_months))

        if company and month_selected and st.button("Load Pending Reviews"):
            res = index.query(
                vector=random_vector(),
                top_k=200,
                include_metadata=True,
                include_values=True,
                filter={"company": {"$eq": company}, "month": {"$eq": month_selected}, "reviewed": {"$eq": False}, "status": {"$ne": "Assigned"}}
            )

            if not res.matches:
                st.success(f"All tasks for {company} in {month_selected} are reviewed or pending submission.")
            else:
                st.session_state["pending_reviews"] = res.matches

        if "pending_reviews" in st.session_state and st.session_state["pending_reviews"]:
            tasks_metadata = [m.metadata for m in st.session_state["pending_reviews"]]
            clustered_tasks = cluster_tasks(tasks_metadata)

            st.write(f"Task Clusters for {month_selected}:")
            for t in clustered_tasks:
                st.write(f"{t.get('task','?')} → Cluster {t.get('cluster','?')} "
                         f"| Completion: {t.get('completion',0)}% | Marks: {t.get('marks',0):.2f}")

            for match in st.session_state["pending_reviews"]:
                md = match.metadata or {}
                emp = md.get("employee", "?")
                task = md.get("task", "?")
                emp_completion = float(md.get("completion", 0))

                st.write(f"{emp} | Task: {task} | Completion: {emp_completion}%")

                manager_completion = st.slider(
                    f"Adjust Completion ({emp} - {task})",
                    0, 100, int(emp_completion),
                    key=f"adj_{match.id}"
                )

                predicted_marks = float(lin_reg.predict([[manager_completion]])[0])
                status = log_reg.predict([[manager_completion]])[0]
                status_text = "On Track" if status == 1 else "Delayed"

                comments = st.text_area(f"Manager Comments for {emp} - {task}", key=f"c_{match.id}")

                sentiment_text = "N/A"
                if comments:
                    try:
                        X_new = vectorizer.transform([comments])
                        sentiment = svm_clf.predict(X_new)[0]
                        sentiment_text = "Positive" if sentiment == 1 else "Negative"
                    except Exception:
                        sentiment_text = "N/A"

                if st.button(f"Save Review for {emp} - {task}", key=f"s_{match.id}"):
                    index.upsert(vectors=[{
                        "id": match.id,
                        "values": match.values if hasattr(match, "values") else random_vector(),
                        "metadata": safe_metadata({
                            **md,
                            "completion": float(manager_completion),
                            "marks": predicted_marks,
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
        if company and employee:
            res = index.query(
                vector=random_vector(),
                top_k=100,
                include_metadata=True,
                filter={"company": {"$eq": company}, "employee": {"$eq": employee}}
            )

            if res.matches:
                months = sorted(list(set([m.metadata.get("month") for m in res.matches])))
                month_selected = st.selectbox("Select Month", months)

                st.subheader(f"Your Tasks for {month_selected}")
                for match in res.matches:
                    md = match.metadata or {}
                    if md.get("status") == "Assigned" and md.get("month") == month_selected:
                        st.write(f"Task: {md.get('task','?')} — Deadline: {md.get('deadline','N/A')}")
                        st.write(f"{md.get('description','No description')}")
                        completion = st.slider(f"Update Completion ({md.get('task','?')})", 0, 100, 0, key=f"comp_{match.id}")

                        if st.button(f"Submit Progress ({md.get('task','?')})", key=f"submit_{match.id}"):
                            marks = lin_reg.predict([[completion]])[0]
                            status = log_reg.predict([[completion]])[0]
                            status_text = "On Track" if status == 1 else "Delayed"

                            index.upsert(vectors=[{
                                "id": match.id,
                                "values": match.values if hasattr(match, "values") else random_vector(),
                                "metadata": safe_metadata({
                                    **md,
                                    "completion": float(completion),
                                    "marks": float(marks),
                                    "status": status_text,
                                    "reviewed": False
                                })
                            }])
                            st.success(f"Progress updated for {md.get('task','?')}")
            else:
                st.warning("No tasks assigned to you.")
        else:
            st.error("Enter both company name and your name.")

# ------------------------------------------------------
# CLIENT SECTION
# ------------------------------------------------------
elif role == "Client":
    st.header("Client Section")
    company = st.text_input("Company Name")

    if st.button("View Tasks Month-wise"):
        if company:
            res = index.query(
                vector=random_vector(),
                top_k=300,
                include_metadata=True,
                filter={"company": {"$eq": company}}
            )

            if res.matches:
                months = sorted(list(set([m.metadata.get("month") for m in res.matches])))
                for mth in months:
                    st.subheader(f"{mth}")
                    for match in res.matches:
                        md = match.metadata or {}
                        if md.get("month") == mth:
                            reviewed = md.get("reviewed", False)
                            employee = md.get("employee", "?")
                            task = md.get("task", "?")
                            completion = md.get("completion", 0)
                            marks = md.get("marks", 0)
                            status = md.get("status", "Unknown")
                            sentiment = md.get("sentiment", "N/A")
                            deadline = md.get("deadline", "N/A")

                            if reviewed:
                                st.success(
                                    f"{employee} | {task} → {completion}% | Marks: {marks:.2f} | {status} | Sentiment: {sentiment} | Deadline: {deadline}"
                                )
                            else:
                                st.warning(
                                    f"{employee} | {task} → {completion}% | {status} | Deadline: {deadline} | Pending Review"
                                )
            else:
                st.warning("No tasks found for this company.")
        else:
            st.error("Enter company name.")
