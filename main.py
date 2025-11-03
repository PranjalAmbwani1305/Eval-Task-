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
    X = np.array([[t.get("completion", 0), t.get("marks", 0)] for t in tasks])
    if len(tasks) < n_clusters:
        n_clusters = len(tasks)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    for t, c in zip(tasks, clusters):
        t["cluster"] = int(c)
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
    company = st.text_input("Company Name")
    month = st.selectbox("Select Month", [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ])

    with st.form("assign_task_form"):
        task_title = st.text_input("Task Title")
        description = st.text_area("Task Description")
        deadline = st.date_input("Deadline")
        employee = st.text_input("Assign to Employee Name")
        submit = st.form_submit_button("Assign Task")

        if submit:
            if company and task_title and employee:
                task_id = str(uuid.uuid4())
                index.upsert(
                    vectors=[{
                        "id": task_id,
                        "values": random_vector(),
                        "metadata": safe_metadata({
                            "company": company,
                            "month": month,
                            "task": task_title,
                            "description": description,
                            "deadline": str(deadline),
                            "employee": employee,
                            "completion": 0.0,
                            "marks": 0.0,
                            "status": "Assigned",
                            "reviewed": False,
                            "assigned_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                    }]
                )
                st.success(f"Task '{task_title}' assigned to {employee}")
            else:
                st.error("Please fill all fields before assigning.")

    if st.button("View All Assigned Tasks"):
        res = index.query(
            vector=random_vector(),
            top_k=300,
            include_metadata=True,
            filter={"company": {"$eq": company}, "month": {"$eq": month}}
        )
        if not res.matches:
            st.warning("No tasks found.")
        else:
            tasks = [m.metadata for m in res.matches]
            clustered = cluster_tasks(tasks)
            for t in clustered:
                st.write(
                    f"Task: {t.get('task')} | Employee: {t.get('employee')} | "
                    f"Completion: {t.get('completion')}% | Cluster: {t.get('cluster')}"
                )

# ------------------------------------------------------
# TEAM MEMBER SECTION
# ------------------------------------------------------
elif role == "Team Member":
    st.header("Team Member Section")
    company = st.text_input("Company Name")
    employee = st.text_input("Your Name")

    # Load Tasks
    if st.button("Load My Tasks"):
        if not company or not employee:
            st.error("Please enter both company and your name.")
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
                st.session_state["tasks_loaded"] = [
                    (m.id, m.metadata, m.values) for m in res.matches
                ]

    # Display tasks
    if "tasks_loaded" in st.session_state:
        tasks = st.session_state["tasks_loaded"]

        # Group by month
        tasks_by_month = {}
        for match_id, md, match_values in tasks:
            m = md.get("month", "Unknown")
            tasks_by_month.setdefault(m, []).append((match_id, md, match_values))

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

                st.markdown(f"**Task:** {task}")
                st.markdown(f"**Description:** {desc}")
                st.markdown(f"**Deadline:** {deadline}")
                st.markdown(f"**Status:** {status}")

                # Dynamic progress bar color
                progress_color = "red" if completion < 50 else ("orange" if completion < 80 else "green")
                st.progress(completion / 100, text=f"{completion:.0f}% complete")
                st.markdown(f"<span style='color:{progress_color}; font-weight:bold;'>Completion: {completion:.0f}%</span>", unsafe_allow_html=True)

                new_completion = st.slider(
                    f"Update Completion for {task}",
                    0, 100, int(completion),
                    key=f"comp_{match_id}"
                )

                if st.button(f"Submit Progress ({task})", key=f"submit_{match_id}"):
                    marks = float(lin_reg.predict([[new_completion]])[0])
                    status_pred = log_reg.predict([[new_completion]])[0]
                    status_text = "On Track" if status_pred == 1 else "Delayed"

                    updated_md = safe_metadata({
                        **md,
                        "completion": float(new_completion),
                        "marks": marks,
                        "status": status_text,
                        "reviewed": False,
                        "submitted_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })

                    index.upsert(vectors=[{
                        "id": match_id,
                        "values": match_values if match_values else random_vector(),
                        "metadata": updated_md
                    }])

                    # Update local session state immediately
                    for i, (tid, old_md, vals) in enumerate(st.session_state["tasks_loaded"]):
                        if tid == match_id:
                            st.session_state["tasks_loaded"][i] = (tid, updated_md, vals)
                            break

                    st.success(f"Progress for '{task}' updated successfully at {updated_md['submitted_on']}.")

# ------------------------------------------------------
# CLIENT SECTION
# ------------------------------------------------------
elif role == "Client":
    st.header("Client Section")
    company = st.text_input("Company Name")

    if st.button("View Approved Tasks"):
        if company:
            res = index.query(
                vector=random_vector(),
                top_k=300,
                include_metadata=True,
                filter={"company": {"$eq": company}, "reviewed": {"$eq": True}}
            )

            if not res.matches:
                st.warning("No approved tasks found.")
            else:
                st.subheader(f"Approved Tasks for {company}")
                for match in res.matches:
                    md = match.metadata or {}
                    st.write(
                        f"Employee: {md.get('employee','?')} | Task: {md.get('task','?')} | "
                        f"Completion: {md.get('completion',0)}% | Marks: {md.get('marks',0):.2f} | "
                        f"Status: {md.get('status','?')} | Submitted: {md.get('submitted_on','N/A')}"
                    )
        else:
            st.error("Enter company name first.")
