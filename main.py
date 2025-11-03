import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import uuid
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.cluster import KMeans

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
# Step 4: Streamlit App
# ------------------------------
st.title("AI-Powered Task Completion & Review System")

role = st.sidebar.selectbox("Login as", ["Team Member", "Manager", "Client", "Admin / Debug"])

# ------------------------------
# Team Member Section
# ------------------------------
if role == "Team Member":
    st.header("Team Member Section")
    company = st.text_input("Company Name")
    employee = st.text_input("Your Name")
    task = st.text_input("Task Title")
    completion = st.slider("Completion %", 0, 100, 0)

    if st.button("Submit Task"):
        if company and employee and task:
            marks = lin_reg.predict([[completion]])[0]
            status = log_reg.predict([[completion]])[0]
            status_text = "On Track" if status == 1 else "Delayed"
            task_id = str(uuid.uuid4())

            index.upsert(
                vectors=[{
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
                }]
            )
            st.success(f" Task '{task}' submitted by {employee}")
        else:
            st.error(" Fill all fields before submitting")

# ------------------------------
# Client Section (Updated)
# ------------------------------
elif role == "Client":
    st.header("Client Section")
    company = st.text_input("Company Name")

    if st.button("View All Tasks") and company:
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
                        f"{employee} | Task: **{task}** → {completion}% "
                        f"(Marks: {marks:.2f}) | Status: {status} | Sentiment: {sentiment}"
                    )
                else:
                    st.warning(
                        f"{employee} | Task: **{task}** → {completion}% "
                        f"(Marks: {marks:.2f}) | Status: Pending Manager Review"
                    )
        else:
            st.warning("No tasks found for this company.")
    elif not company:
        st.error("Enter company name")

# ------------------------------
# Manager Section
# ------------------------------
elif role == "Manager":
    st.header("Manager Review Section")
    all_res = index.query(vector=random_vector(), top_k=100, include_metadata=True)
    companies = list(set([m.metadata.get("company","?") for m in all_res.matches])) if all_res.matches else []

    if companies:
        company = st.selectbox("Select Company", companies)
    else:
        st.warning("No companies found.")
        company = None

    if company:
        res = index.query(
            vector=random_vector(),
            top_k=100,
            include_metadata=True,
            include_values=True,
            filter={"company": {"$eq": company}, "reviewed": {"$eq": False}}
        )

        pending_tasks = res.matches or []
        if pending_tasks:
            st.subheader(f"Pending Tasks for {company}")

            # Auto-assign & clustering
            employees = list(set([m.metadata.get("employee","?") for m in pending_tasks]))
            if st.button("Auto-Assign Tasks"):
                assigned_tasks = assign_task_auto([m.metadata for m in pending_tasks], employees)
                for i, match in enumerate(pending_tasks):
                    match.metadata['assigned_to'] = assigned_tasks[i]['assigned_to']
                st.success("Tasks auto-assigned based on workload")

            tasks_metadata = [m.metadata for m in pending_tasks]
            clustered_tasks = cluster_tasks(tasks_metadata)
            st.subheader("Task Clusters")
            for t in clustered_tasks:
                st.write(f"{t.get('task','?')} → Cluster {t.get('cluster','?')} | Completion: {t.get('completion',0)}%")

            st.subheader("Employee Performance")
            perf = classify_performance([m.metadata for m in pending_tasks])
            for emp, cat in perf.items():
                st.write(f"{emp} → {cat} Performer")

            with st.form(key="manager_review_form"):
                for match in pending_tasks:
                    md = match.metadata or {}
                    emp = md.get("employee", "?")
                    task = md.get("task", "?")
                    emp_completion = float(md.get("completion", 0))
                    st.write(f"{emp} | Task: **{task}**")
                    st.slider(
                        f"Adjust Completion ({emp} - {task})",
                        0, 100, int(emp_completion),
                        key=f"adj_{match.id}"
                    )
                    st.text_area(
                        f"Manager Comments ({emp} - {task})",
                        key=f"c_{match.id}"
                    )

                submit = st.form_submit_button("Save All Reviews")
                if submit:
                    for match in pending_tasks:
                        md = match.metadata or {}
                        manager_completion = st.session_state[f"adj_{match.id}"]
                        comments = st.session_state[f"c_{match.id}"]
                        predicted_marks = float(lin_reg.predict([[manager_completion]])[0])
                        status = log_reg.predict([[manager_completion]])[0]
                        status_text = "On Track" if status == 1 else "Delayed"

                        sentiment_text = "N/A"
                        if comments:
                            try:
                                X_new = vectorizer.transform([comments])
                                sentiment = svm_clf.predict(X_new)[0]
                                sentiment_text = "Positive" if sentiment == 1 else "Negative"
                            except Exception:
                                sentiment_text = "N/A"

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
                    st.success("All reviews saved successfully!")
        else:
            st.success(f"All tasks for {company} have already been reviewed!")

# ------------------------------
# Admin / Debug Section
# ------------------------------
elif role == "Admin / Debug":
    st.header("Pinecone Data Viewer / Debug Mode")

    if st.button("List All Indexes"):
        st.write(pc.list_indexes())

    st.subheader("Query Pinecone Records")
    company_filter = st.text_input("Filter by Company (optional)")
    reviewed_filter = st.selectbox("Filter by Reviewed Status", ["All", "True", "False"])

    filter_dict = {}
    if company_filter:
        filter_dict["company"] = {"$eq": company_filter}
    if reviewed_filter != "All":
        filter_dict["reviewed"] = {"$eq": reviewed_filter == "True"}

    if st.button("Query Pinecone"):
        query_vector = random_vector()
        res = index.query(
            vector=query_vector,
            top_k=50,
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )
        if res.matches:
            st.success(f"Found {len(res.matches)} records")
            for match in res.matches:
                st.write(f"**ID:** {match.id}")
                st.json(match.metadata)
        else:
            st.warning("No records found for given filters.")

    if st.button("Show Index Stats"):
        stats = index.describe_index_stats()
        st.json(stats)
