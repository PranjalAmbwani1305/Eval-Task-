import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import uuid
from datetime import datetime, date
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.cluster import KMeans

# ------------------------------
# Configuration / Init
# ------------------------------
st.set_page_config(page_title="Task Completion & Review", layout="wide")

PC_API_KEY = st.secrets["PINECONE_API_KEY"]
pc = Pinecone(api_key=PC_API_KEY)
INDEX_NAME = "task"
DIMENSION = 1024

if INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

# ------------------------------
# Simple ML Models (demo)
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
# Helpers
# ------------------------------
def random_vector(dim=DIMENSION):
    return np.random.rand(dim).tolist()

def safe_metadata(md: dict):
    clean = {}
    for k, v in md.items():
        # convert numpy types
        if isinstance(v, (np.generic,)):
            v = v.item()
        clean[k] = v
    return clean

def cluster_tasks(tasks, n_clusters=3):
    if not tasks:
        return []
    X = np.array([[t.get("completion", 0), t.get("marks", 0)] for t in tasks])
    n_clusters = min(n_clusters, len(tasks))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    for t, c in zip(tasks, clusters):
        t["cluster"] = int(c)
    return tasks

def now_ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ------------------------------
# UI
# ------------------------------
st.title("AI-Powered Task Completion & Review System")

role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client"])

# ------------------------------
# MANAGER: Assign & Final Review (only after client approval)
# ------------------------------
if role == "Manager":
    st.header("Manager: Assign Tasks / Final Review")
    company = st.text_input("Company Name (for assignment & review)")

    st.subheader("Assign New Task")
    with st.form("assign_form"):
        assign_month = st.selectbox(
            "Month",
            ["January", "February", "March", "April", "May", "June",
             "July", "August", "September", "October", "November", "December"]
        )
        task_title = st.text_input("Task Title")
        description = st.text_area("Task Description")
        deadline = st.date_input("Deadline", value=date.today())
        assignee = st.text_input("Assign to Employee (exact name)")
        submit_assign = st.form_submit_button("Assign Task")

        if submit_assign:
            if not all([company, task_title, assignee]):
                st.error("Please provide Company Name, Task Title and Employee name.")
            else:
                tid = str(uuid.uuid4())
                metadata = safe_metadata({
                    "company": company,
                    "month": assign_month,
                    "task": task_title,
                    "description": description,
                    "deadline": str(deadline),
                    "employee": assignee,
                    "completion": 0.0,
                    "marks": 0.0,
                    "status": "Assigned",
                    "reviewed": False,
                    "client_approved": False,
                    "assigned_on": now_ts()
                })
                index.upsert(vectors=[{"id": tid, "values": random_vector(), "metadata": metadata}])
                st.success(f"Assigned '{task_title}' to {assignee} for {assign_month}.")

    st.markdown("---")
    st.subheader("Final Review (only tasks approved by client)")

    if st.button("Load Client-Approved Tasks for Review"):
        if not company:
            st.error("Provide Company Name to load tasks.")
        else:
            res = index.query(
                vector=random_vector(),
                top_k=500,
                include_metadata=True,
                include_values=True,
                filter={"company": {"$eq": company}, "client_approved": {"$eq": True}, "reviewed": {"$eq": False}}
            )
            st.session_state["manager_review"] = res.matches if res.matches else []
            st.write(f"Loaded {len(st.session_state.get('manager_review', []))} tasks awaiting final review.")

    if st.session_state.get("manager_review"):
        tasks_meta = [m.metadata for m in st.session_state["manager_review"]]
        clustered = cluster_tasks(tasks_meta)
        st.write("Clusters (based on completion & marks):")
        for t in clustered:
            st.write(f"{t.get('task')} | Employee: {t.get('employee')} | Completion: {t.get('completion')}% | Cluster: {t.get('cluster')}")

        for match in st.session_state["manager_review"]:
            md = match.metadata or {}
            st.markdown("### " + md.get("task", "?"))
            st.write(f"Employee: {md.get('employee','?')}")
            st.write(f"Completion: {md.get('completion',0)}% | Current status: {md.get('status','?')}")
            st.write(f"Submitted on: {md.get('submitted_on','N/A')}")
            final_marks = st.number_input(f"Final Marks for {md.get('task','?')} (0-5)", min_value=0.0, max_value=5.0, step=0.1, key=f"fm_{match.id}")
            final_comments = st.text_area(f"Final Comments for {md.get('task','?')}", key=f"fc_{match.id}")
            approve = st.button(f"Complete Review & Approve ({md.get('task','?')})", key=f"approve_{match.id}")
            if approve:
                # compute final status from completion if needed
                final_status = md.get("status", "Unknown")
                updated_md = safe_metadata({
                    **md,
                    "marks": float(final_marks),
                    "status": final_status,
                    "reviewed": True,
                    "reviewed_on": now_ts(),
                    "manager_comments": final_comments
                })
                values_to_use = match.values if hasattr(match, "values") and match.values else random_vector()
                index.upsert(vectors=[{"id": match.id, "values": values_to_use, "metadata": updated_md}])
                st.success(f"Task '{md.get('task')}' reviewed and marked complete.")
                # remove from session_state list
                st.session_state["manager_review"] = [m for m in st.session_state["manager_review"] if m.id != match.id]

# ------------------------------
# TEAM MEMBER: Load assigned tasks & submit progress
# ------------------------------
elif role == "Team Member":
    st.header("Team Member: View & Update Assigned Tasks")
    company = st.text_input("Company Name")
    employee = st.text_input("Your Name (exact)")

    if st.button("Load My Tasks"):
        if not company or not employee:
            st.error("Enter both Company and Your Name.")
        else:
            res = index.query(
                vector=random_vector(),
                top_k=500,
                include_metadata=True,
                include_values=True,
                filter={"company": {"$eq": company}, "employee": {"$eq": employee}}
            )
            st.session_state["tasks_loaded"] = [(m.id, m.metadata, m.values) for m in res.matches] if res.matches else []
            st.write(f"Loaded {len(st.session_state.get('tasks_loaded', []))} tasks.")

    if st.session_state.get("tasks_loaded"):
        tasks = st.session_state["tasks_loaded"]
        # group by month
        tasks_by_month = {}
        for tid, md, vals in tasks:
            month = md.get("month", "Unknown")
            tasks_by_month.setdefault(month, []).append((tid, md, vals))

        months = sorted(tasks_by_month.keys())
        month_selected = st.selectbox("Select Month", months)

        if month_selected:
            entries = tasks_by_month.get(month_selected, [])
            if not entries:
                st.info("No tasks for this month.")
            for tid, md, vals in entries:
                st.markdown("### " + md.get("task", "?"))
                st.write(f"Description: {md.get('description','No description')}")
                st.write(f"Deadline: {md.get('deadline','N/A')}")
                completion = float(md.get("completion", 0))
                # progress bar with color indicator (textual color since st.progress doesn't support color)
                if completion < 50:
                    color_label = "Low"
                elif completion < 80:
                    color_label = "Moderate"
                else:
                    color_label = "Good"
                st.progress(min(max(completion/100.0, 0.0), 1.0))
                st.write(f"Completion: {completion}% ({color_label})")
                new_completion = st.slider(f"Update completion for '{md.get('task')}'", 0, 100, int(completion), key=f"slider_{tid}")

                submit_key = f"submit_{tid}"
                if st.button(f"Submit Progress ({md.get('task')})", key=submit_key):
                    marks = float(lin_reg.predict([[new_completion]])[0])
                    status_pred = log_reg.predict([[new_completion]])[0]
                    status_text = "On Track" if status_pred == 1 else "Delayed"
                    updated_md = safe_metadata({
                        **md,
                        "completion": float(new_completion),
                        "marks": marks,
                        "status": status_text,
                        "reviewed": False,
                        "submitted_on": now_ts(),
                        # client approval reset (client must approve after team update)
                        "client_approved": False,
                        "client_approved_on": None
                    })
                    values_to_use = vals if vals else random_vector()
                    index.upsert(vectors=[{"id": tid, "values": values_to_use, "metadata": updated_md}])
                    # update session state entry
                    for i, (xid, xmd, xvals) in enumerate(st.session_state["tasks_loaded"]):
                        if xid == tid:
                            st.session_state["tasks_loaded"][i] = (xid, updated_md, xvals)
                            break
                    st.success(f"Progress submitted at {updated_md['submitted_on']}")

# ------------------------------
# CLIENT: View tasks & approve after verifying
# ------------------------------
elif role == "Client":
    st.header("Client: View & Approve Tasks")
    company = st.text_input("Company Name")

    if st.button("Load Company Tasks"):
        if not company:
            st.error("Enter company name.")
        else:
            res = index.query(
                vector=random_vector(),
                top_k=800,
                include_metadata=True,
                include_values=True,
                filter={"company": {"$eq": company}}
            )
            st.session_state["client_tasks"] = [(m.id, m.metadata, m.values) for m in res.matches] if res.matches else []
            st.write(f"Loaded {len(st.session_state.get('client_tasks', []))} tasks for {company}.")

    if st.session_state.get("client_tasks"):
        tasks = st.session_state["client_tasks"]
        # group by month
        tasks_by_month = {}
        for tid, md, vals in tasks:
            month = md.get("month", "Unknown")
            tasks_by_month.setdefault(month, []).append((tid, md, vals))

        months = sorted(tasks_by_month.keys())
        month_selected = st.selectbox("Select Month", months, key="client_month")

        if month_selected:
            entries = tasks_by_month.get(month_selected, [])
            if not entries:
                st.info("No tasks for this month.")
            for tid, md, vals in entries:
                st.markdown("### " + md.get("task", "?"))
                st.write(f"Employee: {md.get('employee','?')}")
                st.write(f"Completion: {md.get('completion',0)}%")
                st.write(f"Submitted on: {md.get('submitted_on','N/A')}")
                st.write(f"Client Approved: {md.get('client_approved', False)}")
                if not md.get("client_approved", False):
                    if st.button(f"Approve Task ({md.get('task')})", key=f"client_approve_{tid}"):
                        updated_md = safe_metadata({
                            **md,
                            "client_approved": True,
                            "client_approved_on": now_ts()
                        })
                        values_to_use = vals if vals else random_vector()
                        index.upsert(vectors=[{"id": tid, "values": values_to_use, "metadata": updated_md}])
                        # update session_state
                        for i, (xid, xmd, xvals) in enumerate(st.session_state["client_tasks"]):
                            if xid == tid:
                                st.session_state["client_tasks"][i] = (xid, updated_md, xvals)
                                break
                        st.success(f"Task '{md.get('task')}' approved by client at {updated_md['client_approved_on']}.")
                else:
                    st.write(f"Approved on: {md.get('client_approved_on', 'N/A')}")

# ------------------------------
# End
# ------------------------------
