import streamlit as st
import pandas as pd
import numpy as np
import uuid
from datetime import datetime, date
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import plotly.express as px
from pinecone import Pinecone

# -------------------------------------
# SETUP
# -------------------------------------
st.set_page_config(page_title="AI Task Management System", layout="wide")
st.title("AI-Powered Task Management System")

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "task"
DIMENSION = 1024

# Recreate the Pinecone index (Option 2)
if INDEX_NAME in [i["name"] for i in pc.list_indexes()]:
    pc.delete_index(INDEX_NAME)

pc.create_index(name=INDEX_NAME, dimension=DIMENSION, metric="cosine")
index = pc.Index(INDEX_NAME)

# Utility functions
def now(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def rand_vec(): return np.random.rand(DIMENSION).tolist()

def safe_meta(md):
    clean = {}
    for k, v in md.items():
        if isinstance(v, (datetime, date)):
            v = v.isoformat()
        if v is None:
            v = ""
        clean[k] = v
    return clean

def fetch_all():
    try:
        res = index.query(vector=rand_vec(), top_k=1000, include_metadata=True)
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

# -------------------------------------
# SIMPLE AI MODELS
# -------------------------------------
lin_reg = LinearRegression()
lin_reg.fit([[0], [50], [100]], [0, 2.5, 5])

log_reg = LogisticRegression()
log_reg.fit([[0], [40], [80], [100]], [0, 0, 1, 1])

vectorizer = CountVectorizer()
comments = ["excellent work", "needs improvement", "bad performance", "great job", "average"]
sentiments = [1, 0, 0, 1, 0]
X_train = vectorizer.fit_transform(comments)
svm_clf = SVC()
svm_clf.fit(X_train, sentiments)

rf = RandomForestClassifier()
X_rf = np.array([[10, 2], [50, 1], [90, 0], [100, 0]])
y_rf = [1, 1, 0, 0]
rf.fit(X_rf, y_rf)

# -------------------------------------
# ROLE SELECTION
# -------------------------------------
role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client", "Admin"])
current_month = datetime.now().strftime("%B %Y")

# -------------------------------------
# MANAGER PANEL
# -------------------------------------
if role == "Manager":
    st.header("Manager Dashboard")
    tab1, tab2 = st.tabs(["Assign Task", "Review Tasks"])

    # Assign Task
    with tab1:
        with st.form("assign_task"):
            company = st.text_input("Company Name")
            employee = st.text_input("Employee Name")
            task = st.text_input("Task Title")
            desc = st.text_area("Task Description")
            deadline = st.date_input("Deadline", value=date.today())
            month = st.text_input("Month", value=current_month)
            submit = st.form_submit_button("Assign Task")

            if submit and company and employee and task:
                tid = str(uuid.uuid4())
                md = safe_meta({
                    "company": company,
                    "employee": employee,
                    "task": task,
                    "description": desc,
                    "deadline": deadline,
                    "month": month,
                    "completion": 0,
                    "marks": 0,
                    "status": "Assigned",
                    "manager_reviewed": False,
                    "client_reviewed": False,
                    "created_on": now()
                })
                index.upsert([{"id": tid, "values": rand_vec(), "metadata": md}])
                st.success(f"Task '{task}' assigned to {employee}")

    # Review Tasks
    with tab2:
        df = fetch_all()
        if not df.empty and "completion" in df.columns:
            pending = df[df["manager_reviewed"].astype(str).str.lower() != "true"]
        else:
            pending = pd.DataFrame()

        if pending.empty:
            st.info("No pending tasks for review.")
        else:
            for _, r in pending.iterrows():
                st.subheader(r["task"])
                st.write(f"Employee: {r['employee']}")
                st.write(f"Reported Completion: {r['completion']}%")
                st.write(f"Marks: {r['marks']}")
                st.write(f"Deadline: {r['deadline']}")

                adj_completion = st.slider(f"Adjust Completion for {r['task']}", 0, 100, int(r["completion"]))
                boss_comments = st.text_area(f"Manager Comments for {r['task']}")
                approve = st.radio(f"Approve {r['task']}?", ["Yes", "No"], key=f"approve_{r['_id']}")

                if st.button(f"Finalize Review {r['task']}", key=r["_id"]):
                    marks = float(lin_reg.predict([[adj_completion]])[0])
                    sentiment = "Positive" if svm_clf.predict(vectorizer.transform([boss_comments]))[0] == 1 else "Negative"
                    md2 = safe_meta({
                        **r,
                        "completion": adj_completion,
                        "marks": marks,
                        "manager_comments": boss_comments,
                        "manager_reviewed": True,
                        "approved": approve == "Yes",
                        "sentiment": sentiment,
                        "reviewed_on": now()
                    })
                    index.upsert([{"id": r["_id"], "values": rand_vec(), "metadata": md2}])
                    st.success(f"Review completed for {r['task']}")

# -------------------------------------
# TEAM MEMBER PANEL
# -------------------------------------
elif role == "Team Member":
    st.header("Team Member Dashboard")
    company = st.text_input("Company")
    employee = st.text_input("Your Name")

    if st.button("Load Tasks"):
        df = fetch_all()
        df = df[(df["company"] == company) & (df["employee"] == employee)]
        st.session_state["tasks"] = df.to_dict("records") if not df.empty else []
        st.success(f"Loaded {len(st.session_state['tasks'])} tasks.")

    for task in st.session_state.get("tasks", []):
        st.subheader(task["task"])
        st.write(task["description"])
        new_comp = st.slider(f"Update Completion for {task['task']}", 0, 100, int(task["completion"]))
        if st.button(f"Submit Update {task['task']}", key=task["_id"]):
            marks = float(lin_reg.predict([[new_comp]])[0])
            track = "On Track" if log_reg.predict([[new_comp]])[0] == 1 else "Delayed"
            risk = "High" if rf.predict([[new_comp, 0]])[0] == 1 else "Low"
            md2 = safe_meta({
                **task,
                "completion": new_comp,
                "marks": marks,
                "status": track,
                "risk": risk,
                "updated_on": now()
            })
            index.upsert([{"id": task["_id"], "values": rand_vec(), "metadata": md2}])
            st.success(f"Updated {task['task']}")

# -------------------------------------
# CLIENT PANEL
# -------------------------------------
elif role == "Client":
    st.header("Client Dashboard")
    company = st.text_input("Company Name")

    if st.button("Load Approved"):
        df = fetch_all()
        df = df[(df["company"] == company) &
                (df["manager_reviewed"].astype(str).str.lower() == "true")]
        st.session_state["client_tasks"] = df.to_dict("records") if not df.empty else []
        st.success(f"Loaded {len(st.session_state['client_tasks'])} tasks.")

    for task in st.session_state.get("client_tasks", []):
        st.subheader(task["task"])
        st.write(f"Employee: {task['employee']}")
        st.write(f"Completion: {task['completion']}%")
        comment = st.text_area(f"Feedback for {task['task']}", key=f"client_{task['_id']}")
        if st.button(f"Approve {task['task']}", key=f"approve_{task['_id']}"):
            md2 = safe_meta({
                **task,
                "client_reviewed": True,
                "client_comment": comment,
                "client_reviewed_on": now()
            })
            index.upsert([{"id": task["_id"], "values": rand_vec(), "metadata": md2}])
            st.success(f"Approved {task['task']}")

# -------------------------------------
# ADMIN PANEL
# -------------------------------------
elif role == "Admin":
    st.header("Admin Dashboard")
    df = fetch_all()
    if df.empty:
        st.info("No data found.")
    else:
        df["marks"] = pd.to_numeric(df["marks"], errors="coerce")
        df["completion"] = pd.to_numeric(df["completion"], errors="coerce")

        st.subheader("Top Performers")
        top = df.groupby("employee")["marks"].mean().reset_index().sort_values(by="marks", ascending=False).head(5)
        st.dataframe(top)

        st.subheader("K-Means Clustering (Performance)")
        if len(df) >= 3:
            km = KMeans(n_clusters=3, n_init="auto").fit(df[["completion", "marks"]].fillna(0))
            df["cluster"] = km.labels_
            st.plotly_chart(px.scatter(df, x="completion", y="marks", color=df["cluster"].astype(str),
                                       hover_data=["employee", "task"], title="Employee Task Clusters"))
        else:
            st.warning("Not enough data for K-Means (need at least 3 tasks).")

        st.download_button("Download All Tasks (CSV)", df.to_csv(index=False), "tasks.csv", "text/csv")
