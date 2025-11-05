import streamlit as st
from pinecone import Pinecone
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

# -----------------------------
# CONFIG & INIT
# -----------------------------
st.set_page_config(page_title="AI-Powered Task Management System", layout="wide")
st.title("AI-Powered Task Management System")

# ✅ Secure API key loading
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "task"
DIMENSION = 1024

# ✅ Connect to existing index (don’t delete or recreate)
index = pc.Index(INDEX_NAME)

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def rand_vec():
    return np.random.rand(DIMENSION).tolist()

# -----------------------------
# SIMPLE AI MODELS
# -----------------------------
lin_reg = LinearRegression().fit([[0], [50], [100]], [0, 2.5, 5])
log_reg = LogisticRegression().fit([[0], [40], [80], [100]], [0, 0, 1, 1])

comments = ["excellent work", "needs improvement", "bad performance", "great job", "average"]
sentiments = [1, 0, 0, 1, 0]
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(comments)
svm_clf = SVC()
svm_clf.fit(X_train, sentiments)

rf = RandomForestClassifier()
rf.fit(np.array([[10, 2], [50, 1], [90, 0], [100, 0]]), [0, 1, 0, 0])

# -----------------------------
# HELPERS
# -----------------------------
def safe_meta(md):
    clean = {}
    for k, v in md.items():
        if v is None:
            v = ""
        elif isinstance(v, (datetime, date)):
            v = v.isoformat()
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

# -----------------------------
# ROLE SELECTION
# -----------------------------
role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client", "Admin"])
current_month = datetime.now().strftime("%B %Y")

# -----------------------------
# MANAGER: ASSIGN + REVIEW
# -----------------------------
if role == "Manager":
    st.header("Manager Dashboard")
    tab1, tab2 = st.tabs(["Assign Task", "Review Tasks"])

    # --- Assign Task ---
    with tab1:
        with st.form("assign"):
            company = st.text_input("Company Name")
            employee = st.text_input("Employee Name")
            task = st.text_input("Task Title")
            desc = st.text_area("Description")
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
                    "deadline": deadline.isoformat(),
                    "month": month,
                    "completion": 0,
                    "marks": 0,
                    "status": "Assigned",
                    "manager_reviewed": False,
                    "client_reviewed": False,
                    "assigned_on": now()
                })
                index.upsert([{"id": tid, "values": rand_vec(), "metadata": md}])
                st.success(f"Task '{task}' assigned to {employee}")

    # --- Review Tasks ---
    with tab2:
        df = fetch_all()
        if df.empty:
            st.info("No tasks found.")
        else:
            df["manager_reviewed"] = df.get("manager_reviewed", False)
            pending = df[df["manager_reviewed"].astype(str).str.lower() != "true"]

            if pending.empty:
                st.info("No tasks pending manager review.")
            else:
                for _, r in pending.iterrows():
                    st.subheader(r["task"])
                    st.write(f"Employee: {r.get('employee')}")
                    st.write(f"Completion: {r.get('completion')}%")
                    st.write(f"Deadline: {r.get('deadline')}")
                    marks = float(lin_reg.predict([[float(r.get('completion', 0))]])[0])
                    comments = st.text_area(f"Manager Comments for {r['task']}", key=f"mgr_comments_{r['_id']}")
                    approve = st.radio(f"Approve {r['task']}?", ["Yes", "No"], key=f"mgr_approve_{r['_id']}")

                    if st.button(f"Finalize Review: {r['task']}", key=f"mgr_finalize_{r['_id']}"):
                        sent_val = int(svm_clf.predict(vectorizer.transform([comments]))[0])
                        sent = "Positive" if sent_val == 1 else "Negative"
                        md = safe_meta({
                            **r,
                            "marks": marks,
                            "manager_comments": comments,
                            "manager_reviewed": True,
                            "approved": approve,
                            "sentiment": sent,
                            "reviewed_on": now()
                        })
                        index.upsert([{"id": r["_id"], "values": rand_vec(), "metadata": md}])
                        st.success(f"Review finalized for {r['task']} ({sent})")

# -----------------------------
# TEAM MEMBER DASHBOARD
# -----------------------------
elif role == "Team Member":
    st.header("Team Member Dashboard")
    company = st.text_input("Company")
    employee = st.text_input("Your Name")

    if st.button("Load Tasks"):
        res = index.query(vector=rand_vec(), top_k=500, include_metadata=True)
        all_tasks = [(m.id, m.metadata) for m in res.matches or []]
        tasks = [(tid, md) for tid, md in all_tasks if md.get("company") == company and md.get("employee") == employee]
        st.session_state["tasks"] = tasks
        st.success(f"Loaded {len(tasks)} tasks.")

    tasks = st.session_state.get("tasks", [])
    if tasks:
        for tid, md in tasks:
            st.subheader(md.get("task"))
            st.write(md.get("description"))
            curr = float(md.get("completion", 0))
            new = st.slider(f"Completion for {md.get('task')}", 0, 100, int(curr), key=f"slider_{tid}")
            if st.button(f"Submit {md.get('task')}", key=f"submit_{tid}"):
                marks = float(lin_reg.predict([[new]])[0])
                track = "On Track" if log_reg.predict([[new]])[0] == 1 else "Delayed"
                risk = "High" if rf.predict([[new, 0]])[0] else "Low"
                md2 = safe_meta({
                    **md,
                    "completion": new,
                    "marks": marks,
                    "status": track,
                    "deadline_risk": risk,
                    "submitted_on": now()
                })
                index.upsert([{"id": tid, "values": rand_vec(), "metadata": md2}])
                st.success(f"Updated {md.get('task')} ({track}, Risk={risk})")

# -----------------------------
# CLIENT DASHBOARD
# -----------------------------
elif role == "Client":
    st.header("Client Dashboard")
    company = st.text_input("Company Name")

    if st.button("Load Manager-Reviewed Tasks"):
        df = fetch_all()
        if not df.empty:
            df["manager_reviewed"] = df.get("manager_reviewed", False)
            df = df[(df["company"] == company) & (df["manager_reviewed"].astype(str).str.lower() == "true")]
            st.session_state["client_tasks"] = df
            st.success(f"Loaded {len(df)} tasks.")

    if "client_tasks" in st.session_state:
        df = st.session_state["client_tasks"]
        if not df.empty:
            for _, r in df.iterrows():
                st.subheader(r["task"])
                st.write(f"Employee: {r['employee']}")
                st.write(f"Completion: {r['completion']}%")
                st.write(f"Marks: {r['marks']}")
                comment = st.text_area(f"Client Comments for {r['task']}", key=f"client_comment_{r['_id']}")
                approve = st.radio(f"Approve {r['task']}?", ["Yes", "No"], key=f"client_approve_{r['_id']}")
                if st.button(f"Submit Review for {r['task']}", key=f"client_submit_{r['_id']}"):
                    md2 = safe_meta({
                        **r,
                        "client_reviewed": True,
                        "client_approved": approve,
                        "client_comments": comment,
                        "client_reviewed_on": now()
                    })
                    index.upsert([{"id": r["_id"], "values": rand_vec(), "metadata": md2}])
                    st.success(f"Client review submitted for {r['task']}")

# -----------------------------
# ADMIN DASHBOARD
# -----------------------------
elif role == "Admin":
    st.header("Admin Dashboard")
    df = fetch_all()
    if df.empty:
        st.warning("No tasks found.")
    else:
        df["marks"] = pd.to_numeric(df["marks"], errors="coerce")
        df["completion"] = pd.to_numeric(df["completion"], errors="coerce")

        st.subheader("Top Employees by Average Marks")
        top = df.groupby("employee")["marks"].mean().reset_index().sort_values("marks", ascending=False).head(5)
        st.dataframe(top)

        st.subheader("K-Means Clustering (Performance)")
        valid_df = df[["completion", "marks"]].fillna(0)
        n_clusters = min(3, len(valid_df))
        if n_clusters > 1:
            km = KMeans(n_clusters=n_clusters, n_init="auto").fit(valid_df)
            df["cluster"] = km.labels_
            st.plotly_chart(px.scatter(df, x="completion", y="marks", color=df["cluster"].astype(str),
                                       hover_data=["employee", "task"], title="Employee Task Clusters"))

        st.subheader("Summary")
        avg_m = df["marks"].mean()
        avg_c = df["completion"].mean()
        st.info(f"Average Marks: {avg_m:.2f}, Average Completion: {avg_c:.1f}%")

        st.download_button("Download All Tasks (CSV)", df.to_csv(index=False), "tasks.csv", "text/csv")
