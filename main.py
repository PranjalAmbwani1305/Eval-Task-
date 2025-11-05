import streamlit as st
from pinecone import Pinecone, ServerlessSpec
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

# -----------------------------
# STREAMLIT CONFIG
# -----------------------------
st.set_page_config(page_title="AI-Powered Task Management System", layout="wide")
st.title("AI-Powered Task Management System")

# -----------------------------
# PINECONE INIT (SAFE)
# -----------------------------
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "task"
DIMENSION = 1024

existing_indexes = [i["name"] for i in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    st.info(f"Creating index '{INDEX_NAME}' ...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def rand_vec():
    return np.random.rand(DIMENSION).tolist()

# -----------------------------
# MODELS
# -----------------------------
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
y_rf = [0, 1, 0, 0]
rf.fit(X_rf, y_rf)

# -----------------------------
# HELPERS
# -----------------------------
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
        data = []
        for m in res.matches:
            md = m.metadata or {}
            md["_id"] = m.id
            data.append(md)
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error fetching from Pinecone: {e}")
        return pd.DataFrame()

# -----------------------------
# DASHBOARD SELECTION
# -----------------------------
role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client", "Admin"])
current_month = datetime.now().strftime("%B %Y")

# -----------------------------
# MANAGER
# -----------------------------
if role == "Manager":
    st.header("Manager Dashboard")
    tab1, tab2 = st.tabs(["Assign Task", "Review Tasks"])

    with tab1:
        with st.form("assign_task"):
            company = st.text_input("Company Name")
            employee = st.text_input("Employee Name")
            task = st.text_input("Task Title")
            desc = st.text_area("Task Description")
            deadline = st.date_input("Deadline", date.today())
            month = st.text_input("Month", value=current_month)
            submitted = st.form_submit_button("Assign Task")

            if submitted and company and employee and task:
                tid = str(uuid.uuid4())
                meta = safe_meta({
                    "company": company,
                    "employee": employee,
                    "task": task,
                    "description": desc,
                    "deadline": deadline,
                    "month": month,
                    "completion": 0,
                    "marks": 0,
                    "manager_reviewed": False,
                    "client_reviewed": False,
                    "assigned_on": now(),
                    "status": "Assigned"
                })
                index.upsert([{"id": tid, "values": rand_vec(), "metadata": meta}])
                st.success(f"Task '{task}' assigned to {employee}")

    with tab2:
        df = fetch_all()
        if df.empty:
            st.info("No tasks found.")
        else:
            pending = df[df.get("manager_reviewed", "false").astype(str).str.lower() != "true"]
            if pending.empty:
                st.info("No pending reviews.")
            else:
                for _, row in pending.iterrows():
                    st.subheader(row["task"])
                    st.write(f"Employee: {row['employee']}")
                    st.write(f"Reported Completion: {row.get('completion', 0)}%")
                    st.write(f"Current Marks: {row.get('marks', 0)}")
                    st.write(f"Deadline Risk: {row.get('deadline_risk', 'Unknown')}")

                    adj = st.slider(f"Adjust Completion % for {row['task']}", 0, 100, int(row.get('completion', 0)))
                    marks = float(lin_reg.predict([[adj]])[0])
                    comments = st.text_area(f"Boss Comments for {row['task']}", "")
                    approve = st.radio(f"Approve Task {row['task']}?", ["Yes", "No"], horizontal=True)

                    if st.button(f"Finalize Review {row['_id']}", key=row["_id"]):
                        sent_val = int(svm_clf.predict(vectorizer.transform([comments]))[0])
                        sentiment = "Positive" if sent_val == 1 else "Negative"
                        meta = safe_meta({
                            **row,
                            "completion": adj,
                            "marks": marks,
                            "manager_comments": comments,
                            "manager_reviewed": True,
                            "manager_approval": approve,
                            "sentiment": sentiment,
                            "reviewed_on": now()
                        })
                        index.upsert([{"id": row["_id"], "values": rand_vec(), "metadata": meta}])
                        st.success(f"Task {row['task']} reviewed and saved ({sentiment})")

# -----------------------------
# TEAM MEMBER
# -----------------------------
elif role == "Team Member":
    st.header("Team Member Dashboard")
    company = st.text_input("Company")
    employee = st.text_input("Your Name")

    if st.button("Load Tasks"):
        df = fetch_all()
        if not df.empty:
            tasks = df[
                (df["company"].astype(str).str.lower() == company.lower()) &
                (df["employee"].astype(str).str.lower() == employee.lower())
            ]
            st.session_state["tasks"] = tasks
            st.success(f"Loaded {len(tasks)} tasks.")
        else:
            st.warning("No tasks found.")

    if "tasks" in st.session_state and not st.session_state["tasks"].empty:
        for _, t in st.session_state["tasks"].iterrows():
            st.subheader(t["task"])
            st.write(t["description"])
            new = st.slider(f"Completion for {t['task']}", 0, 100, int(t["completion"]))
            if st.button(f"Submit Progress {t['_id']}", key=t["_id"]):
                marks = float(lin_reg.predict([[new]])[0])
                track = "On Track" if log_reg.predict([[new]])[0] == 1 else "Delayed"
                risk = "High" if rf.predict([[new, 0]])[0] else "Low"
                meta = safe_meta({
                    **t,
                    "completion": new,
                    "marks": marks,
                    "status": track,
                    "deadline_risk": risk,
                    "updated_on": now()
                })
                index.upsert([{"id": t["_id"], "values": rand_vec(), "metadata": meta}])
                st.success(f"{t['task']} updated ({track}, Risk: {risk})")

# -----------------------------
# CLIENT
# -----------------------------
elif role == "Client":
    st.header("Client Dashboard")
    company = st.text_input("Company Name")

    if st.button("Load Reviewed Tasks"):
        df = fetch_all()
        if not df.empty:
            approved = df[
                (df["company"].astype(str).str.lower() == company.lower()) &
                (df.get("manager_reviewed", "false").astype(str).str.lower() == "true") &
                (df.get("client_reviewed", "false").astype(str).str.lower() != "true")
            ]
            st.session_state["client_tasks"] = approved
            st.success(f"Loaded {len(approved)} tasks.")
        else:
            st.warning("No tasks found.")

    if "client_tasks" in st.session_state and not st.session_state["client_tasks"].empty:
        for _, t in st.session_state["client_tasks"].iterrows():
            st.subheader(t["task"])
            st.write(f"Employee: {t['employee']}")
            st.write(f"Completion: {t['completion']}%")
            comment = st.text_area(f"Client Feedback for {t['task']}", key=f"comm_{t['_id']}")
            if st.button(f"Approve {t['_id']}", key=f"approve_{t['_id']}"):
                meta = safe_meta({
                    **t,
                    "client_comments": comment,
                    "client_reviewed": True,
                    "client_approved_on": now()
                })
                index.upsert([{"id": t["_id"], "values": rand_vec(), "metadata": meta}])
                st.success(f"{t['task']} approved successfully")

# -----------------------------
# ADMIN
# -----------------------------
elif role == "Admin":
    st.header("Admin Dashboard")
    df = fetch_all()

    if df.empty:
        st.warning("No data found in Pinecone.")
    else:
        df["marks"] = pd.to_numeric(df["marks"], errors="coerce").fillna(0)
        df["completion"] = pd.to_numeric(df["completion"], errors="coerce").fillna(0)

        st.subheader("K-Means Clustering (Performance)")
        n_clusters = min(3, len(df)) if len(df) >= 1 else 1
        km = KMeans(n_clusters=n_clusters, n_init="auto").fit(df[["completion", "marks"]])
        df["cluster"] = km.labels_
        st.plotly_chart(px.scatter(df, x="completion", y="marks", color=df["cluster"].astype(str),
                                   hover_data=["employee", "task"],
                                   title="Employee Performance Clusters"))

        st.subheader("Top Employees by Marks")
        top = df.groupby("employee")["marks"].mean().reset_index().sort_values("marks", ascending=False).head(5)
        st.dataframe(top)

        st.subheader("Summary")
        avg_c = df["completion"].mean()
        avg_m = df["marks"].mean()
        st.info(f"Avg Completion: {avg_c:.2f}%, Avg Marks: {avg_m:.2f}")
        st.download_button("Download All Tasks", df.to_csv(index=False), "tasks.csv", "text/csv")
