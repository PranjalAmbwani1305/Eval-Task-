import streamlit as st
from pinecone import Pinecone, ServerlessSpec
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
st.set_page_config(page_title="AI Task System", layout="wide")
st.title("AI-Powered Task Management System")

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "task"
DIMENSION = 1024

if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(INDEX_NAME)

def now(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def rand_vec(): return np.random.rand(DIMENSION).tolist()

# -----------------------------
# SIMPLE AI MODELS
# -----------------------------
lin_reg = LinearRegression()
lin_reg.fit([[0], [50], [100]], [0, 2.5, 5])

log_reg = LogisticRegression()
log_reg.fit([[0], [40], [80], [100]], [0, 0, 1, 1])

comments = ["excellent work", "needs improvement", "bad performance", "great job", "average"]
sentiments = [1, 0, 0, 1, 0]
vectorizer = CountVectorizer()
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
        if v is None:
            v = ""
        elif isinstance(v, (datetime, date)):
            v = v.isoformat()
        clean[k] = v
    return clean

def fetch_all():
    try:
        res = index.query(vector=rand_vec(), top_k=1000, include_metadata=True)
        if not res.matches: return pd.DataFrame()
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
# MANAGER
# -----------------------------
if role == "Manager":
    st.header("Manager Dashboard")
    tab1, tab2 = st.tabs(["Assign Task", "Review Team Updates"])

    # Assign Task
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
                    "reviewed": False,
                    "client_reviewed": False,
                    "assigned_on": now()
                })
                index.upsert([{"id": tid, "values": rand_vec(), "metadata": md}])
                st.success(f"Task '{task}' assigned to {employee}")

    # Review Section
    with tab2:
        df = fetch_all()
        if df.empty:
            st.warning("No tasks found in Pinecone.")
        else:
            df.columns = [c.lower() for c in df.columns]
            df["completion"] = pd.to_numeric(df.get("completion", 0), errors="coerce").fillna(0)
            df["reviewed"] = df.get("reviewed", False).astype(str).str.lower()

            pending_reviews = df[
                (df["completion"] > 0) &
                (df["reviewed"] != "true")
            ]

            if pending_reviews.empty:
                st.info("No team-submitted tasks pending review.")
            else:
                st.subheader("Tasks Ready for Manager Review")
                for _, r in pending_reviews.iterrows():
                    st.markdown(f"#### {r['task']}")
                    st.write(f"**Employee:** {r.get('employee', 'Unknown')}")
                    st.write(f"**Completion:** {r.get('completion', 0)}%")
                    st.write(f"**Status:** {r.get('status', 'N/A')}")
                    st.write(f"**Deadline Risk:** {r.get('deadline_risk', 'N/A')}")
                    st.write(f"**Description:** {r.get('description', '')}")

                    marks = st.number_input(f"Marks (0–5) for {r['task']}", 0.0, 5.0, step=0.1)
                    comments = st.text_area(f"Manager Comments for {r['task']}", key=f"mgr_{r['_id']}")

                    if st.button(f"Finalize Review for {r['task']}", key=f"btn_{r['_id']}"):
                        sent_val = int(svm_clf.predict(vectorizer.transform([comments]))[0])
                        sentiment = "Positive" if sent_val == 1 else "Negative"
                        md = safe_meta({
                            **r,
                            "marks": marks,
                            "manager_comments": comments,
                            "reviewed": True,
                            "sentiment": sentiment,
                            "reviewed_on": now()
                        })
                        index.upsert([{"id": r["_id"], "values": rand_vec(), "metadata": md}])
                        st.success(f"Review completed for {r['task']} ({sentiment}).")

# -----------------------------
# TEAM MEMBER
# -----------------------------
elif role == "Team Member":
    st.header("Team Member Progress Update")
    company = st.text_input("Company")
    employee = st.text_input("Your Name")

    if st.button("Load Tasks"):
        res = index.query(vector=rand_vec(), top_k=500, include_metadata=True,
                          filter={"company": {"$eq": company}, "employee": {"$eq": employee}})
        st.session_state["tasks"] = [(m.id, m.metadata) for m in res.matches or []]
        st.success(f"Loaded {len(st.session_state['tasks'])} tasks.")

    for tid, md in st.session_state.get("tasks", []):
        st.subheader(md.get("task"))
        st.write(md.get("description"))
        curr = float(md.get("completion", 0))
        new = st.slider(f"Completion for {md.get('task')}", 0, 100, int(curr))
        if st.button(f"Submit {md.get('task')}", key=tid):
            marks = float(lin_reg.predict([[new]])[0])
            track = "On Track" if log_reg.predict([[new]])[0] == 1 else "Delayed"
            miss = rf.predict([[new, 0]])[0]
            md2 = safe_meta({
                **md,
                "completion": new,
                "marks": marks,
                "status": track,
                "deadline_risk": "High" if miss else "Low",
                "submitted_on": now(),
                "client_reviewed": False
            })
            index.upsert([{"id": tid, "values": rand_vec(), "metadata": md2}])
            st.success(f"Updated {md.get('task')} ({track}, Risk={md2['deadline_risk']})")

# -----------------------------
# CLIENT
# -----------------------------
elif role == "Client":
    st.header("Client Review")
    company = st.text_input("Company Name")

    if st.button("Load Completed"):
        res = index.query(vector=rand_vec(), top_k=500, include_metadata=True,
                          filter={"company": {"$eq": company}, "reviewed": {"$eq": True}})
        st.session_state["ctasks"] = [(m.id, m.metadata) for m in res.matches or []]
        st.success(f"Loaded {len(st.session_state['ctasks'])} tasks.")

    for tid, md in st.session_state.get("ctasks", []):
        st.subheader(md.get("task"))
        st.write(f"Employee: {md.get('employee')}")
        st.write(f"Completion: {md.get('completion')}%")
        comment = st.text_area(f"Feedback for {md.get('task')}", key=f"c_{tid}")
        if st.button(f"Approve {md.get('task')}", key=f"approve_{tid}"):
            md2 = safe_meta({
                **md,
                "client_reviewed": True,
                "client_comments": comment,
                "client_approved_on": now()
            })
            index.upsert([{"id": tid, "values": rand_vec(), "metadata": md2}])
            st.success(f"Approved {md.get('task')}")

# -----------------------------
# ADMIN
# -----------------------------
elif role == "Admin":
    st.header("Admin Dashboard")
    df = fetch_all()
    if df.empty:
        st.warning("No data found.")
    else:
        df["marks"] = pd.to_numeric(df["marks"], errors="coerce")
        df["completion"] = pd.to_numeric(df["completion"], errors="coerce")

        st.subheader("Top Employees by Marks")
        top = df.groupby("employee")["marks"].mean().reset_index().sort_values("marks", ascending=False).head(5)
        st.dataframe(top)

        st.subheader("K-Means Clustering (Performance)")
        if len(df) >= 1:
            n_clusters = min(3, len(df))
            km = KMeans(n_clusters=n_clusters, n_init=10).fit(df[["completion", "marks"]].fillna(0))
            df["cluster"] = km.labels_
            st.plotly_chart(px.scatter(df, x="completion", y="marks", color=df["cluster"].astype(str),
                                       hover_data=["employee", "task"],
                                       title="Employee Task Clusters"))
        else:
            st.info("Not enough data for clustering.")

        avg_m = df["marks"].mean()
        avg_c = df["completion"].mean()
        summary = f"Average marks: {avg_m:.2f}, completion: {avg_c:.1f}%. Top performers: {', '.join(top['employee'].tolist())}."
        st.info(summary)

        st.download_button("Download All Tasks (CSV)", df.to_csv(index=False), "tasks.csv", "text/csv")
