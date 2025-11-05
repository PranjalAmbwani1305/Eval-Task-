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
# CONFIG
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

# -----------------------------
# HELPERS
# -----------------------------
def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def rand_vec():
    return np.random.rand(DIMENSION).tolist()

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
# AI MODELS
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
# ROLE SELECTION
# -----------------------------
role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client", "Admin"])
current_month = datetime.now().strftime("%B %Y")

# -----------------------------
# MANAGER
# -----------------------------
if role == "Manager":
    st.header("Manager Dashboard")
    tab1, tab2 = st.tabs(["Assign Task", "Review Client-Approved"])

    # ----- ASSIGN -----
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

    # ----- REVIEW -----
    with tab2:
        st.subheader("Client-Approved Tasks")
        df = fetch_all()

        if not df.empty:
            # Normalize booleans
            df["client_reviewed"] = df["client_reviewed"].apply(
                lambda x: str(x).strip().lower() in ["true", "1", "yes"]
            )
            df = df[df["client_reviewed"] == True]

        if df.empty:
            st.warning("No client-approved tasks found yet. Approve a task in the Client panel first.")
        else:
            for _, r in df.iterrows():
                st.markdown(f"### {r.get('task', 'Untitled Task')}")
                st.write(f"Employee: {r.get('employee', '')}")
                st.write(f"Client Comments: {r.get('client_comments', 'No comments')}")

                marks = st.number_input(f"Marks (0–5)", 0.0, 5.0, float(r.get("marks", 0)), step=0.1, key=f"m_{r['_id']}")
                comments = st.text_area("Manager Comments", key=f"c_{r['_id']}")

                if st.button("Finalize Review", key=f"b_{r['_id']}"):
                    sent_val = int(svm_clf.predict(vectorizer.transform([comments]))[0])
                    sent = "Positive" if sent_val == 1 else "Negative"
                    md = safe_meta({
                        **r,
                        "marks": marks,
                        "manager_comments": comments,
                        "reviewed": True,
                        "sentiment": sent,
                        "reviewed_on": now()
                    })
                    index.upsert([{"id": r["_id"], "values": rand_vec(), "metadata": md}])
                    st.success(f"Review saved for '{r['task']}' ({sent})")
                    st.experimental_rerun()

# -----------------------------
# TEAM MEMBER
# -----------------------------
elif role == "Team Member":
    st.header("Team Member Progress Update")
    company = st.text_input("Company")
    employee = st.text_input("Your Name")

    if st.button("Load Tasks"):
        res = index.query(
            vector=rand_vec(), top_k=500, include_metadata=True,
            filter={"company": {"$eq": company}, "employee": {"$eq": employee}}
        )
        st.session_state["tasks"] = [(m.id, m.metadata) for m in res.matches or []]
        st.success(f"Loaded {len(st.session_state['tasks'])} tasks.")

    tasks = st.session_state.get("tasks", [])
    if tasks:
        for tid, md in tasks:
            st.subheader(md.get("task"))
            st.write(md.get("description"))
            curr = float(md.get("completion", 0))
            new = st.slider(f"Completion (%)", 0, 100, int(curr), key=f"s_{tid}")
            if st.button("Submit Update", key=tid):
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
                st.success(f"Progress updated ({track}, Risk={md2['deadline_risk']})")

# -----------------------------
# CLIENT
# -----------------------------
elif role == "Client":
    st.header("Client Review")
    company = st.text_input("Company Name")

    if st.button("Load Completed Tasks"):
        res = index.query(
            vector=rand_vec(), top_k=500, include_metadata=True,
            filter={"company": {"$eq": company}, "completion": {"$gte": 95}}
        )
        st.session_state["ctasks"] = [(m.id, m.metadata) for m in res.matches or []]
        st.success(f"Loaded {len(st.session_state['ctasks'])} tasks.")

    for tid, md in st.session_state.get("ctasks", []):
        st.subheader(md.get("task"))
        st.write(f"Employee: {md.get('employee')}")
        st.write(f"Completion: {md.get('completion')}%")
        comment = st.text_area("Feedback", key=f"f_{tid}")
        if st.button("Approve Task", key=f"a_{tid}"):
            md2 = safe_meta({
                **md,
                "client_reviewed": True,
                "client_comments": comment,
                "client_approved_on": now()
            })
            index.upsert([{"id": tid, "values": rand_vec(), "metadata": md2}])
            st.success(f"Task '{md.get('task')}' approved.")
            st.experimental_rerun()

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
        if len(df) >= 3:
            km = KMeans(n_clusters=3, n_init="auto").fit(df[["completion", "marks"]].fillna(0))
            df["cluster"] = km.labels_
            st.plotly_chart(px.scatter(df, x="completion", y="marks", color=df["cluster"].astype(str),
                                       hover_data=["employee", "task"], title="Employee Task Clusters"))
        else:
            st.info("Not enough data for clustering (need at least 3 tasks).")

        avg_m = df["marks"].mean()
        avg_c = df["completion"].mean()
        summary = f"Average marks: {avg_m:.2f}, completion: {avg_c:.1f}%. Top performers: {', '.join(top['employee'].tolist())}."
        st.info(summary)

        st.download_button("Download All Tasks (CSV)", df.to_csv(index=False), "tasks.csv", "text/csv")
