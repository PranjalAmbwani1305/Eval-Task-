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
st.set_page_config(page_title="AI Task System", layout="wide")
st.title("AI-Powered Task Management System")

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "task"

# Try to connect to existing index
try:
    index = pc.Index(index_name)
except Exception:
    st.error("Could not connect to Pinecone index. Make sure it exists.")
    st.stop()

def now(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def rand_vec(): return np.random.rand(64).tolist()

# -----------------------------
# SIMPLE AI MODELS
# -----------------------------
lin_reg = LinearRegression()
lin_reg.fit([[0], [50], [100]], [0, 2.5, 5])

log_reg = LogisticRegression()
log_reg.fit([[0], [50], [100]], [0, 0, 1])

rf = RandomForestClassifier()
rf.fit([[10, 2], [50, 1], [90, 0], [100, 0]], [1, 0, 0, 0])

vectorizer = CountVectorizer()
comments = ["great job", "bad work", "average performance", "excellent effort", "needs improvement"]
sentiments = [1, 0, 0, 1, 0]
X_train = vectorizer.fit_transform(comments)
svm = SVC()
svm.fit(X_train, sentiments)

# -----------------------------
# HELPERS
# -----------------------------
def safe_meta(md):
    clean = {}
    for k, v in md.items():
        if isinstance(v, (datetime, date)):
            clean[k] = v.isoformat()
        elif v is None:
            clean[k] = ""
        else:
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
        df = pd.DataFrame(rows)
        return df
    except Exception as e:
        st.error(f"Error fetching from Pinecone: {e}")
        return pd.DataFrame()

# -----------------------------
# ROLE SELECTION
# -----------------------------
role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client", "Admin"])
current_month = datetime.now().strftime("%B %Y")

# -----------------------------
# MANAGER DASHBOARD
# -----------------------------
if role == "Manager":
    st.header("Manager Dashboard")
    tab1, tab2 = st.tabs(["Assign Task", "Review Tasks"])

    # --- Assign Task ---
    with tab1:
        with st.form("assign_form"):
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
                    "deadline": deadline,
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
            # Ensure safety for missing columns
            for col in ["manager_reviewed", "completion", "marks", "deadline_risk"]:
                if col not in df.columns:
                    df[col] = ""
            pending = df[df["manager_reviewed"].astype(str).str.lower() != "true"]

            if pending.empty:
                st.info("No pending reviews.")
            else:
                for _, r in pending.iterrows():
                    st.subheader(r.get("task", "Untitled Task"))
                    st.write(f"Employee: {r.get('employee','')}")
                    st.write(f"Reported Completion: {r.get('completion',0)}%")
                    st.write(f"Current Marks: {r.get('marks',0)}")
                    st.write(f"Deadline Risk: {r.get('deadline_risk','N/A')}")

                    adj = st.slider(f"Adjust Completion % for {r.get('task')}",
                                    0, 100, int(float(r.get('completion', 0))), key=f"adj_{r['_id']}")
                    boss_comments = st.text_area(f"Boss Comments for {r.get('task')}", key=f"comm_{r['_id']}")
                    approve = st.radio(f"Approve Task {r.get('task')}?",
                                       ["Yes", "No"], key=f"appr_{r['_id']}")

                    if st.button(f"Finalize {r.get('task')}", key=f"final_{r['_id']}"):
                        marks = float(lin_reg.predict([[adj]])[0])
                        sent_val = int(svm.predict(vectorizer.transform([boss_comments]))[0])
                        sent = "Positive" if sent_val == 1 else "Negative"
                        md = safe_meta({
                            **r,
                            "completion": adj,
                            "marks": marks,
                            "manager_comments": boss_comments,
                            "manager_reviewed": True,
                            "manager_approved": (approve == "Yes"),
                            "sentiment": sent,
                            "reviewed_on": now()
                        })
                        index.upsert([{"id": r["_id"], "values": rand_vec(), "metadata": md}])
                        st.success(f"Review finalized for {r.get('task')} ({sent})")

# -----------------------------
# TEAM MEMBER DASHBOARD
# -----------------------------
elif role == "Team Member":
    st.header("Team Member Dashboard")
    company = st.text_input("Company")
    employee = st.text_input("Your Name")
    if st.button("Load Tasks"):
        res = index.query(vector=rand_vec(), top_k=500, include_metadata=True)
        st.session_state["tasks"] = [(m.id, m.metadata)
                                     for m in res.matches
                                     if m.metadata.get("company") == company and m.metadata.get("employee") == employee]
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
                "submitted_on": now()
            })
            index.upsert([{"id": tid, "values": rand_vec(), "metadata": md2}])
            st.success(f"Updated {md.get('task')} ({track})")

# -----------------------------
# CLIENT DASHBOARD
# -----------------------------
elif role == "Client":
    st.header("Client Dashboard")
    company = st.text_input("Company Name")
    if st.button("Load Reviewed Tasks"):
        df = fetch_all()
        if not df.empty:
            for col in ["manager_reviewed", "client_reviewed"]:
                if col not in df.columns:
                    df[col] = False
            df = df[
                (df["company"].astype(str) == company)
                & (df["manager_reviewed"].astype(str).str.lower() == "true")
                & (df["client_reviewed"].astype(str).str.lower() != "true")
            ]
            st.session_state["client_tasks"] = df.to_dict("records")
            st.success(f"Loaded {len(df)} tasks.")
        else:
            st.warning("No data found.")

    for r in st.session_state.get("client_tasks", []):
        st.subheader(r.get("task"))
        st.write(f"Employee: {r.get('employee')}")
        st.write(f"Completion: {r.get('completion')}%")
        st.write(f"Manager Comments: {r.get('manager_comments','')}")
        comment = st.text_area(f"Feedback for {r.get('task')}", key=f"fb_{r['_id']}")
        approve = st.radio(f"Approve {r.get('task')}?", ["Yes", "No"], key=f"appr_c_{r['_id']}")
        if st.button(f"Submit Client Review {r.get('task')}", key=f"cl_{r['_id']}"):
            md2 = safe_meta({
                **r,
                "client_reviewed": True,
                "client_approved": (approve == "Yes"),
                "client_comments": comment,
                "client_approved_on": now()
            })
            index.upsert([{"id": r["_id"], "values": rand_vec(), "metadata": md2}])
            st.success(f"Client reviewed {r.get('task')}")

# -----------------------------
# ADMIN DASHBOARD
# -----------------------------
elif role == "Admin":
    st.header("Admin Dashboard")
    df = fetch_all()
    if df.empty:
        st.warning("No data found.")
    else:
        # Safe conversion
        for col in ["marks", "completion"]:
            df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)

        st.subheader("Top Employees by Marks")
        top = df.groupby("employee")["marks"].mean().reset_index().sort_values("marks", ascending=False)
        st.dataframe(top)

        st.subheader("Performance Clustering")
        if len(df) >= 3:
            km = KMeans(n_clusters=min(3, len(df)), n_init=10).fit(df[["completion", "marks"]])
            df["cluster"] = km.labels_
            st.plotly_chart(px.scatter(df, x="completion", y="marks", color=df["cluster"].astype(str),
                                       hover_data=["employee", "task"], title="Employee Task Clusters"))
        else:
            st.info("Not enough data for K-Means clustering.")

        st.subheader("Download Data")
        st.download_button("Download CSV", df.to_csv(index=False), "tasks.csv", "text/csv")
