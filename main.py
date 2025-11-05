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

# Create index if not exists
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

def stable_vec(seed_text):
    np.random.seed(abs(hash(seed_text)) % (2**32))
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

def fetch_filtered(filter_dict=None):
    """Fetch filtered results from Pinecone deterministically."""
    try:
        res = index.query(
            vector=np.zeros(DIMENSION).tolist(),
            top_k=10000,
            include_metadata=True,
            filter=filter_dict or {}
        )
        if not res.matches:
            return pd.DataFrame()
        rows = []
        for m in res.matches:
            md = m.metadata or {}
            md["_id"] = m.id
            rows.append(md)
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def fetch_all():
    """Admin-only: fetch all records without filters."""
    try:
        res = index.query(
            vector=np.zeros(DIMENSION).tolist(),
            top_k=10000,
            include_metadata=True
        )
        if not res.matches:
            return pd.DataFrame()
        rows = []
        for m in res.matches:
            md = m.metadata or {}
            md["_id"] = m.id
            rows.append(md)
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Error fetching all data: {e}")
        return pd.DataFrame()

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
# ROLE SELECTION
# -----------------------------
role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client", "Admin"])
current_month = datetime.now().strftime("%B %Y")

# -----------------------------
# MANAGER (BOSS)
# -----------------------------
if role == "Manager":
    st.header("Manager (Boss) Dashboard")
    tab1, tab2 = st.tabs(["Assign Task", "Boss Review & Adjustment"])

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
                index.upsert([{"id": tid, "values": stable_vec(task), "metadata": md}])
                st.success(f"Task '{task}' assigned to {employee}")

    # Boss Review Section
    with tab2:
        company_name = st.text_input("Filter by Company Name", "")
        if company_name:
            df = fetch_filtered({"company": {"$eq": company_name}})
        else:
            df = pd.DataFrame()

        if df.empty:
            st.info("No tasks found for this company.")
        else:
            for _, r in df.iterrows():
                st.markdown(f"### {r['task']}")
                st.write(f"**Employee:** {r.get('employee', 'Unknown')}")
                st.write(f"**Reported Completion:** {r.get('completion', 0)}%")
                st.write(f"**Current Marks:** {r.get('marks', 0):.2f}")
                st.write(f"**Deadline Risk:** {r.get('deadline_risk', 'N/A')}")

                adjusted_completion = st.slider(
                    f"Adjust Completion % for {r['task']}",
                    0, 100, int(r.get("completion", 0)), key=f"adj_{r['_id']}"
                )

                adjusted_marks = float(lin_reg.predict([[adjusted_completion]])[0])
                comments = st.text_area(f"Boss Comments for {r['task']}", key=f"boss_cmt_{r['_id']}")
                approve = st.radio(f"Approve Task {r['task']}?", ["Yes", "No"], key=f"boss_app_{r['_id']}")

                if st.button(f"Finalize Review for {r['task']}", key=f"final_{r['_id']}"):
                    sentiment_val = int(svm_clf.predict(vectorizer.transform([comments]))[0])
                    sentiment = "Positive" if sentiment_val == 1 else "Negative"
                    md = safe_meta({
                        **r,
                        "completion": adjusted_completion,
                        "marks": adjusted_marks,
                        "manager_comments": comments,
                        "reviewed": True,
                        "sentiment": sentiment,
                        "approved_by_boss": approve == "Yes",
                        "reviewed_on": now()
                    })
                    index.upsert([{"id": r["_id"], "values": stable_vec(r['task']), "metadata": md}])
                    st.success(f"Review finalized for {r['task']} ({sentiment}).")

# -----------------------------
# TEAM MEMBER
# -----------------------------
elif role == "Team Member":
    st.header("Team Member Progress Update")
    company = st.text_input("Company")
    employee = st.text_input("Your Name")

    if st.button("Load Tasks"):
        if not company or not employee:
            st.warning("Please enter both company and your name.")
        else:
            df = fetch_filtered({"company": {"$eq": company}, "employee": {"$eq": employee}})
            if df.empty:
                st.info("No tasks found for you.")
            else:
                st.session_state["tasks"] = df.to_dict("records")
                st.success(f"Loaded {len(df)} tasks.")

    for r in st.session_state.get("tasks", []):
        st.subheader(r.get("task"))
        st.write(r.get("description"))
        curr = float(r.get("completion", 0))
        new = st.slider(f"Completion for {r.get('task')}", 0, 100, int(curr), key=r["_id"])
        if st.button(f"Submit {r.get('task')}", key=f"sub_{r['_id']}"):
            marks = float(lin_reg.predict([[new]])[0])
            track = "On Track" if log_reg.predict([[new]])[0] == 1 else "Delayed"
            miss = rf.predict([[new, 0]])[0]
            md2 = safe_meta({
                **r,
                "completion": new,
                "marks": marks,
                "status": track,
                "deadline_risk": "High" if miss else "Low",
                "submitted_on": now(),
                "client_reviewed": False
            })
            index.upsert([{"id": r["_id"], "values": stable_vec(r['task']), "metadata": md2}])
            st.success(f"Updated {r.get('task')} ({track}, Risk={md2['deadline_risk']})")

# -----------------------------
# CLIENT
# -----------------------------
elif role == "Client":
    st.header("Client Review")
    company = st.text_input("Company Name")

    if st.button("Load Completed"):
        if not company:
            st.warning("Enter company name.")
        else:
            df = fetch_filtered({"company": {"$eq": company}, "reviewed": {"$eq": True}})
            if df.empty:
                st.info("No reviewed tasks yet.")
            else:
                st.session_state["ctasks"] = df.to_dict("records")
                st.success(f"Loaded {len(df)} reviewed tasks.")

    for r in st.session_state.get("ctasks", []):
        st.subheader(r.get("task"))
        st.write(f"Employee: {r.get('employee')}")
        st.write(f"Final Completion: {r.get('completion')}%")
        st.write(f"Boss-Assigned Marks: {r.get('marks')}")
        comment = st.text_area(f"Feedback for {r.get('task')}", key=f"c_{r['_id']}")
        if st.button(f"Approve {r.get('task')}", key=f"approve_{r['_id']}"):
            md2 = safe_meta({
                **r,
                "client_reviewed": True,
                "client_comments": comment,
                "client_approved_on": now()
            })
            index.upsert([{"id": r["_id"], "values": stable_vec(r['task']), "metadata": md2}])
            st.success(f"Approved {r.get('task')}")

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
        if len(df) > 2:
            n_clusters = min(3, len(df))
            km = KMeans(n_clusters=n_clusters, n_init=10).fit(df[["completion", "marks"]].fillna(0))
            df["cluster"] = km.labels_
            st.plotly_chart(px.scatter(
                df, x="completion", y="marks",
                color=df["cluster"].astype(str),
                hover_data=["employee", "task"],
                title="Employee Task Clusters"
            ))
        else:
            st.info("Not enough data for clustering.")

        avg_m = df["marks"].mean()
        avg_c = df["completion"].mean()
        summary = f"Average marks: {avg_m:.2f}, completion: {avg_c:.1f}%. Top performers: {', '.join(top['employee'].tolist())}."
        st.info(summary)

        st.download_button("Download All Tasks (CSV)", df.to_csv(index=False), "tasks.csv", "text/csv")
