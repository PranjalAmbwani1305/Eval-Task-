import os
import uuid
import time
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Optional: semantic embeddings + Pinecone
USE_PINECONE = True
try:
    import pinecone
    from sentence_transformers import SentenceTransformer
except Exception:
    USE_PINECONE = False

# ---------------------- Helpers & Sample Data ----------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

@st.cache_data
def generate_sample_data(n_employees=8, n_tasks=60):
    np.random.seed(42)
    employees = []
    depts = ["Audit", "Advisory", "Tax", "Consulting"]
    roles = ["Analyst", "Senior", "Manager", "Director"]
    for i in range(n_employees):
        employees.append({
            "employee_id": f"E{i+1}",
            "name": f"Employee {i+1}",
            "department": np.random.choice(depts),
            "role": np.random.choice(roles),
            "skill_score": np.random.randint(50, 100),
            "capacity": np.random.randint(60, 100)  # percentage
        })

    employees = pd.DataFrame(employees)

    tasks = []
    now = datetime.utcnow()
    for t in range(n_tasks):
        est_days = np.random.randint(1, 15)
        assigned = employees.sample(1).iloc[0]
        created = now - timedelta(days=np.random.randint(0, 30))
        due = created + timedelta(days=est_days)
        status = np.random.choice(["Pending", "In Progress", "Completed"], p=[0.2, 0.6, 0.2])
        progress = 100 if status == "Completed" else np.random.randint(5, 95)
        tasks.append({
            "task_id": f"T{t+1}",
            "title": f"Task {t+1} - {np.random.choice(['Audit', 'Model', 'Report', 'Review'])}",
            "assigned_to": assigned["employee_id"],
            "priority": np.random.choice(["Low", "Medium", "High"], p=[0.4, 0.4, 0.2]),
            "created_at": created,
            "due_date": due,
            "status": status,
            "progress": progress,
            "complexity": np.random.randint(1, 10),
            "estimated_hours": np.random.randint(1, 40),
            "client_approved": np.random.choice([True, False], p=[0.6, 0.4])
        })

    tasks = pd.DataFrame(tasks)
    feedback = []
    sample_texts = [
        "Great collaboration and timely delivery.",
        "Needs improvement in documentation.",
        "Consistently excellent analytical thinking.",
        "Communication could be clearer.",
        "Very reliable and takes ownership.",
        "Missed a few deadlines but quality is good."
    ]
    for i in range(40):
        by = employees.sample(1).iloc[0]
        about = employees.sample(1).iloc[0]
        feedback.append({
            "feedback_id": f"F{i+1}",
            "from": by["employee_id"],
            "about": about["employee_id"],
            "text": np.random.choice(sample_texts),
            "created_at": now - timedelta(days=np.random.randint(0, 60))
        })
    feedback = pd.DataFrame(feedback)

    employees.to_csv(os.path.join(DATA_DIR, "employees.csv"), index=False)
    tasks.to_csv(os.path.join(DATA_DIR, "tasks.csv"), index=False)
    feedback.to_csv(os.path.join(DATA_DIR, "feedback.csv"), index=False)

    return employees, tasks, feedback

@st.cache_data
def load_data():
    employees_path = os.path.join(DATA_DIR, "employees.csv")
    tasks_path = os.path.join(DATA_DIR, "tasks.csv")
    feedback_path = os.path.join(DATA_DIR, "feedback.csv")
    if not (os.path.exists(employees_path) and os.path.exists(tasks_path) and os.path.exists(feedback_path)):
        return generate_sample_data()
    employees = pd.read_csv(employees_path)
    tasks = pd.read_csv(tasks_path, parse_dates=["created_at", "due_date"]) 
    feedback = pd.read_csv(feedback_path, parse_dates=["created_at"]) 
    return employees, tasks, feedback

employees, tasks, feedback = load_data()

# ---------------------- ML: Train simple demo models ----------------------
@st.cache_resource
def train_demo_models(tasks_df, employees_df):
    # Predict marks (simulated continuous score) with linear regression
    df = tasks_df.copy()
    # feature: complexity, estimated_hours, progress
    X_reg = df[["complexity", "estimated_hours", "progress"]].fillna(0)
    # synthetic target: higher complexity and more hours -> higher 'marks' (pretend score)
    y_reg = (100 - df["complexity"] * 3 + (df["progress"] * 0.2) + np.random.randn(len(df)) * 5).clip(0, 100)
    lin = LinearRegression().fit(X_reg, y_reg)

    # Task status classifier: Completed (1) vs not (0)
    X_clf = df[["progress", "estimated_hours"]].fillna(0)
    y_clf = (df["status"] == "Completed").astype(int)
    log = LogisticRegression(max_iter=500).fit(X_clf, y_clf)

    # Deadline risk: high risk if due soon and low progress
    X_rf = df[["progress", "estimated_hours", "complexity"]].fillna(0)
    # Create risk label
    days_left = (df["due_date"] - pd.Timestamp.utcnow()).dt.days.fillna(0)
    y_rf = ((days_left < 2) & (df["progress"] < 50)).astype(int)
    rf = RandomForestClassifier(n_estimators=50).fit(X_rf, y_rf)

    # KMeans on employee KPI features
    emp_feats = employees_df[["skill_score", "capacity"]].fillna(0)
    scaler = StandardScaler().fit(emp_feats)
    emp_scaled = scaler.transform(emp_feats)
    n_clusters = min(3, max(1, len(emp_feats)))
    if len(emp_feats) < 2:
        kmeans = None
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(emp_scaled)

    return {
        "linear": lin,
        "logistic": log,
        "rf": rf,
        "kmeans": kmeans,
        "scaler": scaler
    }

models = train_demo_models(tasks, employees)

# ---------------------- Pinecone: Embeddings & Index ----------------------
EMBED_MODEL = None
PINECONE_INDEX_NAME = "feedback-embeds"

def init_pinecone():
    global EMBED_MODEL
    if not USE_PINECONE:
        return None

    api_key = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY", None)
    env = os.getenv("PINECONE_ENV") or st.secrets.get("PINECONE_ENV", None)
    if not api_key or not env:
        st.info("Pinecone not configured. Set PINECONE_API_KEY and PINECONE_ENV in environment or Streamlit secrets to enable semantic search.")
        return None
    try:
        pinecone.init(api_key=api_key, environment=env)
        if PINECONE_INDEX_NAME not in pinecone.list_indexes():
            pinecone.create_index(PINECONE_INDEX_NAME, dimension=384, metric="cosine")
        index = pinecone.Index(PINECONE_INDEX_NAME)
        EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        return index
    except Exception as e:
        st.error(f"Pinecone init failed: {e}")
        return None

pinecone_index = init_pinecone()

def upsert_feedback_to_pinecone(feedback_df):
    if not pinecone_index or EMBED_MODEL is None:
        return
    to_upsert = []
    for _, row in feedback_df.iterrows():
        vec = EMBED_MODEL.encode(row["text"]).tolist()
        meta = {"feedback_id": row["feedback_id"], "from": row["from"], "about": row["about"]}
        to_upsert.append((row["feedback_id"], vec, meta))
    # Pinecone upsert in batches
    batch_size = 50
    for i in range(0, len(to_upsert), batch_size):
        pinecone_index.upsert(vectors=to_upsert[i:i+batch_size])

# optionally upsert sample feedback
if pinecone_index is not None and len(feedback) > 0:
    try:
        upsert_feedback_to_pinecone(feedback)
    except Exception:
        pass

# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="AI Workforce Dashboard — Big 4 Edition", layout="wide")

st.header("AI Workforce Performance & Task Intelligence — Big 4 Edition")

menu = st.sidebar.selectbox("Select module", ["Manager Dashboard", "Task Management", "360 Feedback", "Admin / Exports", "Semantic Search (Pinecone)"])

# ---------------------- Manager Dashboard ----------------------
if menu == "Manager Dashboard":
    st.subheader("Single-pane Manager View")
    total = len(tasks)
    pending = len(tasks[tasks["status"] == "Pending"])
    in_progress = len(tasks[tasks["status"] == "In Progress"])
    completed = len(tasks[tasks["status"] == "Completed"])
    overdue = len(tasks[(tasks["due_date"] < pd.Timestamp.utcnow()) & (tasks["status"] != "Completed")])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Tasks", total)
    col2.metric("In Progress", in_progress)
    col3.metric("Completed", completed)
    col4.metric("Overdue", overdue)

    st.markdown("---")
    st.subheader("Team Performance Heatmap")
    # build performance matrix: rows employees, cols KPI
    perf = employees.set_index("employee_id")[['skill_score','capacity']]
    fig, ax = plt.subplots(figsize=(8, max(2, len(employees) * 0.5)))
    sns.heatmap(perf, annot=True, cmap="YlOrBr", ax=ax)
    st.pyplot(fig)

    st.subheader("AI Alerts")
    # Simple checks
    overload = employees[employees['capacity'] > 85]
    if not overload.empty:
        for _, r in overload.iterrows():
            st.warning(f"High capacity: {r['name']} ({r['employee_id']}) — capacity {r['capacity']}%")
    else:
        st.success("No immediate capacity alerts")

    # Goal tracker: sample
    st.subheader("Goal Tracker")
    target_completed = 30
    prog = min(100, int((completed / target_completed) * 100))
    st.progress(prog)
    st.caption(f"{completed} of {target_completed} tasks completed this period ({prog}%)")

# ---------------------- Task Management ----------------------
elif menu == "Task Management":
    st.subheader("Task Management Panel — Create, Assign, Prioritize")
    with st.form("create_task_form"):
        title = st.text_input("Task title")
        assigned_to = st.selectbox("Assign To", employees['employee_id'].tolist())
        priority = st.selectbox("Priority", ["Low","Medium","High"])
        est_hours = st.number_input("Estimated hours", min_value=1, max_value=200, value=8)
        complexity = st.slider("Complexity (1-10)", 1, 10, 5)
        due_days = st.number_input("Due in (days)", min_value=1, max_value=90, value=7)
        create_btn = st.form_submit_button("Create Task")
        if create_btn:
            new_t = {
                'task_id': f"T{len(tasks)+1}",
                'title': title or f"Task {len(tasks)+1}",
                'assigned_to': assigned_to,
                'priority': priority,
                'created_at': pd.Timestamp.utcnow(),
                'due_date': pd.Timestamp.utcnow() + pd.Timedelta(days=int(due_days)),
                'status': 'Pending',
                'progress': 0,
                'complexity': complexity,
                'estimated_hours': est_hours,
                'client_approved': False
            }
            tasks.loc[len(tasks)] = new_t
            tasks.to_csv(os.path.join(DATA_DIR, "tasks.csv"), index=False)
            st.success("Task created and saved")

    st.markdown("---")
    st.subheader("Task List")
    st.dataframe(tasks.sort_values('due_date'))

    st.markdown("---")
    st.subheader("AI Predictions for Selected Task")
    sel_task_id = st.selectbox("Select task", tasks['task_id'].tolist())
    sel_task = tasks[tasks['task_id'] == sel_task_id].iloc[0]
    X_reg = np.array([[sel_task['complexity'], sel_task['estimated_hours'], sel_task['progress']]])
    pred_marks = models['linear'].predict(X_reg)[0]
    X_clf = np.array([[sel_task['progress'], sel_task['estimated_hours']]])
    pred_status_prob = models['logistic'].predict_proba(X_clf)[0][1]
    X_rf = np.array([[sel_task['progress'], sel_task['estimated_hours'], sel_task['complexity']]])
    pred_deadline_risk = models['rf'].predict_proba(X_rf)[0][1]

    st.metric("Predicted Marks", f"{pred_marks:.1f} / 100")
    st.metric("Probability Completed Soon", f"{pred_status_prob*100:.1f}%")
    st.metric("Deadline Risk", f"{pred_deadline_risk*100:.1f}%")

    st.caption("AI suggestions: use manager actions to reassign or set reminders if deadline risk > 50%")

# ---------------------- 360 Feedback ----------------------
elif menu == "360 Feedback":
    st.subheader("Employee Performance & 360° Feedback Module")
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("**Submit Feedback**")
        fb_from = st.selectbox("From", employees['employee_id'].tolist())
        fb_about = st.selectbox("About", employees['employee_id'].tolist(), index=1)
        fb_text = st.text_area("Feedback text")
        if st.button("Submit Feedback") and fb_text.strip():
            new_fb = {
                'feedback_id': f"F{len(feedback)+1}",
                'from': fb_from,
                'about': fb_about,
                'text': fb_text,
                'created_at': pd.Timestamp.utcnow()
            }
            feedback.loc[len(feedback)] = new_fb
            feedback.to_csv(os.path.join(DATA_DIR, "feedback.csv"), index=False)
            # upsert to pinecone if available
            try:
                if pinecone_index and EMBED_MODEL:
                    vec = EMBED_MODEL.encode(fb_text).tolist()
                    pinecone_index.upsert(vectors=[(new_fb['feedback_id'], vec, {'from': fb_from, 'about': fb_about})])
            except Exception:
                pass
            st.success("Feedback submitted")

    with col2:
        st.markdown("**Aggregate Sentiment (quick heuristic)**")
        # quick sentiment heuristic: positive words
        positive = feedback['text'].str.contains("great|excellent|reliable|timely|ownership", case=False, na=False).sum()
        negative = feedback['text'].str.contains("missed|needs|improvement|unclear", case=False, na=False).sum()
        st.write(f"Positive mentions: {positive}")
        st.write(f"Negative mentions: {negative}")

    st.markdown("---")
    st.subheader("Feedback Table")
    st.dataframe(feedback.sort_values('created_at', ascending=False))

# ---------------------- Admin / Exports ----------------------
elif menu == "Admin / Exports":
    st.subheader("Admin Dashboard & Exports")
    st.markdown("**Employee stats**")
    st.dataframe(employees)

    st.markdown("**K-Means clustering on employees (safe fallback for 1 record)**")
    emp_feats = employees[["skill_score","capacity"]].fillna(0)
    if len(emp_feats) < 2:
        st.info("Not enough records for clustering. Showing single-cluster fallback.")
        employees['cluster'] = 0
    else:
        scaled = models['scaler'].transform(emp_feats)
        k = min(3, len(emp_feats))
        kmeans = KMeans(n_clusters=k, random_state=42).fit(scaled)
        employees['cluster'] = kmeans.labels_
    st.dataframe(employees)

    st.markdown("---")
    st.subheader("Export Summary + CSV")
    summary = {
        'total_tasks': len(tasks),
        'pending': len(tasks[tasks['status']=='Pending']),
        'in_progress': len(tasks[tasks['status']=='In Progress']),
        'completed': len(tasks[tasks['status']=='Completed']),
        'overdue': len(tasks[(tasks['due_date'] < pd.Timestamp.utcnow()) & (tasks['status'] != 'Completed')])
    }
    st.json(summary)

    if st.button("Download full CSV export"):
        export_df = tasks.merge(employees, left_on='assigned_to', right_on='employee_id', how='left')
        csv = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download CSV", data=csv, file_name="tasks_export.csv", mime='text/csv')

# ---------------------- Semantic Search / Pinecone ----------------------
elif menu == "Semantic Search (Pinecone)":
    st.subheader("Semantic Search over Feedback (Pinecone)")
    if not pinecone_index or EMBED_MODEL is None:
        st.info("Pinecone not configured or missing embedding model. Please set PINECONE_API_KEY and PINECONE_ENV and install sentence-transformers.")
    else:
        q = st.text_input("Search feedback (natural language)")
        top_k = st.slider("Top K", 1, 10, 5)
        if st.button("Search") and q.strip():
            qv = EMBED_MODEL.encode(q).tolist()
            try:
                res = pinecone_index.query(vector=qv, top_k=top_k, include_metadata=True)
                hits = res['matches']
                out = []
                for h in hits:
                    out.append({
                        'feedback_id': h['id'],
                        'score': h['score'],
                        'from': h['metadata'].get('from'),
                        'about': h['metadata'].get('about')
                    })
                st.table(pd.DataFrame(out))
            except Exception as e:
                st.error(f"Query failed: {e}")

# ---------------------- Footer ----------------------
st.markdown("---")
st.caption("Demo app: replace demo models with production models, secure API keys and use a proper DB for persistence.")
