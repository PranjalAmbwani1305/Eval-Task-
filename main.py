# ===========================
# Agentic Workforce Dashboard
# ===========================
import streamlit as st
import numpy as np
import pandas as pd
import uuid
import time
from datetime import datetime, date
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import plotly.express as px
import plotly.graph_objects as go

# Optional AI
try:
    from langchain import HuggingFaceHub
    from langchain.agents import initialize_agent, Tool
    from langchain.memory import ConversationBufferMemory
    HF_AVAILABLE = True
except:
    HF_AVAILABLE = False

try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except:
    PINECONE_AVAILABLE = False

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="AI Workforce Dashboard", layout="wide")
st.title("🤖 AI-Powered Workforce Performance & Task Management Dashboard")

# ------------------------------
# GLOBAL VARIABLES
# ------------------------------
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
HF_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")
INDEX_NAME = "task"
DIMENSION = 1024

# ------------------------------
# PINECONE INITIALIZATION (SAFE)
# ------------------------------
if PINECONE_AVAILABLE and PINECONE_API_KEY:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        existing = [i["name"] for i in pc.list_indexes()]
        if INDEX_NAME not in existing:
            st.info("⏳ Creating Pinecone index, please wait ~1 minute...")
            pc.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while True:
                desc = pc.describe_index(INDEX_NAME)
                if desc["status"]["ready"]:
                    break
                time.sleep(5)
                st.write("... still initializing index ...")

        index = pc.Index(INDEX_NAME)
        st.success("✅ Pinecone connected successfully!")
    except Exception as e:
        st.warning(f"Pinecone initialization failed: {e}")
        index = None
else:
    st.warning("⚠️ Pinecone API Key not found — running in local mode.")
    index = None

# ------------------------------
# HELPERS
# ------------------------------
def now_iso(): return datetime.now().isoformat()
def rand_vec(): return np.random.rand(DIMENSION).tolist()

def safe_upsert(id_, metadata):
    """Store metadata in Pinecone or local memory"""
    if index:
        try:
            index.upsert([{"id": id_, "values": rand_vec(), "metadata": metadata}])
        except Exception as e:
            st.error(f"Pinecone upsert failed: {e}")
    else:
        st.session_state.LOCAL_TASKS[id_] = metadata

def fetch_all():
    """Safely fetch all tasks"""
    if not index:
        return pd.DataFrame(list(st.session_state.LOCAL_TASKS.values()))

    try:
        stats = index.describe_index_stats()
        if stats["total_vector_count"] == 0:
            return pd.DataFrame()
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
        st.warning(f"⚠️ Pinecone fetch error: {e}")
        return pd.DataFrame()

# Initialize local session memory
if "LOCAL_TASKS" not in st.session_state:
    st.session_state.LOCAL_TASKS = {}

# ------------------------------
# BASIC ML MODELS
# ------------------------------
lin_reg = LinearRegression().fit([[0], [50], [100]], [0, 2.5, 5])
log_reg = LogisticRegression().fit([[0], [40], [80], [100]], [0, 0, 1, 1])
rf = RandomForestClassifier().fit(np.array([[10, 2], [50, 1], [90, 0]]), [1, 0, 0])

vectorizer = CountVectorizer()
svm = SVC()
X = vectorizer.fit_transform(["excellent", "good", "bad", "poor"])
svm.fit(X, [1, 1, 0, 0])

# ------------------------------
# ROLE SELECTION
# ------------------------------
role = st.sidebar.selectbox("Login as", ["Manager", "HR", "Employee", "Admin"])
current_month = datetime.now().strftime("%B %Y")

# ====================================================
# 👑 MANAGER COMMAND CENTER
# ====================================================
if role == "Manager":
    st.header("👑 Manager Command Center")

    # Task Assign & Reassign
    st.subheader("📋 Assign / Reassign Tasks")

    df = fetch_all()
    depts = ["IT", "HR", "Finance", "Marketing", "R&D"]
    employees = sorted(df["employee"].dropna().unique().tolist()) if not df.empty else []

    mode = st.radio("Select Mode", ["Assign Task", "Reassign Task"], horizontal=True)

    if mode == "Assign Task":
        with st.form("assign"):
            company = st.text_input("🏢 Company")
            dept = st.selectbox("🏬 Department", depts)
            emp = st.text_input("👤 Employee")
            task = st.text_input("📌 Task Title")
            desc = st.text_area("🗒️ Description")
            deadline = st.date_input("📅 Deadline", value=date.today())
            if st.form_submit_button("✅ Assign"):
                tid = str(uuid.uuid4())
                md = {
                    "company": company,
                    "department": dept,
                    "employee": emp,
                    "task": task,
                    "description": desc,
                    "deadline": deadline.isoformat(),
                    "assigned_on": now_iso(),
                    "status": "Assigned",
                    "completion": 0,
                    "marks": 0,
                }
                safe_upsert(tid, md)
                st.success(f"✅ Task '{task}' assigned to {emp}")
    else:
        if df.empty:
            st.warning("No tasks available for reassignment.")
        else:
            selected = st.selectbox("Select Task", df["task"].unique())
            new_emp = st.text_input("Reassign to Employee")
            if st.button("🔁 Confirm Reassign"):
                row = df[df["task"] == selected].iloc[0].to_dict()
                row["employee"] = new_emp
                row["status"] = "Reassigned"
                row["reassigned_on"] = now_iso()
                safe_upsert(str(uuid.uuid4()), row)
                st.success(f"🔁 Task '{selected}' reassigned to {new_emp}")

    # -----------------------------------------------
    # 📊 Task Summary + Goal Tracker + Heatmap
    # -----------------------------------------------
    st.markdown("---")
    st.subheader("📊 Task Summary, 🎯 Goals & 🔥 Heatmap")

    df = fetch_all()
    if df.empty:
        st.info("No task data found.")
    else:
        df["completion"] = pd.to_numeric(df.get("completion", 0), errors="coerce").fillna(0)
        df["marks"] = pd.to_numeric(df.get("marks", 0), errors="coerce").fillna(0)

        c1, c2, c3 = st.columns(3)
        c1.metric("Average Completion", f"{df['completion'].mean():.2f}%")
        c2.metric("Average Marks", f"{df['marks'].mean():.2f}")
        c3.metric("Total Tasks", len(df))

        dept_summary = df.groupby("department")[["completion", "marks"]].mean().reset_index()
        st.plotly_chart(px.bar(dept_summary, x="department", y="completion", color="department", title="Average Completion by Department"))

        goals = df.groupby("employee")["completion"].mean().reset_index()
        goals["Target"] = 80
        st.plotly_chart(px.bar(goals, x="employee", y=["completion", "Target"], barmode="group", title="Employee Progress vs Target (80%)"))

        heatmap = df.pivot_table(values="marks", index="employee", columns="department", aggfunc="mean").fillna(0)
        st.plotly_chart(px.imshow(heatmap, text_auto=True, color_continuous_scale="Blues", title="Employee vs Department Heatmap (Marks)"))

    # -----------------------------------------------
    # 🤖 Agentic AI & Gen AI Insights
    # -----------------------------------------------
    st.markdown("---")
    st.subheader("🤖 Agentic AI & Gen AI Insights")

    if HF_AVAILABLE and HF_TOKEN:
        llm = HuggingFaceHub(
            repo_id="mistralai/Mixtral-8x7B-Instruct",
            model_kwargs={"temperature": 0.3, "max_new_tokens": 256},
            huggingfacehub_api_token=HF_TOKEN,
        )
        tools = []
        memory = ConversationBufferMemory()
        agent = initialize_agent(tools, llm, agent_type="conversational-react-description", memory=memory)
        query = st.text_input("Ask the AI", "Who are top performers this week?")
        if st.button("Run Agent"):
            with st.spinner("🤖 Thinking..."):
                res = agent.run(query)
                st.success(res)
    else:
        st.info("⚙️ Hugging Face token not available — skipping Agent.")

    # -----------------------------------------------
    # 📆 Leave Management
    # -----------------------------------------------
    st.markdown("---")
    st.subheader("📆 Leave Management")
    st.write("Managers can view and approve team leaves here (future expansion).")

# ====================================================
# HR DASHBOARD
# ====================================================
elif role == "HR":
    st.header("👥 HR Dashboard")
    st.write("View leaves, attrition risk, and department analytics.")

# ====================================================
# EMPLOYEE DASHBOARD
# ====================================================
elif role == "Employee":
    st.header("👨‍💻 Employee Dashboard")
    st.write("View and update your assigned tasks.")

# ====================================================
# ADMIN DASHBOARD
# ====================================================
elif role == "Admin":
    st.header("🛠️ Admin Dashboard")
    st.write("System monitoring and configuration.")
