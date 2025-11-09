# app.py
"""
Agentic Workforce Intelligence - Streamlit prototype
Single-file demo for Manager Command Center + AI integrations.
"""

import streamlit as st
from datetime import datetime, date
import uuid, json, time, logging
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ML
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

# Optional agent / LLM
try:
    from langchain import HuggingFaceHub
    from langchain.agents import initialize_agent, Tool
    from langchain.memory import ConversationBufferMemory
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# Pinecone
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except Exception:
    PINECONE_AVAILABLE = False

# Sentence embedding (local)
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except Exception:
    SBERT_AVAILABLE = False

# ---------------------------
# Config & secrets
# ---------------------------
st.set_page_config(page_title="Agentic Workforce Intelligence", layout="wide")
st.title("Agentic Workforce Intelligence — Manager Dashboard (Prototype)")

# Load secrets (expect in .streamlit/secrets.toml)
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
HF_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")

# Pinecone client setup (if available)
if PINECONE_AVAILABLE and PINECONE_API_KEY:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    TASK_INDEX_NAME = "awi-task"
    FEEDBACK_INDEX_NAME = "awi-feedback"
    LEAVE_INDEX_NAME = "awi-leave"
    # Create indexes if not exist (safely)
    for idx in [TASK_INDEX_NAME, FEEDBACK_INDEX_NAME, LEAVE_INDEX_NAME]:
        if idx not in [i["name"] for i in pc.list_indexes()]:
            pc.create_index(name=idx, dimension=512, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
    task_index = pc.Index(TASK_INDEX_NAME)
    feedback_index = pc.Index(FEEDBACK_INDEX_NAME)
    leave_index = pc.Index(LEAVE_INDEX_NAME)
else:
    # Fallback local in-memory store (dev only)
    PINECONE_API_KEY = ""
    task_index = None
    feedback_index = None
    leave_index = None
    st.warning("Pinecone not configured — using in-memory store for demo (not persistent).")
    if "LOCAL_TASKS" not in st.session_state:
        st.session_state.LOCAL_TASKS = {}   # id -> metadata
        st.session_state.LOCAL_FEEDBACK = {}
        st.session_state.LOCAL_LEAVES = {}

# ---------------------------
# Utility helpers
# ---------------------------
def now_iso():
    return datetime.now().isoformat()

def rand_vec(dim=512):
    return np.random.rand(dim).tolist()

def safe_meta(md: dict):
    clean = {}
    for k,v in md.items():
        if isinstance(v, (np.generic, np.number)):
            clean[k] = float(v)
        elif isinstance(v, (dict, list)):
            clean[k] = json.dumps(v)
        elif isinstance(v, (datetime,)):
            clean[k] = v.isoformat()
        else:
            clean[k] = v
    return clean

def upsert_task_local(id_, md):
    st.session_state.LOCAL_TASKS[id_] = md

def upsert_feedback_local(id_, md):
    st.session_state.LOCAL_FEEDBACK[id_] = md

def upsert_leave_local(id_, md):
    st.session_state.LOCAL_LEAVES[id_] = md

def fetch_tasks():
    if PINECONE_AVAILABLE and task_index:
        res = task_index.query(vector=rand_vec(), top_k=1000, include_metadata=True)
        return pd.DataFrame([m.metadata for m in res.matches])
    else:
        return pd.DataFrame(list(st.session_state.LOCAL_TASKS.values()))

def fetch_feedback():
    if PINECONE_AVAILABLE and feedback_index:
        res = feedback_index.query(vector=rand_vec(), top_k=1000, include_metadata=True)
        return pd.DataFrame([m.metadata for m in res.matches])
    else:
        return pd.DataFrame(list(st.session_state.LOCAL_FEEDBACK.values()))

def fetch_leaves():
    if PINECONE_AVAILABLE and leave_index:
        res = leave_index.query(vector=rand_vec(), top_k=1000, include_metadata=True)
        return pd.DataFrame([m.metadata for m in res.matches])
    else:
        return pd.DataFrame(list(st.session_state.LOCAL_LEAVES.values()))

def safe_upsert(index_name, id_, md):
    # index_name: 'task','feedback','leave'
    if PINECONE_AVAILABLE and task_index:
        idx = {"task": task_index, "feedback": feedback_index, "leave": leave_index}.get(index_name)
        if idx:
            idx.upsert([{"id": id_, "values": rand_vec(), "metadata": safe_meta(md)}])
            return True
    # fallback local
    if index_name == "task":
        upsert_task_local(id_, md)
    elif index_name == "feedback":
        upsert_feedback_local(id_, md)
    elif index_name == "leave":
        upsert_leave_local(id_, md)
    return True

# ---------------------------
# Lightweight ML models (demo)
# ---------------------------
lin_reg = LinearRegression().fit([[0],[50],[100]], [0,2.5,5])
log_reg = LogisticRegression().fit([[0],[40],[80],[100]], [0,0,1,1])
rf = RandomForestClassifier().fit(np.array([[10,2],[50,1],[90,0]]), [1,0,0])

# sentiment baseline
vectorizer = CountVectorizer()
svm_tf = SVC()
# quick fit on tiny dataset (demo) - in production, replace with HF fine-tuned model
try:
    X = vectorizer.fit_transform(["excellent work","needs improvement","bad job","great"])
    svm_tf.fit(X, [1,0,0,1])
except Exception:
    pass

# clustering
def run_kmeans(df, n=3):
    if len(df) < n:
        return df.assign(cluster=0)
    km = KMeans(n_clusters=n, random_state=42)
    df_ = df.copy()
    df_[["completion","marks"]] = df_[["completion","marks"]].fillna(0)
    df_["cluster"] = km.fit_predict(df_[["completion","marks"]])
    return df_

# ---------------------------
# Agentic / HF setup (optional)
# ---------------------------
AGENT_AVAILABLE = HF_AVAILABLE and HF_TOKEN
if AGENT_AVAILABLE:
    llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature":0.2,"max_new_tokens":256}, huggingfacehub_api_token=HF_TOKEN)
    # Tools will be defined at usage time

# ---------------------------
# Role-based access (simple)
# ---------------------------
role = st.sidebar.selectbox("Role", ["Manager","HR","Employee","Admin"])
current_month = datetime.now().strftime("%B %Y")

# ---------------------------
# Manager unified pane
# ---------------------------
if role == "Manager":
    st.header("Manager Command Center — Unified Pane")
    st.markdown("Top: Assign / Reassign — Middle: Visuals (Task summary, Goals, Heatmap) — Right: Agent & GenAI — Bottom: Feedback & Leaves")
    colA, colB = st.columns([1.2, 2.0])

    # ----------------- Left: Task assign/reassign -----------------
    with colA:
        st.subheader("📝 Assign / Reassign Tasks")

        df_tasks = fetch_tasks()
        depts = ["IT","HR","Finance","Marketing","R&D","Ops"]
        employees = sorted(df_tasks["employee"].dropna().unique().tolist()) if not df_tasks.empty else []

        act = st.radio("Action", ["Assign New","AI Suggest & Assign","Reassign"], index=0)
        if act == "Assign New":
            with st.form("assign_new"):
                company = st.text_input("Company")
                dept = st.selectbox("Department", depts)
                emp = st.text_input("Assign To (Employee)")
                title = st.text_input("Task Title")
                desc = st.text_area("Description")
                deadline = st.date_input("Deadline", value=date.today())
                ok = st.form_submit_button("Assign Task")
                if ok and title and emp:
                    tid = str(uuid.uuid4())
                    md = dict(id=tid, company=company, department=dept, employee=emp, task=title, description=desc, deadline=str(deadline), assigned_on=now_iso(), status="Assigned", completion=0, marks=0, priority="Normal")
                    safe_upsert("task", tid, md)
                    st.success(f"Assigned '{title}' to {emp}")

        elif act == "AI Suggest & Assign":
            # simple heuristic AI assignment: pick employee with lowest workload (fewest tasks)
            title = st.text_input("Task Title (AI will find assignee)")
            desc = st.text_area("Description")
            dept = st.selectbox("Department", depts)
            if st.button("Suggest & Assign"):
                df = fetch_tasks()
                # workload heuristic
                if df.empty:
                    candidate = "unassigned"
                else:
                    workload = df.groupby("employee").size().reset_index(name="count")
                    workload = workload.sort_values("count")
                    candidate = workload.iloc[0]["employee"] if len(workload)>0 else "unassigned"
                tid = str(uuid.uuid4())
                md = dict(id=tid, company="", department=dept, employee=candidate, task=title, description=desc, assigned_on=now_iso(), deadline=str(date.today()), status="AutoAssigned", completion=0, marks=0)
                safe_upsert("task", tid, md)
                st.success(f"AI suggested & assigned to {candidate}")

        else:  # Reassign
            df = fetch_tasks()
            if df.empty:
                st.info("No tasks to reassign")
            else:
                task_choice = st.selectbox("Choose Task", df["task"].unique())
                task_row = df[df["task"]==task_choice].iloc[0]
                st.write("Current owner:", task_row.get("employee"))
                new_owner = st.text_input("New Owner")
                if st.button("Confirm Reassign"):
                    task_row["employee"] = new_owner
                    task_row["status"] = "Reassigned"
                    safe_upsert("task", str(uuid.uuid4()), task_row.to_dict())
                    st.success("Reassigned")

        st.markdown("---")
        st.subheader("Gamification")
        # simple points system stored in task metadata
        leaderboard = {}
        for _, r in fetch_tasks().iterrows():
            e = r.get("employee")
            if e:
                leaderboard[e] = leaderboard.get(e, 0) + int(r.get("completion",0)//10)
        st.write("Leaderboard (points):")
        if leaderboard:
            lb = pd.DataFrame(sorted(leaderboard.items(), key=lambda x:-x[1]), columns=["employee","points"])
            st.dataframe(lb)
        else:
            st.write("No leaderboard data yet")

    # ----------------- Middle: Visuals -----------------
    with colB:
        st.subheader("📊 Task Summary & Visuals")
        df = fetch_tasks()
        if df.empty:
            st.info("No tasks available yet")
        else:
            # ensure numeric types
            df["completion"] = pd.to_numeric(df.get("completion",0), errors="coerce").fillna(0)
            df["marks"] = pd.to_numeric(df.get("marks",0), errors="coerce").fillna(0)

            # KPIs
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Avg Completion", f"{df['completion'].mean():.1f}%")
            c2.metric("Avg Marks", f"{df['marks'].mean():.2f}")
            c3.metric("Total Tasks", len(df))
            # attrition demo: simple model on the fly
            # derive features
            df_emp = df.groupby("employee").agg({"completion":"mean","task":"count"}).reset_index().rename(columns={"task":"open_tasks"})
            # mock attrition risk model (rule-based for demo)
            df_emp["attrition_risk_score"] = 0.3*(100-df_emp["completion"])/100 + 0.7*(df_emp["open_tasks"]/max(1,df_emp["open_tasks"].max()))
            c4.metric("Avg Attrition Risk", f"{df_emp['attrition_risk_score'].mean():.2f}")

            # Dept-level bar
            dept_summary = df.groupby("department")[["completion","marks"]].mean().reset_index()
            fig = px.bar(dept_summary, x="department", y="completion", color="department", title="Avg Completion by Dept")
            st.plotly_chart(fig, use_container_width=True)

            # Goal tracker (target 80)
            goals = df.groupby("employee")["completion"].mean().reset_index()
            goals["Target"] = 80
            goal_fig = px.bar(goals, x="employee", y=["completion","Target"], barmode="group", title="Employee vs Target (80%)")
            st.plotly_chart(goal_fig, use_container_width=True)

            # heatmap: employee vs dept marks
            heatmap_data = df.pivot_table(values="marks", index="employee", columns="department", aggfunc="mean").fillna(0)
            st.subheader("Heatmap: Employee vs Department (marks)")
            heat_fig = px.imshow(heatmap_data, text_auto=True)
            st.plotly_chart(heat_fig, use_container_width=True)

            # clustering
            st.subheader("Clustering (performance buckets)")
            if len(df) >= 3:
                dfc = run_kmeans(df, n=3)
                figc = px.scatter(dfc, x="completion", y="marks", color=dfc["cluster"].astype(str), hover_data=["employee","task"])
                st.plotly_chart(figc, use_container_width=True)
            else:
                st.info("Need at least 3 tasks for clustering visualization")

    # ----------------- Right: Agentic AI & GenAI -----------------
    colC = st.columns([1])[0]
    with colC:
        st.subheader("🤖 Agentic AI & Natural Language Insights")
        if AGENT_AVAILABLE:
            st.info("Hugging Face agent available")
            # Define tools for the agent
            def tool_analyze():
                return analyze_performance_impl()
            def tool_review_leaves():
                return review_leaves_impl()
            tools = [
                Tool(name="analyze_performance", func=lambda _: analyze_performance_impl(), description="analyze"),
                Tool(name="review_leaves", func=lambda _: review_leaves_impl(), description="review"),
            ]
            memory = ConversationBufferMemory()
            agent = initialize_agent(tools, llm, agent_type="conversational-react-description", memory=memory, verbose=False)

            q = st.text_input("Chat with Manager Agent", key="agent_query")
            if st.button("Ask Agent"):
                with st.spinner("Agent thinking..."):
                    try:
                        res = agent.run(q)
                        st.success(res)
                    except Exception as e:
                        st.error(f"Agent error: {e}")
        else:
            st.info("Agent not configured (HF token missing). You can still use GenAI Insights below.")

        # GenAI Insights using HF inference client (simple)
        st.markdown("### AI-generated summary")
        if st.button("Generate Summary"):
            df = fetch_tasks()
            if df.empty:
                st.info("No tasks to summarize")
            else:
                # Create a prompt (short) and call simple HF inference via huggingface_hub if available
                try:
                    from huggingface_hub import InferenceClient
                    client = InferenceClient(token=HF_TOKEN) if HF_TOKEN else None
                    avg = df["completion"].mean()
                    lowc = df[df["completion"]<50].shape[0]
                    prompt = f"Summarize team performance: average completion {avg:.1f}%, low performers {lowc}. Provide 3 suggestions."
                    if client:
                        r = client.text_generation("tiiuae/falcon-7b-instruct", inputs=prompt, max_new_tokens=120)
                        out = r[0]["generated_text"] if isinstance(r, list) and "generated_text" in r[0] else str(r)
                        st.info(out)
                    else:
                        st.warning("Hugging Face token not set; showing a generated template")
                        st.info(f"Average completion {avg:.1f}%. Low performers: {lowc}. Suggestions: 1) Mentorship 2) Re-balance tasks 3) Training")
                except Exception as e:
                    st.error(f"GenAI invocation error: {e}")

    # ----------------- Bottom: Feedback & Leaves -----------------
    st.markdown("---")
    st.subheader("Feedback & Leaves Queue")
    fb_col, leave_col = st.columns(2)

    with fb_col:
        st.markdown("### 360° Feedback")
        with st.form("feedback_form"):
            emp = st.text_input("Employee (feedback about)")
            source = st.selectbox("Source", ["Self","Peer","Manager","Client"])
            anon = st.checkbox("Submit Anonymously", value=False)
            text = st.text_area("Feedback text")
            if st.form_submit_button("Submit Feedback"):
                fid = str(uuid.uuid4())
                md = dict(id=fid, employee=emp, source=source if not anon else "Anonymous", text=text, timestamp=now_iso())
                # sentiment quick
                try:
                    v = vectorizer.transform([text])
                    sent = int(svm_tf.predict(v)[0])
                    md["sentiment"] = "Positive" if sent==1 else "Negative"
                except Exception:
                    md["sentiment"] = "Neutral"
                safe_upsert("feedback", fid, md)
                st.success("Feedback submitted")

        st.markdown("Recent feedback")
        ff = fetch_feedback()
        if not ff.empty:
            st.dataframe(ff.head(10))

    with leave_col:
        st.markdown("### Leave Requests")
        with st.form("leave_form"):
            name = st.text_input("Employee applying")
            ltype = st.selectbox("Leave Type", ["Casual","Sick","Paid"])
            fd = st.date_input("From", value=date.today())
            td = st.date_input("To", value=date.today())
            reason = st.text_area("Reason")
            if st.form_submit_button("Request Leave"):
                lid = str(uuid.uuid4())
                md = dict(id=lid, employee=name, leave_type=ltype, from_date=str(fd), to_date=str(td), reason=reason, status="Pending", requested_on=now_iso())
                safe_upsert("leave", lid, md)
                st.success("Leave requested")

        st.markdown("Pending leaves")
        lv = fetch_leaves()
        if not lv.empty:
            st.dataframe(lv[lv["status"]=="Pending"])

# ---------------------------
# HR view
# ---------------------------
elif role == "HR":
    st.header("HR Dashboard (Leaves & Attrition)")
    leaves = fetch_leaves()
    st.subheader("Pending Leaves")
    if not leaves.empty:
        st.dataframe(leaves)
    else:
        st.info("No leaves")

    st.subheader("Attrition Prediction (demo)")
    df = fetch_tasks()
    if df.empty:
        st.info("No data")
    else:
        # simple feature engineering for demo
        emp = df.groupby("employee").agg({"completion":"mean","task":"count"}).rename(columns={"task":"open_tasks"}).reset_index()
        emp["risk"] = 0.3*(100-emp["completion"])/100 + 0.7*(emp["open_tasks"]/emp["open_tasks"].max())
        st.dataframe(emp.sort_values("risk", ascending=False))

# ---------------------------
# Employee view
# ---------------------------
elif role == "Employee":
    st.header("Employee Portal")
    name = st.text_input("Your name")
    if st.button("Load My Tasks"):
        df = fetch_tasks()
        my = df[df["employee"]==name]
        if my.empty:
            st.info("No tasks assigned")
        else:
            for _, r in my.iterrows():
                st.write("Task:", r["task"])
                newc = st.slider(f"Completion for {r['task']}", 0,100, int(r.get("completion",0)))
                if st.button(f"Update {r['id']}", key=r["id"]):
                    r["completion"]=newc
                    r["status"] = "Completed" if newc==100 else "In Progress"
                    safe_upsert("task", str(uuid.uuid4()), r)
                    st.success("Updated")

# ---------------------------
# Admin view
# ---------------------------
elif role == "Admin":
    st.header("Admin Console")
    st.subheader("All tasks")
    st.dataframe(fetch_tasks())
    st.subheader("All feedback")
    st.dataframe(fetch_feedback())
    st.subheader("All leaves")
    st.dataframe(fetch_leaves())

# ---------------------------
# End
# ---------------------------
