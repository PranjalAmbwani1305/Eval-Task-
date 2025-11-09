# main.py
# AI-Powered Task Management System — Enterprise-ready single-file Streamlit app
# Features: Safe Pinecone init, Manager/Team/Client/Admin roles, Client Portal,
# AI Insights (optional via Hugging Face), metadata sanitization, retries.

import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import pandas as pd
import uuid
import json
import logging
import time
from datetime import datetime, date
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import plotly.express as px

# Optional: Hugging Face (generative) features
try:
    from huggingface_hub import InferenceClient
    HF_HUB_AVAILABLE = True
except Exception:
    HF_HUB_AVAILABLE = False

# Optional: LangChain agent usage (not required)
try:
    from langchain import HuggingFaceHub
    from langchain.agents import initialize_agent, Tool
    from langchain.memory import ConversationBufferMemory
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

# -----------------------------
# CONFIG & INIT
# -----------------------------
st.set_page_config(page_title="AI Task System (Enterprise)", layout="wide")
st.title("AI-Powered Task Management System — Enterprise Edition")

# Secrets
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
HF_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")

# Pinecone constants — KEEP DIMENSION consistent everywhere
INDEX_NAME = "task"
DIMENSION = 1024  # MUST match index dimension used in creation and vector generation

# -----------------------------
# LOGGER
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_task_system")

# -----------------------------
# PINECONE SAFE INITIALIZATION
# -----------------------------
index = None
if PINECONE_API_KEY:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)

        # create index if missing
        existing = [i["name"] for i in pc.list_indexes()]
        if INDEX_NAME not in existing:
            st.info(f"Creating Pinecone index '{INDEX_NAME}' (dim={DIMENSION}). This may take ~30-60s.")
            pc.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            # wait until ready
            while True:
                try:
                    desc = pc.describe_index(INDEX_NAME)
                    # some SDKs expose different structures; try both
                    status = None
                    if isinstance(desc, dict):
                        status = desc.get("status", {}).get("ready", None)
                    else:
                        status = getattr(desc, "status", {}).get("ready", None) if desc else None
                    if status:
                        break
                except Exception:
                    pass
                st.write("⏳ Waiting for Pinecone index to initialize...")
                time.sleep(3)

        index = pc.Index(INDEX_NAME)
        st.success("✅ Connected to Pinecone.")
    except Exception as e:
        index = None
        st.warning(f"Pinecone initialization failed: {str(e)}. Running in LOCAL mode (non-persistent).")
else:
    st.warning("Pinecone API key not provided — running in LOCAL mode (non-persistent).")

# -----------------------------
# Utilities
# -----------------------------
def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def rand_vec():
    return np.random.rand(DIMENSION).tolist()

def safe_meta(md: dict):
    """Sanitize metadata before upsert to Pinecone."""
    clean, invalid = {}, {}
    for k, v in md.items():
        try:
            if v is None:
                v = ""
            elif isinstance(v, (datetime, date)):
                v = v.isoformat()
            elif isinstance(v, (np.generic, np.number)):
                v = float(v)
            elif isinstance(v, (list, dict)):
                try:
                    v = json.dumps(v)
                except Exception:
                    invalid[k] = str(v)
                    v = str(v)
            elif not isinstance(v, (str, int, float, bool)):
                invalid[k] = str(type(v))
                v = str(v)
        except Exception as ex:
            invalid[k] = f"ERROR({type(ex).__name__}):{str(ex)}"
            v = str(v)
        clean[k] = v

    if invalid:
        logger.warning(f"Metadata sanitized for keys: {list(invalid.keys())}")
        # show the warnings in a non-intrusive way
        with st.expander("⚠️ Metadata sanitization (automatically fixed)", expanded=False):
            for key, val in invalid.items():
                st.warning(f"{key}: {val}")

    return clean

def safe_upsert(id_, vec, md, retries=2, delay=1.5):
    """Robust upsert with retries. md must already be sanitized."""
    if index:
        for attempt in range(1, retries + 1):
            try:
                index.upsert([{"id": id_, "values": vec, "metadata": md}])
                return True
            except Exception as e:
                logger.warning(f"Pinecone upsert attempt {attempt} failed: {e}")
                time.sleep(delay)
        st.error("Pinecone upsert failed after retries.")
        return False
    else:
        # local fallback (non-persistent)
        if "LOCAL_DATA" not in st.session_state:
            st.session_state.LOCAL_DATA = {}
        st.session_state.LOCAL_DATA[id_] = md
        return True

def fetch_all():
    """Return DataFrame of all metadata — safe for empty index and local fallback."""
    if index:
        try:
            stats = index.describe_index_stats()
            total = stats.get("total_vector_count", 0) if isinstance(stats, dict) else stats.get("total_vector_count", 0)
            if total == 0:
                return pd.DataFrame()
            res = index.query(vector=rand_vec(), top_k=1000, include_metadata=True)
            if not getattr(res, "matches", None):
                return pd.DataFrame()
            rows = []
            for m in res.matches:
                md = m.metadata or {}
                md["_id"] = m.id
                rows.append(md)
            return pd.DataFrame(rows)
        except Exception as e:
            st.warning(f"Pinecone query error: {e}")
            return pd.DataFrame()
    else:
        # local fallback
        data = st.session_state.get("LOCAL_DATA", {})
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(list(data.values()))

# -----------------------------
# SIMPLE AI MODELS (existing)
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
X_rf = np.array([[10, 2], [50, 1], [90, 0], [100, 0]])
y_rf = [0, 1, 0, 0]
rf.fit(X_rf, y_rf)

# -----------------------------
# INITIALIZE LOCAL STORE (if needed)
# -----------------------------
if index is None:
    if "LOCAL_DATA" not in st.session_state:
        st.session_state.LOCAL_DATA = {}
        # optionally, you can seed some demo entries for local testing

# -----------------------------
# RBAC / ROLE SELECTION
# -----------------------------
role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client", "Admin"])
current_month = datetime.now().strftime("%B %Y")

# ====================================================
# MANAGER (keep your existing UI flow intact)
# ====================================================
if role == "Manager":
    st.header("Manager (Boss) Dashboard")
    tab1, tab2 = st.tabs(["Assign Task", "Boss Review & Adjustment"])

    # Assign Task tab (unchanged logic, using safe_upsert)
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
                md_raw = {
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
                }
                md = safe_meta(md_raw)
                ok = safe_upsert(tid, rand_vec(), md)
                if ok:
                    st.success(f"Task '{task}' assigned to {employee}")

    # Boss Review tab
    with tab2:
        if st.button("🔄 Refresh Data"):
            st.session_state.pop("manager_df", None)

        if "manager_df" not in st.session_state:
            st.session_state["manager_df"] = fetch_all()

        df = st.session_state["manager_df"]

        if df.empty:
            st.warning("No tasks found.")
        else:
            # normalize column names
            df.columns = [c.lower() for c in df.columns]
            if "completion" in df.columns:
                df["completion"] = pd.to_numeric(df["completion"], errors="coerce").fillna(0)
            else:
                df["completion"] = 0

            tasks_ready = df[df["completion"] > 0]

            if tasks_ready.empty:
                st.info("No team-submitted tasks yet.")
            else:
                for _, r in tasks_ready.iterrows():
                    st.markdown(f"### {r['task']}")
                    st.write(f"**Employee:** {r.get('employee', 'Unknown')}")
                    st.write(f"**Reported Completion:** {r.get('completion', 0)}%")
                    st.write(f"**Current Marks:** {r.get('marks', 0):.2f}")
                    st.write(f"**Deadline Risk:** {r.get('deadline_risk', 'N/A')}")

                    # Boss slides to adjust completion
                    adjusted_completion = st.slider(
                        f"Adjust Completion % for {r['task']}",
                        0, 100, int(r.get("completion", 0)), key=f"adj_{r['_id']}"
                    )
                    adjusted_marks = float(lin_reg.predict([[adjusted_completion]])[0])

                    comments = st.text_area(f"Boss Comments for {r['task']}", key=f"boss_cmt_{r['_id']}")
                    approve = st.radio(f"Approve Task {r['task']}?", ["Yes", "No"], key=f"boss_app_{r['_id']}")

                    if st.button(f"✅ Finalize Review for {r['task']}", key=f"final_{r['_id']}"):
                        sentiment_val = int(svm_clf.predict(vectorizer.transform([comments]))[0])
                        sentiment = "Positive" if sentiment_val == 1 else "Negative"
                        md_raw = {
                            **r,
                            "completion": adjusted_completion,
                            "marks": adjusted_marks,
                            "manager_comments": comments,
                            "reviewed": True,
                            "sentiment": sentiment,
                            "approved_by_boss": approve == "Yes",
                            "reviewed_on": now()
                        }
                        md = safe_meta(md_raw)
                        safe_upsert(r["_id"], rand_vec(), md)
                        st.success(f"Review finalized for {r['task']} ({sentiment}).")
                        # refresh manager data
                        st.session_state.pop("manager_df", None)
                        st.experimental_rerun()

# ====================================================
# TEAM MEMBER
# ====================================================
elif role == "Team Member":
    st.header("Team Member Progress Update")
    company = st.text_input("Company")
    employee = st.text_input("Your Name")

    if st.button("🔄 Load My Tasks"):
        try:
            if index:
                res = index.query(vector=rand_vec(), top_k=500, include_metadata=True,
                                  filter={"company": {"$eq": company}, "employee": {"$eq": employee}})
                st.session_state["tasks"] = [(m.id, m.metadata) for m in res.matches or []]
            else:
                # local mode
                data = fetch_all()
                st.session_state["tasks"] = [(r.get("_id", str(i)), r) for i, r in data.iterrows() if r.get("company") == company and r.get("employee") == employee]
            st.success(f"Loaded {len(st.session_state.get('tasks', []))} tasks.")
        except Exception as e:
            st.error(f"Load error: {e}")

    for tid, md in st.session_state.get("tasks", []):
        st.subheader(md.get("task"))
        st.write(md.get("description"))
        curr = float(md.get("completion", 0))
        new = st.slider(f"Completion for {md.get('task')}", 0, 100, int(curr))
        if st.button(f"Submit {md.get('task')}", key=tid):
            marks = float(lin_reg.predict([[new]])[0])
            track = "On Track" if log_reg.predict([[new]])[0] == 1 else "Delayed"
            miss = rf.predict([[new, 0]])[0]
            md_raw = {
                **md,
                "completion": new,
                "marks": marks,
                "status": track,
                "deadline_risk": "High" if miss else "Low",
                "submitted_on": now(),
                "client_reviewed": False
            }
            md2 = safe_meta(md_raw)
            safe_upsert(tid, rand_vec(), md2)
            st.success(f"Updated {md.get('task')} ({track}, Risk={md2['deadline_risk']})")

# ====================================================
# CLIENT (Enterprise-grade Client Portal)
# ====================================================
elif role == "Client":
    st.header("📊 Client Intelligence Portal")
    st.markdown("View project summary, review deliverables, and provide feedback.")

    company = st.text_input("Company Name (enter your company to filter view)")

    if company:
        df_all = fetch_all()
        if df_all.empty:
            st.info("No project data available yet.")
        else:
            # filter by company (case-insensitive)
            df_client = df_all[df_all.get("company", "").str.lower() == company.lower()] if "company" in df_all.columns else pd.DataFrame()
            if df_client.empty:
                st.warning("No tasks found for this company.")
            else:
                # KPIs
                df_client["completion"] = pd.to_numeric(df_client.get("completion", 0), errors="coerce").fillna(0)
                df_client["marks"] = pd.to_numeric(df_client.get("marks", 0), errors="coerce").fillna(0)
                avg_completion = df_client["completion"].mean()
                avg_marks = df_client["marks"].mean()
                c1, c2, c3 = st.columns(3)
                c1.metric("Project Completion", f"{avg_completion:.2f}%")
                c2.metric("Average Marks", f"{avg_marks:.2f}")
                c3.metric("Active Tasks", len(df_client))

                # Departmental breakdown
                if "department" in df_client.columns:
                    dept_chart = df_client.groupby("department")["completion"].mean().reset_index()
                    st.plotly_chart(px.bar(dept_chart, x="department", y="completion", color="department", title="Department Progress"), use_container_width=True)

                # Milestones / Overdue
                st.subheader("Deliverables")
                status_table = df_client[["task", "employee", "deadline", "completion", "status"]].sort_values("deadline")
                st.dataframe(status_table, use_container_width=True)

                # Deliverable review + feedback submission for completed tasks
                st.subheader("Provide Feedback on Completed Deliverables")
                completed = df_client[df_client["completion"] >= 80]  # consider >=80 as deliverable-ready
                if completed.empty:
                    st.info("No completed deliverables available for feedback.")
                else:
                    for _, row in completed.iterrows():
                        with st.expander(f"{row['task']} — {row.get('employee','')}", expanded=False):
                            st.write(f"Deadline: {row.get('deadline')}, Completion: {row.get('completion')}%")
                            feedback_text = st.text_area(f"Feedback for {row['task']}", key=f"fb_{row['_id']}")
                            rating = st.slider(f"Rating (1-5) for {row['task']}", 1, 5, 4, key=f"rate_{row['_id']}")
                            if st.button(f"Submit Feedback for {row['task']}", key=f"submit_fb_{row['_id']}"):
                                # simple sentiment heuristic + safe upsert into metadata
                                sent = "Positive" if any(w in (feedback_text or "").lower() for w in ["good", "great", "excellent", "nice"]) else "Neutral"
                                row_md = {**row, "client_feedback": feedback_text, "client_rating": int(rating), "client_sentiment": sent, "client_feedback_on": now()}
                                md = safe_meta(row_md)
                                safe_upsert(row["_id"], rand_vec(), md)
                                st.success(f"Feedback recorded (sentiment: {sent}).")

                # AI-generated summary for client (requires HF token)
                st.subheader("🤖 AI Project Summary")
                if HF_HUB_AVAILABLE and HF_TOKEN:
                    try:
                        client = InferenceClient(token=HF_TOKEN)
                        # keep context concise: show key fields only
                        sample_ctx = df_client[["task", "employee", "completion", "deadline", "status"]].head(30).to_dict(orient="records")
                        prompt = f"Produce a concise, business-style 5-line summary for the following project tasks:\n{sample_ctx}"
                        with st.spinner("Generating AI summary..."):
                            resp = client.text_generation("mistralai/Mixtral-8x7B-Instruct", inputs=prompt, max_new_tokens=200)
                            if isinstance(resp, list) and "generated_text" in resp[0]:
                                summary = resp[0]["generated_text"]
                            else:
                                # Some HF responses may be raw dicts
                                summary = str(resp)
                            st.info(summary)
                    except Exception as e:
                        st.warning(f"AI summary failed: {e}")
                else:
                    st.info("AI summaries available when HUGGINGFACEHUB_API_TOKEN is set in secrets.")

# ====================================================
# ADMIN
# ====================================================
elif role == "Admin":
    st.header("Admin Dashboard")

    if st.button("🔄 Refresh Data"):
        st.session_state.pop("admin_df", None)

    if "admin_df" not in st.session_state:
        st.session_state["admin_df"] = fetch_all()

    df = st.session_state["admin_df"]

    if df.empty:
        st.warning("No data found.")
    else:
        if "completion" in df.columns:
            df["completion"] = pd.to_numeric(df["completion"], errors="coerce").fillna(0)
        else:
            df["completion"] = 0

        if "marks" in df.columns:
            df["marks"] = pd.to_numeric(df["marks"], errors="coerce").fillna(0)
        else:
            df["marks"] = 0

        st.subheader("Top Employees by Marks")
        top = (
            df.groupby("employee")["marks"]
            .mean()
            .reset_index()
            .sort_values("marks", ascending=False)
            .head(5)
        )
        st.dataframe(top)

        st.subheader("K-Means Clustering (Performance)")
        if len(df) >= 2:
            n_clusters = min(3, len(df))
            km = KMeans(n_clusters=n_clusters, n_init=10).fit(
                df[["completion", "marks"]].fillna(0)
            )
            df["cluster"] = km.labels_
            st.plotly_chart(
                px.scatter(
                    df,
                    x="completion",
                    y="marks",
                    color=df["cluster"].astype(str),
                    hover_data=["employee", "task"],
                    title="Employee Task Clusters",
                )
            )
        else:
            st.info("Not enough data for clustering.")

        avg_m = df["marks"].mean()
        avg_c = df["completion"].mean()
        st.info(
            f"Average marks: {avg_m:.2f}, completion: {avg_c:.1f}%. "
            f"Top performers: {', '.join(top['employee'].tolist())}."
        )

        st.download_button(
            "Download All Tasks (CSV)",
            df.to_csv(index=False),
            "tasks.csv",
            "text/csv",
        )

# -----------------------------
# ADDITIONAL AI INSIGHTS PANEL (non-destructive append)
# -----------------------------
st.markdown("---")
st.header("🤖 AI Insights & Analytics Center (Read-only)")

df_all = fetch_all()
if df_all.empty:
    st.info("No task data available for analytics.")
else:
    # ensure numeric
    for col in ["completion", "marks"]:
        if col in df_all.columns:
            df_all[col] = pd.to_numeric(df_all[col], errors="coerce").fillna(0)
        else:
            df_all[col] = 0

    # Smart metrics
    st.subheader("📊 Smart Analytics Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Completion", f"{df_all['completion'].mean():.2f}%")
    col2.metric("Average Marks", f"{df_all['marks'].mean():.2f}")
    col3.metric("Unique Employees", df_all['employee'].nunique())

    # Department chart
    if "department" in df_all.columns:
        dept_summary = df_all.groupby("department")[["completion", "marks"]].mean().reset_index()
        st.plotly_chart(px.bar(dept_summary, x="department", y="completion", color="department", title="Department Completion Rates"), use_container_width=True)

    # Goal tracker
    st.subheader("🎯 Goal Tracker (Target = 80%)")
    goals = df_all.groupby("employee")["completion"].mean().reset_index()
    goals["Target"] = 80
    st.plotly_chart(px.bar(goals, x="employee", y=["completion", "Target"], barmode="group", title="Employee vs Target"), use_container_width=True)

    # Heatmap
    if "department" in df_all.columns:
        heatmap_data = df_all.pivot_table(values="marks", index="employee", columns="department", aggfunc="mean").fillna(0)
        st.plotly_chart(px.imshow(heatmap_data, text_auto=True, color_continuous_scale="Blues", title="Employee vs Department Heatmap (Marks)"), use_container_width=True)

    # Gamification badges
    st.subheader("🏅 Performance Badges")
    def badge(score):
        if score >= 85:
            return "🥇 Gold"
        elif score >= 70:
            return "🥈 Silver"
        elif score >= 50:
            return "🥉 Bronze"
        return "⚪ Needs Improvement"
    df_all["Badge"] = df_all["completion"].apply(badge)
    st.dataframe(df_all[["employee", "completion", "marks", "Badge"]].drop_duplicates(subset=["employee"]), use_container_width=True)

    # Attrition heuristic
    st.subheader("📉 Attrition Risk (Heuristic)")
    df_all["attrition_risk"] = df_all.apply(lambda r: "High" if (r["completion"] < 40 and r["marks"] < 2.0) else "Low", axis=1)
    st.plotly_chart(px.scatter(df_all, x="completion", y="marks", color="attrition_risk", hover_data=["employee", "task"], title="Attrition Risk Scatter"), use_container_width=True)
    st.info("Explainability: Employees marked 'High' have completion <40% AND marks <2.0. This is a heuristic — replace with ML model for production.")

    # Chat-based quick query (Hugging Face)
    st.subheader("💬 Ask AI (Quick Insights)")
    user_q = st.text_input("Ask (e.g. 'Who is underperforming this month?')", key="ai_query")
    if st.button("Run AI Query"):
        if HF_HUB_AVAILABLE and HF_TOKEN:
            try:
                hf = InferenceClient(token=HF_TOKEN)
                avg = df_all["completion"].mean()
                low_count = int((df_all["completion"] < avg - 10).sum())
                context = f"Average completion: {avg:.1f}%. Employees {low_count} below average."
                prompt = f"Context: {context}\nQuestion: {user_q}\nAnswer concisely with recommended actions."
                with st.spinner("Generating..."):
                    resp = hf.text_generation("mistralai/Mixtral-8x7B-Instruct", inputs=prompt, max_new_tokens=200)
                    if isinstance(resp, list) and "generated_text" in resp[0]:
                        out = resp[0]["generated_text"]
                    else:
                        out = str(resp)
                    st.success(out)
            except Exception as e:
                st.error(f"AI query failed: {e}")
        else:
            st.warning("Hugging Face token not configured. Set HUGGINGFACEHUB_API_TOKEN in secrets to enable.")

# End of file
