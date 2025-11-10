# main.py — AI Workforce Intelligence Platform (Enterprise)
# Requirements:
# pip install streamlit pinecone-client scikit-learn plotly huggingface-hub pandas

import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import pandas as pd
import uuid
import json
import time
from datetime import datetime, date
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from huggingface_hub import InferenceClient
import plotly.express as px

# ----------------------
# App config
# ----------------------
st.set_page_config(page_title="AI Workforce Intelligence Platform", layout="wide")
st.title("AI Workforce Intelligence Platform")

# ----------------------
# Secrets / constants
# ----------------------
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
HF_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")
INDEX_NAME = "task"   # lowercase only
DIMENSION = 1024

# ----------------------
# Initialize Pinecone (best-effort)
# ----------------------
pc = None
index = None
if PINECONE_API_KEY:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        existing = [i["name"] for i in pc.list_indexes()]
        if INDEX_NAME not in existing:
            pc.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            # wait until ready (best-effort)
            with st.spinner("Creating Pinecone index..."):
                for _ in range(20):
                    try:
                        desc = pc.describe_index(INDEX_NAME)
                        if desc.get("status", {}).get("ready"):
                            break
                    except Exception:
                        pass
                    time.sleep(1)
        index = pc.Index(INDEX_NAME)
        st.caption(f"Connected to Pinecone index: {INDEX_NAME}")
    except Exception as e:
        st.warning(f"Pinecone init failed — running local-only. ({e})")
else:
    st.warning("Pinecone API key missing — running local-only.")

# ----------------------
# Helpers
# ----------------------
def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def rand_vec():
    return np.random.rand(DIMENSION).tolist()

def safe_meta(md: dict) -> dict:
    """Convert metadata values into Pinecone-friendly JSON primitives."""
    clean = {}
    for k, v in (md or {}).items():
        if isinstance(v, (datetime, date)):
            clean[k] = v.isoformat()
        elif isinstance(v, (dict, list)):
            try:
                clean[k] = json.dumps(v)
            except Exception:
                clean[k] = str(v)
        elif pd.isna(v):
            clean[k] = ""
        else:
            # ensure basic types only
            if isinstance(v, (str, int, float, bool)) or v is None:
                clean[k] = v
            else:
                clean[k] = str(v)
    return clean

def upsert_data(id_, md: dict):
    """Upsert metadata into Pinecone or into local session-state storage if Pinecone is not available."""
    if not index:
        loc = st.session_state.setdefault("LOCAL_DATA", {})
        loc[id_] = md
        return True
    try:
        index.upsert([{"id": id_, "values": rand_vec(), "metadata": safe_meta(md)}])
        return True
    except Exception as e:
        st.warning(f"Pinecone upsert failed: {e}")
        return False

def fetch_all() -> pd.DataFrame:
    """Fetch all records from Pinecone (or local storage). Returns DataFrame with metadata + _id column."""
    if not index:
        local = st.session_state.get("LOCAL_DATA", {})
        if not local:
            return pd.DataFrame()
        rows = []
        for k, md in local.items():
            rec = dict(md)
            rec["_id"] = k
            rows.append(rec)
        return pd.DataFrame(rows)
    try:
        # check stats first to avoid 0 -> query errors
        try:
            stats = index.describe_index_stats()
            if stats and stats.get("total_vector_count", 0) == 0:
                return pd.DataFrame()
        except Exception:
            # continue to query anyway if describe fails
            pass
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
        st.warning(f"Fetch error: {e}")
        return pd.DataFrame()

# ----------------------
# Small ML models used for marks/heuristics
# ----------------------
lin_reg = LinearRegression().fit([[0], [50], [100]], [0, 2.5, 5])
# we only use this simple model for marks estimation; replace with production model if desired.

# ----------------------
# HF client helper (safe call)
# ----------------------
def hf_text_generation(prompt: str, model: str = "mistralai/Mixtral-8x7B-Instruct", max_new_tokens: int = 150):
    """Call HF InferenceClient.text_generation safely trying both 'inputs' and 'prompt' keywords."""
    if not HF_TOKEN:
        raise RuntimeError("Hugging Face token not configured.")
    client = InferenceClient(token=HF_TOKEN)
    try:
        # try inputs kw first
        res = client.text_generation(model=model, inputs=prompt, max_new_tokens=max_new_tokens)
    except TypeError:
        # fallback to prompt kw
        res = client.text_generation(model=model, prompt=prompt, max_new_tokens=max_new_tokens)
    # res may be dict or list depending on client; extract text safely
    if isinstance(res, dict):
        # common shape: {'generated_text': '...'}
        return res.get("generated_text") or res.get("output") or json.dumps(res)
    if isinstance(res, list) and len(res) > 0:
        # sometimes returned as list of dicts
        if isinstance(res[0], dict):
            return res[0].get("generated_text") or json.dumps(res[0])
    return str(res)

# ----------------------
# Role selection
# ----------------------
role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client", "Admin"])
current_month = datetime.now().strftime("%B %Y")

# ----------------------
# MANAGER
# ----------------------
if role == "Manager":
    st.header("Manager Command Center")

    df_all = fetch_all()
    tab_assign, tab_overview, tab_meet, tab_leave, tab_360 = st.tabs([
        "Task Management", "Team Overview", "Meeting Notes & AI", "Leave Approvals", "360 Overview"
    ])

    # Task Management
    with tab_assign:
        st.subheader("Assign Task")
        with st.form("assign_form"):
            company = st.text_input("Company")
            department = st.text_input("Department")
            employee = st.text_input("Employee")
            task_title = st.text_input("Task Title")
            description = st.text_area("Description")
            deadline = st.date_input("Deadline", value=date.today())
            submitted = st.form_submit_button("Assign Task")
            if submitted:
                if not employee or not task_title:
                    st.warning("Employee and Task Title required.")
                else:
                    tid = str(uuid.uuid4())
                    md = {
                        "company": company or "",
                        "department": department or "General",
                        "employee": employee,
                        "task": task_title,
                        "description": description or "",
                        "deadline": deadline.isoformat(),
                        "completion": 0,
                        "marks": 0,
                        "status": "Assigned",
                        "type": "Task",
                        "created": now()
                    }
                    ok = upsert_data(tid, md)
                    if ok:
                        st.success(f"Assigned '{task_title}' to {employee}.")

        # Reassign quick section (list tasks)
        df = df_all.copy() if not df_all.empty else pd.DataFrame()
        if not df.empty:
            df["task"] = df.get("task", "")
            tasks_list = df[df["task"] != ""]["task"].unique().tolist()
            if tasks_list:
                st.markdown("---")
                st.subheader("Reassign Task")
                sel = st.selectbox("Select task to reassign", tasks_list)
                new_emp = st.text_input("New employee name for reassignment")
                reason = st.text_area("Reason (optional)")
                if st.button("Reassign Task"):
                    rec = df[df["task"] == sel].iloc[0].to_dict()
                    rec["employee"] = new_emp or rec.get("employee", "")
                    rec["status"] = "Reassigned"
                    rec["reassigned_reason"] = reason
                    rec["reassigned_on"] = now()
                    upsert_data(rec.get("_id") or str(uuid.uuid4()), rec)
                    st.success("Task reassigned.")

    # Team Overview
    with tab_overview:
        st.subheader("Team Performance Overview")
        df = df_all.copy() if not df_all.empty else pd.DataFrame()
        if df.empty:
            st.info("No task data available.")
        else:
            # ensure columns
            for c in ["employee", "department", "task", "completion", "status", "created"]:
                if c not in df.columns:
                    df[c] = ""
            df["completion"] = pd.to_numeric(df["completion"], errors="coerce").fillna(0)
            st.dataframe(df[["employee", "department", "task", "completion", "status", "created"]], use_container_width=True)

            # bar: completion by employee (no color if department missing)
            has_dept = "department" in df.columns and len(df["department"].dropna().unique()) > 0
            if has_dept:
                fig = px.bar(df, x="employee", y="completion", color="department", title="Completion by Employee (colored by department)")
            else:
                fig = px.bar(df, x="employee", y="completion", title="Completion by Employee")
            st.plotly_chart(fig, use_container_width=True)

            # Optional time trend if 'created' exists and parseable
            if "created" in df.columns:
                try:
                    df["created_dt"] = pd.to_datetime(df["created"], errors="coerce")
                    trend = df.dropna(subset=["created_dt"]).groupby(pd.Grouper(key="created_dt", freq="D"))["completion"].mean().reset_index()
                    if not trend.empty:
                        fig2 = px.line(trend, x="created_dt", y="completion", title="Average Completion over time")
                        st.plotly_chart(fig2, use_container_width=True)
                except Exception:
                    pass

    # Meeting notes & AI summarization
    with tab_meet:
        st.subheader("Meeting Notes & AI Summaries")
        uploaded = st.file_uploader("Upload meeting notes (.txt)", type=["txt"])
        if uploaded:
            content = uploaded.read().decode("utf-8", errors="ignore")
            st.text_area("Transcript", content, height=250)
            if st.button("Generate Summary"):
                if not HF_TOKEN:
                    st.warning("Hugging Face token not configured.")
                else:
                    prompt = f"Summarize the meeting notes focusing on decisions, owners, and next actions:\n\n{content[:4000]}"
                    with st.spinner("Generating summary..."):
                        try:
                            summary = hf_text_generation(prompt=prompt, model="mistralai/Mixtral-8x7B-Instruct", max_new_tokens=250)
                            st.subheader("AI Summary")
                            st.write(summary)
                        except Exception as e:
                            st.error(f"AI summarization failed: {e}")

    # Leave approvals
    with tab_leave:
        st.subheader("Leave Approvals")
        df = df_all.copy() if not df_all.empty else pd.DataFrame()
        # leave rows identified by presence of 'type' field and typical schema
        leaves = pd.DataFrame()
        if not df.empty:
            if "type" in df.columns:
                leaves = df[df["type"].isin(["Casual", "Sick", "Earned"])]
            else:
                # fallback: rows where status is Pending and don't have task
                leaves = df[df.get("task", "").isna() | (df.get("task", "") == "")]
        if leaves.empty:
            st.info("No leave requests pending.")
        else:
            for _, row in leaves.iterrows():
                if row.get("status", "") == "Pending":
                    st.markdown(f"**{row.get('employee','')}** requested **{row.get('type','Leave')}** ({row.get('from','')} → {row.get('to','')})")
                    st.write(f"Reason: {row.get('reason','-')}")
                    decision = st.radio(f"Decision for {row.get('employee','')}", ["Approve", "Reject"], key=row.get("_id"))
                    if st.button(f"Finalize {row.get('_id')}", key=f"lv_{row.get('_id')}"):
                        row["status"] = "Approved" if decision == "Approve" else "Rejected"
                        row["approved_on"] = now()
                        upsert_data(row.get("_id") or str(uuid.uuid4()), row)
                        st.success(f"Leave {row['status']} for {row.get('employee','')}.")

    # 360 overview
    with tab_360:
        st.subheader("Employee 360 Overview")
        df = df_all.copy() if not df_all.empty else pd.DataFrame()
        if df.empty:
            st.info("No data.")
        else:
            if "employee" not in df.columns:
                st.info("No employee values present.")
            else:
                employees = df["employee"].dropna().unique().tolist()
                sel = st.selectbox("Select employee", employees)
                emp_df = df[df["employee"] == sel]
                # ensure numeric conversion
                emp_df["completion"] = pd.to_numeric(emp_df.get("completion", 0), errors="coerce").fillna(0)
                st.dataframe(emp_df[["task", "completion", "marks", "status", "client_feedback"]].fillna(""), use_container_width=True)
                st.metric("Average completion", f"{emp_df['completion'].mean():.1f}%")

# ----------------------
# TEAM MEMBER
# ----------------------
elif role == "Team Member":
    st.header("Team Member Dashboard")

    name = st.text_input("Enter your name")
    if name:
        df_all = fetch_all()
        if df_all.empty:
            st.info("No records found.")
        else:
            # normalize type field
            if "type" not in df_all.columns:
                df_all["type"] = df_all.apply(lambda r: "Task" if r.get("task") else "Leave", axis=1)
            my_tasks = df_all[(df_all.get("employee","").str.lower() == name.lower()) & (df_all["type"] == "Task")]
            my_leaves = df_all[(df_all.get("employee","").str.lower() == name.lower()) & (df_all["type"] != "Task")]

            t1, t2 = st.tabs(["My Tasks", "My Leaves"])

            with t1:
                st.subheader("My Tasks")
                if my_tasks.empty:
                    st.info("No tasks assigned yet.")
                else:
                    my_tasks = my_tasks.reset_index(drop=True)
                    for _, r in my_tasks.iterrows():
                        task_title = r.get("task", "Untitled")
                        status = r.get("status", "Assigned")
                        # ensure numeric completion
                        try:
                            curr = int(float(r.get("completion", 0) or 0))
                        except Exception:
                            curr = 0
                        st.markdown(f"**{task_title}** — {status}")
                        comp = st.slider("Completion %", 0, 100, curr, key=r.get("_id"))
                        if st.button(f"Update {r.get('_id')}", key=f"upd_{r.get('_id')}"):
                            r["completion"] = comp
                            r["marks"] = float(lin_reg.predict([[comp]])[0])
                            r["status"] = "In Progress" if comp < 100 else "Completed"
                            upsert_data(r.get("_id") or str(uuid.uuid4()), r)
                            st.success("Updated.")

            with t2:
                st.subheader("Leave Requests")
                st.write("Submit a leave request below.")
                typ = st.selectbox("Leave Type", ["Casual", "Sick", "Earned"])
                f = st.date_input("From")
                t = st.date_input("To")
                reason = st.text_area("Reason")
                if st.button("Submit Leave"):
                    lid = str(uuid.uuid4())
                    md = {
                        "employee": name,
                        "type": typ,
                        "from": str(f),
                        "to": str(t),
                        "reason": reason,
                        "status": "Pending",
                        "submitted": now()
                    }
                    upsert_data(lid, md)
                    st.success("Leave requested.")
                # show leave history
                if not my_leaves.empty:
                    my_leaves = my_leaves[["type","from","to","reason","status","submitted"]].fillna("")
                    st.markdown("### My leave history")
                    st.dataframe(my_leaves, use_container_width=True)

# ----------------------
# CLIENT
# ----------------------
elif role == "Client":
    st.header("Client Review Center")
    company = st.text_input("Company Name")
    if company:
        df_all = fetch_all()
        if df_all.empty:
            st.info("No data available.")
        else:
            # filter by company (case-insensitive)
            df_all["company_norm"] = df_all.get("company", "").astype(str).str.lower()
            df_client = df_all[df_all["company_norm"] == company.lower()]
            if df_client.empty:
                st.info("No tasks for this company.")
            else:
                tab1, tab2 = st.tabs(["Initial Review", "Re-Evaluation"])
                with tab1:
                    st.subheader("Initial Review")
                    unreviewed = df_client[df_client.get("client_reviewed") != True]
                    if unreviewed.empty:
                        st.info("No pending reviews.")
                    else:
                        choice = st.selectbox("Select task", unreviewed["task"].dropna().unique().tolist())
                        fb = st.text_area("Feedback")
                        rating = st.slider("Rating (1-5)", 1, 5, 3)
                        if st.button("Submit Feedback"):
                            rec = unreviewed[unreviewed["task"] == choice].iloc[0].to_dict()
                            rec["client_feedback"] = fb
                            rec["client_rating"] = rating
                            rec["client_reviewed"] = True
                            rec["client_approved_on"] = now()
                            upsert_data(rec.get("_id") or str(uuid.uuid4()), rec)
                            st.success("Feedback submitted.")

                with tab2:
                    st.subheader("Re-Evaluation")
                    reviewed = df_client[df_client.get("client_reviewed") == True]
                    if reviewed.empty:
                        st.info("No reviewed tasks to re-evaluate.")
                    else:
                        choice = st.selectbox("Select reviewed task", reviewed["task"].dropna().unique().tolist())
                        fb = st.text_area("Updated feedback (post-improvements)")
                        rating = st.slider("New rating (1-5)", 1, 5, 3)
                        if st.button("Submit Re-evaluation"):
                            rec = reviewed[reviewed["task"] == choice].iloc[0].to_dict()
                            rec["client_reval_feedback"] = fb
                            rec["client_rating_reval"] = rating
                            rec["client_revaluated"] = True
                            rec["client_reval_on"] = now()
                            upsert_data(rec.get("_id") or str(uuid.uuid4()), rec)
                            st.success("Re-evaluation submitted.")

# ----------------------
# ADMIN (HR) unified analytics + clustering
# ----------------------
elif role == "Admin":
    st.header("HR Intelligence Dashboard (Unified View)")
    df_all = fetch_all()
    if df_all.empty:
        st.info("No data available.")
    else:
        df = df_all.copy()

        # ensure required columns exist
        for col in ["employee", "department", "task", "completion", "marks", "type"]:
            if col not in df.columns:
                df[col] = ""

        # normalize numeric fields
        df["completion"] = pd.to_numeric(df.get("completion", 0), errors="coerce").fillna(0)
        df["marks"] = pd.to_numeric(df.get("marks", 0), errors="coerce").fillna(0)

        # compute cluster if enough rows
        if len(df) >= 3:
            km = KMeans(n_clusters=3, random_state=42, n_init=10)
            df["cluster"] = km.fit_predict(df[["completion", "marks"]])
        else:
            df["cluster"] = 0

        # compute risk score (simple heuristic) and bucket
        df["risk_score"] = (100 - df["completion"]) / 100 + (2 - df["marks"]) / 5
        df["risk_bucket"] = pd.cut(df["risk_score"], bins=[-999, 0.5, 1.2, 999], labels=["Low", "Medium", "High"])

        # summary metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Average completion", f"{df['completion'].mean():.1f}%")
        c2.metric("Average marks", f"{df['marks'].mean():.2f}")
        c3.metric("Average risk", f"{df['risk_score'].mean():.2f}")

        st.subheader("Employee Performance & Clusters")
        display_cols = ["employee", "department", "task", "completion", "marks", "cluster", "risk_score", "risk_bucket"]
        st.dataframe(df[display_cols].fillna(""), use_container_width=True)

        # visualization: completion vs marks colored by cluster
        try:
            fig = px.scatter(df, x="completion", y="marks", color=df["cluster"].astype(str),
                             hover_data=["employee", "department", "task", "risk_score"],
                             title="Performance clusters (completion vs marks)", size="risk_score")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Plotting failed: {e}")

        # automated HR alerts for high-risk employees
        st.subheader("Automated HR Alerts")
        high_risk = df[df["risk_bucket"] == "High"]
        if not high_risk.empty:
            st.error("High-risk employees detected:")
            for _, r in high_risk.iterrows():
                st.write(f"- {r.get('employee','<unknown>')} (risk={r.get('risk_score'):.2f}) — low completion {r.get('completion')}%, marks {r.get('marks')}")
        else:
            st.success("No high-risk employees detected.")

# ----------------------
# End of app
# ----------------------
