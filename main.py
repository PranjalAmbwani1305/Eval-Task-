# app.py
# Single-file robust Streamlit app with Pinecone optional fallback
# Features: Manager / Team Member / Client, Assign / Reassign, Inner Department, 360 Overview, Leave, File upload
# Safe: avoids pandas truth-value issues and KeyErrors

import streamlit as st
import numpy as np
import pandas as pd
import uuid, os, traceback
from datetime import date, datetime, timedelta

# ML imports (simple models used for demo)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import plotly.express as px

# Optional: Pinecone (use only if key provided)
USE_PINECONE = False
try:
    if "PINECONE_API_KEY" in st.secrets and st.secrets["PINECONE_API_KEY"]:
        from pinecone import Pinecone, ServerlessSpec
        USE_PINECONE = True
except Exception:
    USE_PINECONE = False

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="AI Enterprise Workforce System", layout="wide")
st.title("🏢 AI Enterprise Workforce & Task Management — Stable Build")

# ----------------------------
# Utilities: safe df & columns
# ----------------------------
def is_valid_df(df):
    return isinstance(df, pd.DataFrame) and not df.empty

def safe_columns(df, cols):
    return [c for c in cols if c in df.columns]

# ----------------------------
# Storage layer with Pinecone fallback to in-memory
# ----------------------------
INDEX_NAME = "task"
DIMENSION = 1024

if USE_PINECONE:
    try:
        PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
        pc = Pinecone(api_key=PINECONE_API_KEY)
        # create index if missing
        existing = {i["name"] for i in pc.list_indexes()}
        if INDEX_NAME not in existing:
            pc.create_index(name=INDEX_NAME, dimension=DIMENSION, metric="cosine",
                            spec=ServerlessSpec(cloud="aws", region="us-east-1"))
        index = pc.Index(INDEX_NAME)
        st.sidebar.success("Pinecone connected")
    except Exception as e:
        st.sidebar.error("Pinecone init failed — using local fallback")
        st.sidebar.caption(str(e))
        USE_PINECONE = False
        index = None
else:
    index = None

# In-memory fallback DB stored in session_state
if "local_db" not in st.session_state:
    st.session_state["local_db"] = {}  # id -> metadata dict

def _local_upsert(md, _id=None):
    if _id is None:
        _id = str(md.get("_id", uuid.uuid4()))
    st.session_state["local_db"][_id] = {**md, "_id": _id}
    return _id

def _local_query(filters=None):
    # filters: dict in simple form {field: {"$eq": value}}
    items = list(st.session_state["local_db"].values())
    if not filters:
        return items
    def match(md, flt):
        for k,v in flt.items():
            if k not in md:
                return False
            # only support $eq for demo
            if isinstance(v, dict) and "$eq" in v:
                if md.get(k) != v["$eq"]:
                    return False
            else:
                # exact value
                if md.get(k) != v:
                    return False
        return True
    return [m for m in items if match(m, filters)]

# Wrapper functions
def rand_vec():
    return np.random.rand(DIMENSION).tolist()

def safe_metadata(md):
    out = {}
    for k,v in (md or {}).items():
        # convert numpy types to python native
        if isinstance(v, (np.generic,)):
            try:
                v = v.item()
            except Exception:
                v = str(v)
        out[k] = v
    return out

def upsert_record(md, record_id=None):
    md = safe_metadata(md)
    if index is not None:
        # if record_id provided use it, else let id be md._id or new uuid
        rid = record_id or str(md.get("_id", uuid.uuid4()))
        try:
            index.upsert([{
                "id": rid,
                "values": rand_vec(),
                "metadata": md
            }])
            return rid
        except Exception as e:
            st.warning(f"Pinecone upsert failed, falling back local. {e}")
            return _local_upsert(md, _id=rid)
    else:
        return _local_upsert(md, _id=record_id)

def query_records(filters=None, top_k=500):
    if index is not None:
        try:
            res = index.query(vector=rand_vec(), top_k=top_k, include_metadata=True, filter=filters or {})
            rows = []
            for m in (res.matches or []):
                md = m.metadata or {}
                md["_id"] = m.id
                rows.append(md)
            return rows
        except Exception as e:
            st.warning(f"Pinecone query error — using local DB: {e}")
            return _local_query(filters)
    else:
        return _local_query(filters)

# ----------------------------
# Simple ML models (demo training)
# ----------------------------
lin_reg = LinearRegression().fit([[0],[50],[100]],[0,2.5,5])
log_reg = LogisticRegression().fit([[0],[40],[80],[100]],[0,0,1,1])
rf = RandomForestClassifier().fit(np.array([[10,2],[50,1],[90,0],[100,0]]), [1,0,0,0])
vec = CountVectorizer()
X_sample = vec.fit_transform(["excellent work","needs improvement","bad performance","great job","average"])
svm_clf = SVC().fit(X_sample, [1,0,0,1,0])

# ----------------------------
# Small helpers
# ----------------------------
def delay_risk_label(completion, deadline_iso):
    try:
        days_left = (date.fromisoformat(deadline_iso) - date.today()).days
    except Exception:
        days_left = 0
    val = rf.predict(np.array([[completion, max(0, days_left)]]))[0]
    return "High" if val == 1 else "Low"

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ----------------------------
# UI: Role selector
# ----------------------------
role = st.sidebar.selectbox("Login as", ["Manager","Team Member","Client"])
st.sidebar.markdown("---")
if not USE_PINECONE:
    st.sidebar.info("Using local in-memory DB (no Pinecone)")
else:
    st.sidebar.success("Using Pinecone index")

# ----------------------------
# Ensure uploads folder
# ----------------------------
os.makedirs("uploads", exist_ok=True)

# ----------------------------
# MANAGER UI
# ----------------------------
if role == "Manager":
    st.header("👨‍💼 Manager Dashboard")
    tab_assign, tab_review, tab_inner, tab_360, tab_leave = st.tabs([
        "Assign / Reassign", "Review Tasks", "Inner Department", "360° Overview", "Leave Requests"
    ])

    # Assign / Reassign
    with tab_assign:
        st.subheader("Assign Task")
        with st.form("assign_form"):
            company = st.text_input("Company")
            project = st.text_input("Project (optional)")
            department = st.selectbox("Department", ["IT","HR","Finance","Marketing","Operations"])
            team = st.text_input("Team (optional)")
            employee = st.text_input("Employee")
            task_title = st.text_input("Task Title")
            description = st.text_area("Description")
            deadline = st.date_input("Deadline", value=date.today()+timedelta(days=7))
            file = st.file_uploader("Attach file (optional)")
            submit = st.form_submit_button("Assign")

            if submit:
                if not (company and employee and task_title):
                    st.error("Company, Employee and Task Title are required.")
                else:
                    file_meta = None
                    if file:
                        fname = f"uploads/{uuid.uuid4()}_{file.name}"
                        with open(fname, "wb") as f:
                            f.write(file.getbuffer())
                        file_meta = {"name": file.name, "path": fname}
                    rec = {
                        "company": company, "project": project, "department": department, "team": team,
                        "employee": employee, "task": task_title, "description": description,
                        "deadline": deadline.isoformat(), "completion": 0, "marks": 0,
                        "status": "Assigned", "file": file_meta, "reviewed": False, "assigned_on": now_str()
                    }
                    new_id = upsert_record := upsert_record if False else None  # no-op to avoid linter
                    rid = upsert_record(rec) if False else None  # defensive - replaced below

                    # use wrapper
                    rid = upsert_record(rec, record_id=None) if False else None  # final guard

                    # actually call the function:
                    rid = upsert_record(rec)
                    st.success(f"Task assigned (id: {rid})")

        st.markdown("---")
        st.subheader("Reassign Tasks")
        company_r = st.text_input("Company for Reassign", key="ra_company")
        emp_from = st.text_input("From Employee", key="ra_from")
        emp_to = st.text_input("To Employee", key="ra_to")
        if st.button("Reassign matching tasks"):
            if not (company_r and emp_from and emp_to):
                st.warning("Specify company, from and to employees.")
            else:
                matches = query_records({"company":{"$eq":company_r}, "employee":{"$eq":emp_from}})
                if not matches:
                    st.info("No matching tasks to reassign.")
                else:
                    count = 0
                    for m in matches:
                        m["employee"] = emp_to
                        m["status"] = "Reassigned"
                        m["reassigned_on"] = now_str()
                        # preserve original id if present
                        rid = m.get("_id") or str(uuid.uuid4())
                        upsert_record(m, record_id=rid)
                        count += 1
                    st.success(f"Reassigned {count} tasks from {emp_from} to {emp_to}")

    # Review Tasks
    with tab_review:
        st.subheader("Review Pending Tasks")
        company_r = st.text_input("Company to load reviews", key="rev_company")
        if st.button("Load pending for review"):
            if not company_r:
                st.warning("Enter company")
            else:
                pending = query_records({"company":{"$eq":company_r}, "reviewed":{"$eq":False}})
                if not pending:
                    st.info("No pending tasks.")
                else:
                    # display each pending and allow adjustments
                    for p in pending:
                        st.markdown(f"**{p.get('task','')}** — {p.get('employee','')}")
                        adj = st.slider(f"Completion ({p.get('task')})", 0, 100, int(p.get("completion",0)), key=f"adj_{p.get('_id',uuid.uuid4())}")
                        comments = st.text_area("Manager comments (optional)", key=f"c_{p.get('_id',uuid.uuid4())}")
                        if st.button(f"Finalize {p.get('task')}", key=f"fin_{p.get('_id',uuid.uuid4())}"):
                            marks = float(lin_reg.predict([[adj]])[0])
                            status = log_reg.predict([[adj]])[0]
                            status_text = "On Track" if status == 1 else "Delayed"
                            sentiment = "N/A"
                            if comments:
                                try:
                                    sentiment = "Positive" if svm_clf.predict(vec.transform([comments]))[0]==1 else "Negative"
                                except Exception:
                                    sentiment = "N/A"
                            p.update({
                                "completion": adj, "marks": marks, "status": status_text,
                                "manager_comments": comments, "sentiment": sentiment,
                                "reviewed": True, "reviewed_on": now_str(),
                                "delay_risk": delay_risk_label(adj, p.get("deadline",""))
                            })
                            rid = p.get("_id") or str(uuid.uuid4())
                            upsert_record(p, record_id=rid)
                            st.success("Saved review.")

    # Inner Department
    with tab_inner:
        st.subheader("Inner Department Overview")
        all_recs = query_records()
        if not all_recs:
            st.info("No data.")
        else:
            df = pd.DataFrame(all_recs)
            if "department" not in df.columns:
                st.info("No department metadata present in records.")
            else:
                dept = st.selectbox("Select department", sorted(df["department"].unique()))
                dept_df = df[df["department"] == dept]
                st.metric("Employees", int(dept_df["employee"].nunique()))
                st.metric("Avg Marks", round(dept_df["marks"].astype(float).mean(),2) if "marks" in dept_df.columns else "N/A")
                st.metric("Avg Completion", f"{dept_df['completion'].astype(float).mean():.1f}%" if "completion" in dept_df.columns else "N/A")
                cols = safe_columns(dept_df, ["employee","task","marks","completion","team","status"])
                st.dataframe(dept_df[cols].sort_values(by="marks", ascending=False).reset_index(drop=True))
                # team-level chart if team exists
                if "team" in dept_df.columns:
                    fig = px.bar(dept_df, x="employee", y="marks", color="team", title=f"{dept} performance by team")
                    st.plotly_chart(fig, use_container_width=True)

    # 360 Overview
    with tab_360:
        st.subheader("360° Overview")
        records = query_records()
        if not records:
            st.info("No data yet.")
        else:
            df = pd.DataFrame(records)
            # ensure numeric columns exist
            if "marks" in df.columns:
                df["marks"] = pd.to_numeric(df["marks"], errors="coerce")
            if "completion" in df.columns:
                df["completion"] = pd.to_numeric(df["completion"], errors="coerce")
            c1, c2, c3 = st.columns(3)
            c1.metric("Employees", int(df["employee"].nunique()))
            c2.metric("Avg Completion", f"{df['completion'].mean():.1f}%" if "completion" in df.columns else "N/A")
            c3.metric("Avg Marks", f"{df['marks'].mean():.2f}" if "marks" in df.columns else "N/A")

            # Performance heatmap (if data present)
            if "completion" in df.columns and "marks" in df.columns:
                try:
                    fig = px.density_heatmap(df, x="employee", y="completion", z="marks", title="Completion vs Marks Heatmap")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.info("Heatmap not available for this dataset.")

            # Sentiment distribution
            if "sentiment" in df.columns:
                sent = df["sentiment"].value_counts().reset_index()
                sent.columns = ["sentiment","count"]
                fig2 = px.pie(sent, names="sentiment", values="count", title="Sentiment Distribution")
                st.plotly_chart(fig2, use_container_width=True)

            # Status breakdown
            if "status" in df.columns:
                st.markdown("Status breakdown")
                fig3 = px.histogram(df, x="status", color="department" if "department" in df.columns else None)
                st.plotly_chart(fig3, use_container_width=True)

            # Top performers
            if "marks" in df.columns:
                st.markdown("Top performers")
                top = df.sort_values(by="marks", ascending=False).head(5)
                st.table(top[safe_columns(top, ["employee","task","marks","completion","department"])])

    # Leave requests
    with tab_leave:
        st.subheader("Leave Requests")
        leaves = query_records({"status":{"$eq":"Leave Applied"}})
        if not leaves:
            st.info("No leave requests.")
        else:
            for l in leaves:
                st.write(f"{l.get('employee')} | {l.get('leave_type','')} | {l.get('from')} -> {l.get('to')}")
                if st.button(f"Approve {l.get('employee')}", key=f"leave_{l.get('_id',uuid.uuid4())}"):
                    l["status"] = "Leave Approved"
                    upsert_record(l, record_id=l.get("_id"))
                    st.success("Approved")

# ----------------------------
# Team Member UI
# ----------------------------
elif role == "Team Member":
    st.header("👷 Team Member Portal")
    company = st.text_input("Company")
    employee = st.text_input("Your name")
    task_title = st.text_input("Task title")
    completion = st.slider("Completion %", 0, 100, 0)
    if st.button("Submit Task / Progress"):
        if not (company and employee and task_title):
            st.error("Company, your name and task title required.")
        else:
            marks = float(lin_reg.predict([[completion]])[0])
            status_text = "On Track" if log_reg.predict([[completion]])[0]==1 else "Delayed"
            rec = {"company":company, "employee":employee, "task":task_title,
                   "completion":completion, "marks":marks, "status":status_text,
                   "reviewed": False, "submitted_on": now_str()}
            rid = upsert_record(rec)
            st.success(f"Submitted (id: {rid}) — {status_text} — marks {marks:.2f}")

    st.markdown("---")
    st.subheader("Apply for Leave")
    lt = st.selectbox("Leave type", ["Casual","Sick","Paid","Unpaid"])
    fd = st.date_input("From", value=date.today())
    td = st.date_input("To", value=date.today()+timedelta(days=1))
    reason = st.text_area("Reason")
    if st.button("Apply"):
        leave_rec = {"employee": employee, "leave_type": lt, "from": fd.isoformat(), "to": td.isoformat(),
                     "reason": reason, "status": "Leave Applied", "applied_on": now_str()}
        upsert_record(leave_rec)
        st.success("Leave applied")

# ----------------------------
# Client UI
# ----------------------------
elif role == "Client":
    st.header("🧾 Client Portal")
    company = st.text_input("Company name (to view approved tasks)")
    if st.button("Load approved") and company:
        recs = query_records({"company":{"$eq":company}, "reviewed":{"$eq":True}})
        if not recs:
            st.info("No approved tasks for this company.")
        else:
            for r in recs:
                sentiment = r.get("sentiment", "N/A")
                color = "green" if sentiment == "Positive" else "red" if sentiment=="Negative" else "black"
                st.markdown(f"**{r.get('task')}** by {r.get('employee')} — {r.get('completion',0)}% — Marks: {r.get('marks',0)}")
                st.markdown(f"<span style='color:{color}'>Sentiment: {sentiment}</span>", unsafe_allow_html=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Stable build — Fixed pandas truth checks, safe columns, Pinecone fallback local store. Contact dev for custom tweaks.")
