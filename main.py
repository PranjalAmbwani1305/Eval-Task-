# ============================================================
# main.py — Enterprise Workforce Performance Intelligence System (AI-enhanced)
# ============================================================
# Requirements:
# pip install streamlit pinecone-client scikit-learn plotly huggingface-hub pandas openpyxl PyPDF2

import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import plotly.express as px
import numpy as np
import pandas as pd
import uuid
import json
import time
from datetime import datetime, date
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError

# optional
try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False
try:
    import PyPDF2
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

# ----------------------------
# App config
# ----------------------------
st.set_page_config(page_title="AI Workforce Intelligence Platform", layout="wide")
st.title("AI Workforce Intelligence Platform")
st.caption("Workforce analytics with lightweight AI predictions")

# ----------------------------
# Secrets & constants
# ----------------------------
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
HF_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")
INDEX_NAME = "task"
DIMENSION = 1024

# ----------------------------
# Pinecone init (best-effort)
# ----------------------------
pc, index = None, None
if PINECONE_API_KEY:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        existing = [i["name"] for i in pc.list_indexes()]
        if INDEX_NAME not in existing and existing:
            INDEX_NAME = existing[0]
        elif INDEX_NAME not in existing:
            pc.create_index(name=INDEX_NAME, dimension=DIMENSION, metric="cosine",
                            spec=ServerlessSpec(cloud="aws", region="us-east-1"))
            # best-effort wait
            for _ in range(10):
                try:
                    desc = pc.describe_index(INDEX_NAME)
                    if desc.get("status", {}).get("ready"):
                        break
                except Exception:
                    time.sleep(1)
        index = pc.Index(INDEX_NAME)
        st.caption(f"Connected to Pinecone index: {INDEX_NAME}")
    except Exception as e:
        st.warning(f"Pinecone connection failed — running in local mode. ({e})")
else:
    st.warning("Pinecone API key not detected — operating in local data mode.")

# ----------------------------
# Utility helpers
# ----------------------------
def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def rand_vec():
    return np.random.rand(DIMENSION).tolist()

def safe_meta(md: dict) -> dict:
    clean = {}
    for k, v in (md or {}).items():
        try:
            if isinstance(v, (datetime, date)):
                clean[k] = v.isoformat()
            elif isinstance(v, (dict, list)):
                clean[k] = json.dumps(v)
            elif pd.isna(v):
                clean[k] = ""
            else:
                clean[k] = str(v) if not isinstance(v, (int, float, bool)) else v
        except Exception:
            clean[k] = str(v)
    return clean

def upsert_data(id_, md: dict) -> bool:
    id_ = str(id_)
    md = dict(md)
    if not index:
        local = st.session_state.setdefault("LOCAL_DATA", {})
        local[id_] = md
        return True
    try:
        index.upsert([{"id": id_, "values": rand_vec(), "metadata": safe_meta(md)}])
        return True
    except Exception as e:
        st.warning(f"Pinecone upsert failed: {e}")
        return False

def fetch_all() -> pd.DataFrame:
    if not index:
        local = st.session_state.get("LOCAL_DATA", {})
        rows = []
        for k, md in local.items():
            rec = dict(md)
            rec["_id"] = k
            rows.append(rec)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)
    try:
        stats = index.describe_index_stats()
        if stats and stats.get("total_vector_count", 0) == 0:
            return pd.DataFrame()
    except Exception:
        pass
    try:
        res = index.query(vector=rand_vec(), top_k=1000, include_metadata=True)
        matches = getattr(res, "matches", None) or res.get("matches", [])
        if not matches:
            return pd.DataFrame()
        rows = []
        for m in matches:
            if hasattr(m, "metadata"):
                md = m.metadata or {}
                mid = getattr(m, "id", None)
            else:
                md = m.get("metadata", {}) or {}
                mid = m.get("id")
            md = dict(md)
            md["_id"] = mid
            rows.append(md)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"Fetch failed: {e}")
        return pd.DataFrame()

def parse_attendees_field(v):
    if isinstance(v, list):
        return [a.strip().lower() for a in v]
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("["):
            try:
                return [a.strip().lower() for a in json.loads(s.replace("'", '"'))]
            except Exception:
                pass
        return [a.strip().lower() for a in s.split(",") if a.strip()]
    return []

def extract_file_text(fobj):
    try:
        name = fobj.name.lower()
        if name.endswith(".txt") or name.endswith(".csv"):
            return fobj.read().decode("utf-8", errors="ignore")[:20000]
        if name.endswith(".pdf") and PDF_AVAILABLE:
            reader = PyPDF2.PdfReader(fobj)
            pages = [p.extract_text() or "" for p in reader.pages]
            return "\n".join(pages)[:20000]
    except Exception:
        return ""
    return ""

# ----------------------------
# Prediction utilities
# ----------------------------
def prepare_task_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and ensure numeric columns and deadline days_remaining."""
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()
    df = raw_df.copy()
    # ensure columns exist
    for c in ["completion", "marks", "deadline", "created", "task", "employee", "company", "status"]:
        if c not in df.columns:
            df[c] = ""
    # numeric conversions
    df["completion"] = pd.to_numeric(df["completion"], errors="coerce").fillna(0)
    df["marks"] = pd.to_numeric(df["marks"], errors="coerce").fillna(0)
    # compute days_remaining where possible
    def days_remaining(val):
        try:
            if not val:
                return np.nan
            # try ISO parse
            d = pd.to_datetime(val)
            delta = (d.date() - datetime.now().date()).days
            return int(delta)
        except Exception:
            try:
                return int(float(val))
            except Exception:
                return np.nan
    df["days_remaining"] = df.get("deadline", "").apply(days_remaining)
    return df

def train_predictors(df_tasks: pd.DataFrame):
    """
    Train lightweight models if sufficient data exists.
    Returns fitted (lr_marks, log_status, rf_risk) where any may be None.
    """
    lr_marks = None
    log_status = None
    rf_risk = None

    # Train linear regression for marks if we have >=2 points
    try:
        mask_marks = df_tasks["marks"].notna() & (df_tasks["marks"].astype(float) >= 0)
        if mask_marks.sum() >= 2:
            Xm = df_tasks.loc[mask_marks, ["completion", "days_remaining"]].fillna(0).astype(float)
            ym = df_tasks.loc[mask_marks, "marks"].astype(float)
            lr = LinearRegression()
            lr.fit(Xm, ym)
            lr_marks = lr
    except Exception:
        lr_marks = None

    # Train logistic regression for status (Completed vs In Progress) if labels exist
    try:
        # define label: Completed if completion >=100 or status contains Completed
        df_tasks["label_status"] = ((df_tasks["completion"].astype(float) >= 100) | df_tasks["status"].astype(str).str.lower().str.contains("completed")).astype(int)
        if df_tasks["label_status"].nunique() > 1 and len(df_tasks) >= 10:
            Xs = df_tasks[["completion", "days_remaining", "marks"]].fillna(0).astype(float)
            ys = df_tasks["label_status"].astype(int)
            lr = LogisticRegression(max_iter=500)
            lr.fit(Xs, ys)
            log_status = lr
    except Exception:
        log_status = None

    # Train RandomForest for risk (On Track vs At Risk)
    try:
        # create a risk label for training: At Risk if completion / max(1, days_remaining) < threshold (heuristic)
        df_tasks["risk_label"] = df_tasks.apply(lambda r: 1 if ( (r["days_remaining"] is not np.nan and pd.notna(r["days_remaining"]) and r["days_remaining"] >= 0 and (r["completion"] / max(1, r["days_remaining"])) < 20) or (r["completion"] < 50 and (r.get("days_remaining") is not np.nan and pd.notna(r["days_remaining"]) and r["days_remaining"] < 3)) ) else 0, axis=1)
        if df_tasks["risk_label"].nunique() > 1 and len(df_tasks) >= 10:
            Xr = df_tasks[["completion", "days_remaining", "marks"]].fillna(0).astype(float)
            yr = df_tasks["risk_label"].astype(int)
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            rf.fit(Xr, yr)
            rf_risk = rf
    except Exception:
        rf_risk = None

    return lr_marks, log_status, rf_risk

def predict_for_tasks(df_tasks: pd.DataFrame):
    """
    Add predicted columns to df_tasks: predicted_marks, predicted_status, predicted_risk.
    Uses trained models where available, else fallbacks to heuristics.
    """
    df = prepare_task_df(df_tasks)
    if df.empty:
        return df

    # Train models on historical tasks from df
    lr_marks, log_status, rf_risk = train_predictors(df)

    # Predicted marks
    preds_marks = []
    for _, r in df.iterrows():
        # prefer model if available
        try:
            if lr_marks is not None:
                Xp = np.array([[r["completion"], r["days_remaining"] if pd.notna(r["days_remaining"]) else 0]])
                pm = float(lr_marks.predict(Xp)[0])
            else:
                # fallback simple linear mapping: marks roughly (completion/100)*5
                pm = float((r["completion"] / 100.0) * 5.0)
        except Exception:
            pm = float((r["completion"] / 100.0) * 5.0)
        preds_marks.append(round(pm, 3))
    df["predicted_marks"] = preds_marks

    # Predicted status: 1 -> Completed, 0 -> In Progress
    preds_status = []
    for _, r in df.iterrows():
        try:
            if log_status is not None:
                Xp = np.array([[r["completion"], r["days_remaining"] if pd.notna(r["days_remaining"]) else 0, r["marks"]]])
                p = log_status.predict(Xp)[0]
                ps = "Completed" if int(p) == 1 else "In Progress"
            else:
                ps = "Completed" if r["completion"] >= 100 else "In Progress"
        except Exception:
            ps = "Completed" if r["completion"] >= 100 else "In Progress"
        preds_status.append(ps)
    df["predicted_status"] = preds_status

    # Predicted risk: 'At Risk' or 'On Track'
    preds_risk = []
    for _, r in df.iterrows():
        try:
            if rf_risk is not None:
                Xp = np.array([[r["completion"], r["days_remaining"] if pd.notna(r["days_remaining"]) else 0, r["marks"]]])
                p = rf_risk.predict(Xp)[0]
                pr = "At Risk" if int(p) == 1 else "On Track"
            else:
                # heuristic fallback:
                dr = r["days_remaining"] if pd.notna(r["days_remaining"]) else 999
                if r["completion"] >= 90 and (dr >= 2 or dr == 999):
                    pr = "On Track"
                elif dr < 0 and r["completion"] < 100:
                    pr = "At Risk"
                elif r["completion"] < 50 and dr <= 7:
                    pr = "At Risk"
                else:
                    pr = "On Track"
        except Exception:
            pr = "On Track"
        preds_risk.append(pr)
    df["predicted_risk"] = preds_risk

    return df

# ----------------------------
# Role-based Views
# ----------------------------
role = st.sidebar.selectbox("Access Portal As", ["Manager", "Team Member", "Client", "HR Administrator"])
today_str = datetime.now().strftime("%Y-%m-%d")

# ----------------------------
# MANAGER
# ----------------------------
if role == "Manager":
    st.header("Manager Command Center — Task & Team Oversight")
    tabs = st.tabs(["Task Management", "Feedback", "Meetings", "Leave Decisions", "Team Overview"])

    # Task Management
    with tabs[0]:
        st.subheader("Assign New Task")
        with st.form("assign_task"):
            company = st.text_input("Company Name (optional)")
            dept = st.text_input("Department (optional)")
            emp = st.text_input("Employee Name")
            task = st.text_input("Task Title")
            desc = st.text_area("Task Description (optional)")
            deadline = st.date_input("Deadline (optional)", value=date.today())
            attachment = st.file_uploader("Optional attachment (pdf/txt/csv/xlsx)", type=["pdf","txt","csv","xlsx"])
            submit = st.form_submit_button("Assign Task")
            if submit:
                if not emp or not task:
                    st.warning("Employee and Task Title required.")
                else:
                    tid = str(uuid.uuid4())
                    md = {
                        "type": "Task",
                        "company": company or "",
                        "department": dept or "",
                        "employee": emp,
                        "task": task,
                        "description": desc or "",
                        "deadline": str(deadline) if deadline else "",
                        "completion": 0,
                        "marks": 0,
                        "status": "Assigned",
                        "created": now()
                    }
                    if attachment:
                        md["attachment_name"] = attachment.name
                        md["attachment_text"] = extract_file_text(attachment)
                    upsert_data(tid, md)
                    st.success(f"Task '{task}' assigned to {emp}")

        df = fetch_all()
        df_tasks = pd.DataFrame()
        if not df.empty:
            df_tasks = df[df.get("type") == "Task"]
        if df_tasks.empty:
            st.info("No tasks.")
        else:
            # predictions
            df_pred = predict_for_tasks(df_tasks)
            display_cols = ["employee", "task", "status", "completion", "predicted_marks", "predicted_status", "predicted_risk", "deadline"]
            for c in display_cols:
                if c not in df_pred.columns:
                    df_pred[c] = ""
            st.dataframe(df_pred[display_cols].fillna(""), use_container_width=True)

    # Feedback
    with tabs[1]:
        st.subheader("Manager Feedback & Evaluation")
        df_all = fetch_all()
        if df_all.empty:
            st.info("No data.")
        else:
            df_tasks = df_all[df_all.get("type") == "Task"]
            if df_tasks.empty:
                st.info("No tasks.")
            else:
                df_tasks["completion"] = pd.to_numeric(df_tasks.get("completion", 0), errors="coerce").fillna(0)
                df_tasks["marks"] = pd.to_numeric(df_tasks.get("marks", 0), errors="coerce").fillna(0)

                pending_review = df_tasks[(df_tasks["completion"] >= 100) & (df_tasks.get("status") != "Under Client Review")]

                if pending_review.empty:
                    st.info("No completed tasks awaiting review.")
                else:
                    emp_search = st.text_input("Search by employee name (optional)", value="")
                    if emp_search:
                        pending_review = pending_review[pending_review["employee"].astype(str).str.contains(emp_search.strip(), case=False, na=False)]

                    task_list = pending_review["task"].dropna().unique().tolist()
                    if not task_list:
                        st.info("No matching completed tasks.")
                    else:
                        task_sel = st.selectbox("Select completed task", task_list)
                        rec = pending_review[pending_review["task"] == task_sel].iloc[0].to_dict()
                        st.write(f"Employee: {rec.get('employee','')}")
                        st.write(f"Task: {rec.get('task','')}")
                        # show AI predictions for this task
                        single_pred = predict_for_tasks(pd.DataFrame([rec]))
                        if not single_pred.empty:
                            st.markdown("**AI Predictions**")
                            st.write({
                                "Predicted Marks": single_pred.iloc[0].get("predicted_marks"),
                                "Predicted Status": single_pred.iloc[0].get("predicted_status"),
                                "Predicted Risk": single_pred.iloc[0].get("predicted_risk")
                            })
                        final_marks = st.slider("Final Performance Score (0–5)", 0.0, 5.0, float(rec.get("marks", 0)))
                        fb = st.text_area("Manager Feedback")
                        if st.button("Finalize Review"):
                            rec["marks"] = final_marks
                            rec["manager_feedback"] = fb
                            rec["manager_reviewed_on"] = now()
                            rec["status"] = "Under Client Review"
                            upsert_data(rec.get("_id") or str(uuid.uuid4()), rec)
                            st.success("Feedback finalized and sent for client evaluation.")

    # Meetings
    with tabs[2]:
        st.subheader("Meeting Scheduler & Notes")
        with st.form("schedule_meet"):
            title = st.text_input("Meeting Title")
            date_meet = st.date_input("Date", date.today())
            time_meet = st.text_input("Time", "10:00 AM")
            attendees = st.text_area("Attendees (comma-separated)")
            notes = st.file_uploader("Optional notes (txt/pdf/csv/xlsx)", type=["txt","pdf","csv","xlsx"])
            submit_meet = st.form_submit_button("Schedule Meeting")
            if submit_meet:
                mid = str(uuid.uuid4())
                md = {
                    "type": "Meeting",
                    "meeting_title": title,
                    "meeting_date": str(date_meet),
                    "meeting_time": time_meet,
                    "attendees": json.dumps(parse_attendees_field(attendees)),
                    "created": now()
                }
                if notes:
                    md["notes_file"] = notes.name
                    md["notes_text"] = extract_file_text(notes)
                upsert_data(mid, md)
                st.success("Meeting scheduled.")

    # Leave Approvals
    with tabs[3]:
        st.subheader("Leave Decision Center")
        df = fetch_all()
        leaves = df[df.get("type") == "Leave"] if not df.empty else pd.DataFrame()
        if leaves.empty:
            st.info("No pending leave requests.")
        else:
            if "status" not in leaves.columns:
                leaves["status"] = ""
            pending = leaves[leaves["status"].astype(str).str.strip().str.lower() == "pending"]
            if pending.empty:
                st.info("No pending leave requests.")
            else:
                for i, r in pending.iterrows():
                    emp = r.get("employee")
                    st.markdown(f"**{emp}** → {r.get('from')} to {r.get('to')} ({r.get('reason')})")
                    with st.form(f"leave_form_{i}"):
                        dec = st.radio("Decision", ["Approve", "Reject"], key=f"dec_{i}")
                        remarks = st.text_area("Remarks (optional)", key=f"rem_{i}")
                        submit_dec = st.form_submit_button("Submit Decision")
                        if submit_dec:
                            updated = r.to_dict()
                            updated["_id"] = str(r.get("_id") or uuid.uuid4())
                            updated["status"] = "Approved" if dec == "Approve" else "Rejected"
                            updated["approved_by"] = "Manager"
                            updated["approved_on"] = now()
                            updated["manager_remarks"] = remarks
                            audit = updated.get("audit_log", "")
                            entry = f"{updated['status']} by Manager on {now()} | {remarks}"
                            updated["audit_log"] = (audit + "\n" + entry) if audit else entry
                            upsert_data(updated["_id"], updated)
                            st.success(f"Leave {updated['status']} for {emp}")
                            st.experimental_rerun()

    # Team Overview
    with tabs[4]:
        st.subheader("Team Overview — Task Snapshot with AI Predictions")
        df_all = fetch_all()
        df_tasks = df_all[df_all.get("type") == "Task"] if not df_all.empty else pd.DataFrame()
        if df_tasks.empty:
            st.info("No tasks.")
        else:
            df_pred = predict_for_tasks(df_tasks)
            for c in ["company", "department", "employee", "task", "completion", "predicted_marks", "predicted_status", "predicted_risk", "deadline"]:
                if c not in df_pred.columns:
                    df_pred[c] = ""
            st.dataframe(df_pred[["company","department","employee","task","completion","predicted_marks","predicted_status","predicted_risk","deadline"]].fillna(""), use_container_width=True)
            try:
                if "department" in df_pred.columns and df_pred["department"].notna().any():
                    fig = px.bar(df_pred, x="employee", y="completion", color="department", title="Completion by Employee")
                else:
                    fig = px.bar(df_pred, x="employee", y="completion", title="Completion by Employee")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

# ----------------------------
# TEAM MEMBER
# ----------------------------
elif role == "Team Member":
    st.header("Employee Workspace — Task Progress & Leave Requests")
    name = st.text_input("Enter your name")
    company = st.text_input("Company (optional)")

    if name:
        df_all = fetch_all()
        if df_all.empty:
            st.info("No data present.")
        else:
            if "type" not in df_all.columns:
                df_all["type"] = df_all.apply(lambda r: "Task" if pd.notna(r.get("task")) and r.get("task") != "" else "Leave", axis=1)

            cond = (df_all.get("employee","").astype(str).str.lower() == name.lower())
            if company:
                cond = cond & (df_all.get("company","").astype(str).str.lower() == company.lower())
            my_tasks = df_all[cond & (df_all["type"] == "Task")]

            st.subheader("My Tasks")
            if my_tasks.empty:
                st.info("No tasks assigned.")
            else:
                for _, r in my_tasks.iterrows():
                    st.markdown(f"**{r.get('task','Untitled')}** — Status: {r.get('status','')}")
                    if r.get("manager_feedback"):
                        st.info(f"Manager Feedback: {r.get('manager_feedback')}")
                    if r.get("client_feedback"):
                        st.success(f"Client Feedback: {r.get('client_feedback')} (Rating: {r.get('client_rating')})")

                    curr = 0
                    try:
                        curr = int(float(r.get("completion", 0) or 0))
                    except Exception:
                        curr = 0
                    comp = st.slider("Completion %", 0, 100, curr, key=r.get("_id"))
                    items_done = st.number_input("Items / subtasks completed (optional)", min_value=0, value=int(r.get("items_done", 0) or 0), step=1, key=f"items_{r.get('_id')}")
                    attach = st.file_uploader("Optional attachment (evidence)", type=["pdf","txt","csv","xlsx"], key=f"up_{r.get('_id')}")
                    if st.button(f"Update {r.get('_id')}", key=f"upd_{r.get('_id')}"):
                        row = r.to_dict()
                        row["completion"] = comp
                        row["items_done"] = int(items_done)
                        # predicted marks quick heuristic
                        try:
                            row["marks"] = float((comp / 100.0) * 5.0)
                        except Exception:
                            row["marks"] = 0.0
                        row["status"] = "In Progress" if comp < 100 else "Completed"
                        if attach:
                            row["update_attachment_name"] = attach.name
                            row["update_attachment_text"] = extract_file_text(attach)
                        upsert_data(row.get("_id") or str(uuid.uuid4()), row)
                        st.success("Updated task progress.")

            st.markdown("---")
            st.subheader("My Meetings")
            meets = df_all[df_all.get("type") == "Meeting"] if not df_all.empty else pd.DataFrame()
            if meets.empty:
                st.info("No meetings scheduled.")
            else:
                def invited(row):
                    attendees = parse_attendees_field(row.get("attendees", ""))
                    return name.strip().lower() in attendees
                try:
                    invited_meets = meets[meets.apply(invited, axis=1)]
                    if invited_meets.empty:
                        st.info("No meetings with you as attendee.")
                    else:
                        display_cols = ["company","meeting_title","meeting_date","meeting_time"]
                        for c in display_cols:
                            if c not in invited_meets.columns:
                                invited_meets[c] = ""
                        st.dataframe(invited_meets[display_cols].fillna(""), use_container_width=True)
                except Exception as e:
                    st.error(f"Error filtering meetings: {e}")

            st.markdown("---")
            st.subheader("Leave Request / Status")
            member_leaves = df_all[(df_all.get("type") == "Leave") & (df_all.get("employee","").astype(str).str.lower() == name.lower())]
            if member_leaves.empty:
                st.info("You have no leave requests on record.")
            else:
                for c in ["leave_type","from","to","status","approved_by","approved_on","audit_log"]:
                    if c not in member_leaves.columns:
                        member_leaves[c] = ""
                st.dataframe(member_leaves[["leave_type","from","to","status","approved_by","approved_on"]].fillna(""), use_container_width=True)

            st.markdown("Apply for Leave")
            lt = st.selectbox("Leave Type", ["Casual","Sick","Earned"])
            f = st.date_input("From")
            t = st.date_input("To")
            reason = st.text_area("Reason")
            if st.button("Submit Leave Request"):
                lid = str(uuid.uuid4())
                md = {
                    "type": "Leave",
                    "employee": name,
                    "company": company or "",
                    "leave_type": lt,
                    "from": str(f),
                    "to": str(t),
                    "reason": reason,
                    "status": "Pending",
                    "submitted": now()
                }
                upsert_data(lid, md)
                st.success("Leave requested.")

# ----------------------------
# CLIENT
# ----------------------------
elif role == "Client":
    st.header("Client Review Portal — Project Feedback & Meetings")
    company = st.text_input("Enter Company Name")
    if company:
        df_all = fetch_all()
        df_client = df_all[df_all.get("company","").astype(str).str.lower() == company.lower()] if not df_all.empty else pd.DataFrame()
        if df_client.empty:
            st.info("No records for this company.")
        else:
            st.subheader("Tasks")
            df_tasks = df_client[df_client.get("type") == "Task"]
            if df_tasks.empty:
                st.info("No tasks for this company.")
            else:
                df_pred = predict_for_tasks(df_tasks)
                st.dataframe(df_pred[["employee","task","status","completion","predicted_marks","predicted_status","predicted_risk"]].fillna(""), use_container_width=True)

                st.markdown("---")
                st.subheader("Provide feedback for Manager-reviewed tasks")
                pending = df_tasks[(df_tasks.get("status") == "Under Client Review") & (df_tasks.get("client_reviewed") != True)]
                if pending.empty:
                    st.info("No tasks pending client review.")
                else:
                    task_list = pending["task"].dropna().unique().tolist()
                    task_sel = st.selectbox("Select task to review", task_list)
                    fb = st.text_area("Feedback")
                    rating = st.slider("Rating (1–5)", 1, 5, 3)
                    if st.button("Submit Feedback"):
                        rec = pending[pending["task"] == task_sel].iloc[0].to_dict()
                        rec["client_feedback"] = fb
                        rec["client_rating"] = rating
                        rec["client_reviewed"] = True
                        rec["client_approved_on"] = now()
                        rec["status"] = "Client Reviewed"
                        upsert_data(rec.get("_id") or str(uuid.uuid4()), rec)
                        st.success("Client feedback submitted.")

            st.markdown("---")
            st.subheader("Company Meetings")
            meets = df_client[df_client.get("type") == "Meeting"]
            if meets.empty:
                st.info("No meetings for this company.")
            else:
                def display_attendees(v):
                    try:
                        if isinstance(v, str):
                            return ", ".join(parse_attendees_field(v))
                        if isinstance(v, list):
                            return ", ".join([a for a in v])
                    except Exception:
                        return str(v)
                    return ""
                meets_display = meets.copy()
                meets_display["attendees_display"] = meets_display.get("attendees","").apply(display_attendees)
                st.dataframe(meets_display[["meeting_title","meeting_date","meeting_time","attendees_display"]].fillna(""), use_container_width=True)

# ----------------------------
# HR Administrator
# ----------------------------
elif role == "HR Administrator":
    st.header("HR Analytics — Performance & Leave Intelligence")
    df_all = fetch_all()
    if df_all.empty:
        st.info("No records.")
    else:
        for c in ["company","employee","department","task","completion","marks","type","status","leave_type","from","to"]:
            if c not in df_all.columns:
                df_all[c] = ""
        df_all["completion"] = pd.to_numeric(df_all.get("completion", 0), errors="coerce").fillna(0)
        df_all["marks"] = pd.to_numeric(df_all.get("marks", 0), errors="coerce").fillna(0)

        df_tasks = df_all[df_all["type"] == "Task"]
        df_leaves = df_all[df_all["type"] == "Leave"]

        tabs_hr = st.tabs(["Performance Clusters", "Leave Tracker", "Summary & Export"])

        with tabs_hr[0]:
            st.subheader("Performance Clustering")
            if df_tasks.empty:
                st.info("No task data.")
            else:
                df_pred = predict_for_tasks(df_tasks)
                # employee-level aggregation
                emp_stats = df_pred.groupby("employee").agg({
                    "completion": "mean",
                    "marks": "mean",
                    "predicted_marks": "mean"
                }).reset_index().rename(columns={"completion":"avg_completion","marks":"avg_marks","predicted_marks":"avg_predicted_marks"})
                st.dataframe(emp_stats.fillna(""), use_container_width=True)
                # safe kmeans: if <2 rows, skip fit and show message
                if len(df_pred) >= 2:
                    try:
                        X = df_pred[["completion","marks"]].fillna(0)
                        km = KMeans(n_clusters=min(3, len(X)), random_state=42, n_init=10)
                        df_pred["cluster"] = km.fit_predict(X)
                        st.dataframe(df_pred[["employee","task","completion","marks","cluster"]].fillna(""), use_container_width=True)
                        fig = px.scatter(df_pred, x="completion", y="marks", color="cluster", hover_data=["employee","task"], title="Performance Clusters")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        st.info("Clustering not available for current dataset.")
                else:
                    st.info("Not enough records for K-Means clustering (need 2+).")

        with tabs_hr[1]:
            st.subheader("Leave Tracker")
            if df_leaves.empty:
                st.info("No leave records.")
            else:
                total = len(df_leaves)
                pending = int((df_leaves["status"] == "Pending").sum())
                approved = int((df_leaves["status"] == "Approved").sum())
                rejected = int((df_leaves["status"] == "Rejected").sum())
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Total requests", total)
                c2.metric("Pending", pending)
                c3.metric("Approved", approved)
                c4.metric("Rejected", rejected)
                st.dataframe(df_leaves[["company","employee","leave_type","from","to","reason","status"]].fillna(""), use_container_width=True)

        with tabs_hr[2]:
            st.subheader("Summary & CSV export")
            df_tasks = df_all[df_all["type"] == "Task"]
            if df_tasks.empty:
                st.info("No task records to export.")
            else:
                df_pred = predict_for_tasks(df_tasks)
                # prepare a clean export table
                export_cols = ["_id","company","employee","department","task","status","completion","marks","predicted_marks","predicted_status","predicted_risk","deadline","created"]
                for c in export_cols:
                    if c not in df_pred.columns:
                        df_pred[c] = ""
                export_df = df_pred[export_cols].fillna("")
                st.dataframe(export_df.head(200), use_container_width=True)
                csv = export_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV (Performance Export)", data=csv, file_name=f"performance_export_{today_str}.csv", mime="text/csv")

# ----------------------------
# End of app
# ----------------------------
