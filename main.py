# ============================================================
# main.py — Enterprise Workforce Intelligence System (Complete Final)
# ============================================================
# Requirements:
# pip install streamlit pinecone-client scikit-learn plotly pandas openpyxl PyPDF2

import streamlit as st
import pandas as pd
import numpy as np
import json
import uuid
import time
from datetime import datetime, date
import plotly.express as px

# ML
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Optional (Pinecone, PyPDF2, HF)
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except Exception:
    PINECONE_AVAILABLE = False

try:
    import PyPDF2
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# ============================================================
# Streamlit UI Config
# ============================================================
st.set_page_config(page_title="Enterprise Workforce Intelligence System", layout="wide")
st.title("Enterprise Workforce Intelligence System")
st.caption("AI-driven workforce analytics • Manager reviews • Client approvals • HR insights")

# ============================================================
# Constants & Keys
# ============================================================
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
HUGGINGFACE_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")
INDEX_NAME = "task"
DIMENSION = 1024

# ============================================================
# Pinecone init (best effort)
# ============================================================
pc, index = None, None
if PINECONE_AVAILABLE and PINECONE_API_KEY:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        existing = [i["name"] for i in pc.list_indexes()]
        if INDEX_NAME not in existing and len(existing) > 0:
            # choose first existing index if default not present
            INDEX = existing[0]
        else:
            INDEX = INDEX_NAME
            if INDEX_NAME not in existing:
                pc.create_index(
                    name=INDEX_NAME,
                    dimension=DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
        index = pc.Index(INDEX_NAME)
        st.success(f"Connected to Pinecone index: {INDEX_NAME}")
    except Exception as e:
        st.warning(f"Pinecone init failed — running local-only. ({e})")
        index = None
else:
    if PINECONE_AVAILABLE:
        st.warning("Pinecone API key missing — running local-only.")
    else:
        st.info("Pinecone not installed — running local-only.")

# ============================================================
# Utility functions (safe meta/upsert/fetch)
# ============================================================
def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def rand_vec():
    # deterministic-ish per second seed
    np.random.seed(int(time.time()) % 10000)
    return np.random.rand(DIMENSION).tolist()

def safe_meta(md):
    """
    Convert any md (series, nested types, numpy types, datetimes) into plain {str: str}.
    This makes metadata Pinecone-safe.
    """
    if isinstance(md, pd.Series):
        md = md.to_dict()
    clean = {}
    for k, v in (md or {}).items():
        try:
            # Convert numpy scalars
            if isinstance(v, (np.generic,)):
                v = v.item()
            # Datetimes -> isoformat
            if isinstance(v, (datetime, date)):
                v = v.isoformat()
            # dict/list -> JSON string
            if isinstance(v, (dict, list)):
                v = json.dumps(v)
            # None or NaN -> empty string
            if v is None or (isinstance(v, float) and np.isnan(v)):
                v = ""
            clean[str(k)] = str(v)
        except Exception:
            clean[str(k)] = str(v)
    return clean

def upsert_data(id_, md):
    """
    Always call this to upsert metadata. If Pinecone isn't available,
    store into session_state LOCAL_DATA for local-only mode.
    """
    id_ = str(id_)
    payload = safe_meta(md)

    if index is None:
        local = st.session_state.setdefault("LOCAL_DATA", {})
        local[id_] = payload
        return True

    try:
        # Upsert vector + metadata
        index.upsert(vectors=[{"id": id_, "values": rand_vec(), "metadata": payload}])
        # small pause for eventual consistency
        time.sleep(0.2)
        return True
    except Exception as e:
        st.error(f"Pinecone upsert failed: {e}")
        return False

def fetch_all():
    """
    Fetch all items either from Pinecone or from local storage.
    Returns pandas.DataFrame.
    """
    if index is None:
        local = st.session_state.get("LOCAL_DATA", {})
        if not local:
            return pd.DataFrame()
        # local metadata are strings already
        rows = []
        for k, md in local.items():
            rec = dict(md)
            rec["_id"] = k
            rows.append(rec)
        return pd.DataFrame(rows)
    try:
        res = index.query(vector=rand_vec(), top_k=1000, include_metadata=True)
        rows = []
        # handle both dict format and attribute-like
        matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", [])
        for m in matches:
            md = m.get("metadata", {}) if isinstance(m, dict) else getattr(m, "metadata", {})
            rec = dict(md)
            rec["_id"] = m.get("id", "") if isinstance(m, dict) else getattr(m, "id", "")
            rows.append(rec)
        return pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"Fetch error (falling back to empty): {e}")
        return pd.DataFrame()

# ============================================================
# File text extraction helper
# ============================================================
def extract_file_text(fobj):
    try:
        name = fobj.name.lower()
        if name.endswith(".txt"):
            return fobj.read().decode("utf-8", errors="ignore")[:20000]
        if name.endswith(".pdf") and PDF_AVAILABLE:
            reader = PyPDF2.PdfReader(fobj)
            return "\n".join([p.extract_text() or "" for p in reader.pages])[:20000]
        if name.endswith(".csv"):
            return pd.read_csv(fobj).to_csv(index=False)[:20000]
        if name.endswith(".xlsx"):
            return pd.read_excel(fobj).to_csv(index=False)[:20000]
    except Exception:
        return ""
    return ""

# ============================================================
# ML models & helpers (pretrained/simple samples)
# ============================================================
# Linear regression for marks mapping from completion -> marks (0..5)
lin_reg = LinearRegression().fit([[0], [50], [100]], [0, 2.5, 5])

# Logistic regression simple classifier for On Track vs Delayed
# trained on simple synthetic points; if real labels available, retrain
log_reg = LogisticRegression()
try:
    log_reg.fit([[0], [50], [100]], [0, 1, 1])
except Exception:
    # if training fails, fallback to a trivial mapping by completion
    log_reg = None

# Random forest used in other places (optional)
try:
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
except Exception:
    rf = None

# Sentiment (TF-IDF + SVM) — small demo training
vectorizer = TfidfVectorizer()
svm_clf = SVC()
try:
    sample_texts = ["good work", "excellent job", "poor work", "needs improvement"]
    sample_labels = [1, 1, 0, 0]
    X_train = vectorizer.fit_transform(sample_texts)
    svm_clf.fit(X_train, sample_labels)
except Exception:
    # if training breaks, keep vectorizer but fallback behavior will be handled later
    pass

# Helper to compute predictions for a tasks DataFrame
def predict_ai(df_tasks):
    if df_tasks.empty:
        return df_tasks
    df = df_tasks.copy()
    # normalize columns
    if "completion" in df.columns:
        df["completion"] = pd.to_numeric(df["completion"], errors="coerce").fillna(0)
    else:
        df["completion"] = 0
    if "marks" not in df.columns:
        df["marks"] = 0
    else:
        df["marks"] = pd.to_numeric(df["marks"], errors="coerce").fillna(0)

    # predicted marks (linear mapping)
    df["predicted_marks"] = df["completion"].apply(lambda x: round(float(lin_reg.predict([[x]])[0]), 2))

    # predicted status
    try:
        if log_reg:
            df["predicted_status"] = df["completion"].apply(lambda x: "Completed" if x >= 100 else ("On Track" if log_reg.predict([[x]])[0] == 1 else "Delayed"))
        else:
            df["predicted_status"] = df["completion"].apply(lambda x: "Completed" if x >= 100 else ("On Track" if x >= 50 else "Delayed"))
    except Exception:
        df["predicted_status"] = df["completion"].apply(lambda x: "Completed" if x >= 100 else ("On Track" if x >= 50 else "Delayed"))

    # risk: a simple heuristic / random forest fallback
    try:
        df["days_left"] = np.random.randint(0, 10, len(df))
        if rf and len(df) > 1:
            y_risk = np.where((df["completion"] < 50) & (df["days_left"] < 3), 1, 0)
            rf.fit(df[["completion", "days_left"]], y_risk)
            preds = rf.predict(df[["completion", "days_left"]])
            df["predicted_risk"] = ["At Risk" if p == 1 else "On Track" for p in preds]
        else:
            df["predicted_risk"] = df["completion"].apply(lambda x: "On Track" if x >= 50 else "At Risk")
    except Exception:
        df["predicted_risk"] = df["completion"].apply(lambda x: "On Track" if x >= 50 else "At Risk")

    return df

# ============================================================
# Role selector (UI)
# ============================================================
role = st.sidebar.selectbox("Access Portal As", ["Manager", "Team Member", "Client", "HR Administrator"])

# ============================================================
# MANAGER: Task Management, Feedback (bulk), Leave Management, Meeting Management, Overview
# ============================================================
if role == "Manager":
    st.header("Manager Dashboard — Tasks · Reviews · Leaves · Meetings")
    tabs = st.tabs(["Task Management", "Feedback", "Leave Management", "Meeting Management", "Overview"])

    # -------------------------
    # Task Management
    # -------------------------
    with tabs[0]:
        st.subheader("Assign New Task")
        with st.form("assign_task_form"):
            company = st.text_input("Company Name")
            department = st.text_input("Department")
            employee = st.text_input("Employee Name")
            task_title = st.text_input("Task Title")
            description = st.text_area("Description")
            deadline = st.date_input("Deadline", value=date.today())
            attach = st.file_uploader("Optional Attachment (txt/pdf/csv/xlsx)", type=["txt", "pdf", "csv", "xlsx"])
            submit = st.form_submit_button("Assign Task")
            if submit:
                if not employee or not task_title:
                    st.warning("Employee and Task Title are required.")
                else:
                    tid = str(uuid.uuid4())
                    md = {
                        "type": "Task",
                        "company": company or "",
                        "department": department or "",
                        "employee": employee,
                        "task": task_title,
                        "description": description or "",
                        "deadline": str(deadline),
                        "completion": 0,
                        "marks": 0,
                        "status": "Assigned",
                        "created": now()
                    }
                    if attach:
                        md["attachment"] = extract_file_text(attach)
                    upsert_data(tid, md)
                    st.success(f"Assigned '{task_title}' to {employee}")

        # show tasks table with AI predictions
        df_all = fetch_all()
        tasks = pd.DataFrame()
        if not df_all.empty:
            # normalize type values safely
            df_all["type"] = df_all.get("type", "").astype(str).str.replace('"', '').str.strip()
            tasks = df_all[df_all["type"].str.lower() == "task"]
        if tasks.empty:
            st.info("No tasks available.")
        else:
            pred = predict_ai(tasks)
            # dedupe employee-task for display
            pred = pred.drop_duplicates(subset=["employee", "task"])
            st.dataframe(pred[["company", "employee", "department", "task", "completion", "predicted_marks", "predicted_status", "predicted_risk"]], width='stretch')

    # -------------------------
    # Manager Feedback (bulk)
    # -------------------------
    with tabs[1]:
        st.subheader("Manager Feedback & Evaluation (bulk)")

        df_all = fetch_all()
        if df_all.empty or "type" not in df_all.columns:
            st.info("No data available.")
        else:
            # normalize and prepare
            df_all["type"] = df_all["type"].astype(str).str.replace('"', '').str.strip()
            df_all["completion"] = pd.to_numeric(df_all.get("completion", 0), errors="coerce").fillna(0)
            df_all.loc[df_all["completion"] >= 100, "status"] = "Completed"
            df_all.loc[(df_all["completion"] >= 75) & (df_all["completion"] < 100), "status"] = "Ready for Review"

            df_tasks = df_all[df_all["type"].str.lower() == "task"].copy()

            # search filter
            search_query = st.text_input("Search by Employee or Task (optional)", placeholder="e.g., Priya or Chatbot Design").strip().lower()
            if search_query:
                df_tasks = df_tasks[
                    df_tasks["employee"].astype(str).str.lower().str.contains(search_query, na=False)
                    | df_tasks["task"].astype(str).str.lower().str.contains(search_query, na=False)
                ]

            # dedupe
            df_tasks = df_tasks.drop_duplicates(subset=["employee", "task"])

            # only show tasks >= 75%
            pending = df_tasks[df_tasks["completion"] >= 75].to_dict("records")

            if not pending:
                st.success("No pending tasks for review (>=75%).")
            else:
                st.markdown("**Adjust completion and add comments for each task, then Save All Reviews**")
                with st.form(key="manager_review_form"):
                    for i, rec in enumerate(pending):
                        emp = rec.get("employee", "?")
                        tname = rec.get("task", "?")
                        comp = float(rec.get("completion", 0))
                        st.write(f"👤 **{emp}** | Task: **{tname}**")
                        st.slider(f"Completion ({emp} - {tname})", 0, 100, int(comp), key=f"comp_{i}")
                        st.text_area(f"Manager Comments ({tname})", key=f"comm_{i}")

                    submit = st.form_submit_button("💾 Save All Reviews")
                    if submit:
                        for i, rec in enumerate(pending):
                            rec_id = str(rec.get("_id", uuid.uuid4()))
                            completion = st.session_state.get(f"comp_{i}", int(rec.get("completion", 0)))
                            comments = st.session_state.get(f"comm_{i}", "").strip()
                            predicted_marks = float(lin_reg.predict([[completion]])[0])
                            # status prediction (log reg fallback to heuristic)
                            try:
                                if log_reg:
                                    status_val = log_reg.predict([[completion]])[0]
                                    status_text = "On Track" if status_val == 1 else "Delayed"
                                else:
                                    status_text = "On Track" if completion >= 50 else "Delayed"
                            except Exception:
                                status_text = "On Track" if completion >= 50 else "Delayed"

                            sentiment_text = "N/A"
                            if comments:
                                try:
                                    X_new = vectorizer.transform([comments])
                                    sentiment = svm_clf.predict(X_new)[0]
                                    sentiment_text = "Positive" if sentiment == 1 else "Negative"
                                except Exception:
                                    sentiment_text = "N/A"

                            updated = {
                                **rec,
                                "completion": float(completion),
                                "marks": predicted_marks,
                                "status": status_text,
                                "manager_feedback": comments,
                                "sentiment": sentiment_text,
                                "manager_reviewed_on": now()
                            }
                            upsert_data(rec_id, updated)

                        st.success("✅ All reviews saved to Pinecone!")
                        st.balloons()

    # -------------------------
    # Leave Management (search by employee)
    # -------------------------
    with tabs[2]:
        st.subheader("Leave Management")
        df_all = fetch_all()
        leaves = pd.DataFrame()
        if not df_all.empty:
            df_all["type"] = df_all.get("type", "").astype(str).str.replace('"', '').str.strip()
            leaves = df_all[df_all["type"].str.lower() == "leave"]
        search_emp = st.text_input("Search leave by employee name (optional)").strip().lower()
        if not leaves.empty:
            if search_emp:
                leaves = leaves[leaves["employee"].astype(str).str.lower().str.contains(search_emp, na=False)]
            pending = leaves[leaves["status"].astype(str).str.lower() == "pending"]
            if not pending.empty:
                st.markdown("### Pending Requests")
                for i, row in pending.iterrows():
                    emp = row.get("employee", "Unknown")
                    st.markdown(f"**{emp}** → {row.get('from')} to {row.get('to')}")
                    decision = st.radio(f"Decision for {emp}", ["Approve", "Reject"], key=f"lv_dec_{i}")
                    if st.button(f"Submit Decision for {emp}", key=f"lv_btn_{i}"):
                        row["status"] = "Approved" if decision == "Approve" else "Rejected"
                        row["approved_on"] = now()
                        upsert_data(row.get("_id", str(uuid.uuid4())), row)
                        st.success(f"Leave {row['status']} for {emp}")
                        st.experimental_rerun()
            st.markdown("### All Leaves")
            st.dataframe(leaves[["employee", "leave_type", "from", "to", "reason", "status"]].fillna(""), width='stretch')
        else:
            st.info("No leave records found.")

    # -------------------------
    # Meeting Management
    # -------------------------
    with tabs[3]:
        st.subheader("Meeting Management")
        with st.form("schedule_meeting"):
            mt_title = st.text_input("Meeting Title")
            mt_company = st.text_input("Company (optional)")
            mt_date = st.date_input("Meeting Date", value=date.today())
            mt_time = st.text_input("Meeting Time", value="10:00 AM")
            mt_att = st.text_area("Attendees (comma-separated)")
            mt_notes = st.file_uploader("Upload notes (txt/pdf/csv/xlsx)", type=["txt", "pdf", "csv", "xlsx"])
            submit_meet = st.form_submit_button("Schedule Meeting")
            if submit_meet:
                mid = str(uuid.uuid4())
                attendees = [a.strip().lower() for a in mt_att.split(",") if a.strip()]
                md = {
                    "type": "Meeting",
                    "meeting_title": mt_title,
                    "company": mt_company or "",
                    "meeting_date": str(mt_date),
                    "meeting_time": mt_time or "",
                    "attendees": json.dumps(attendees),
                    "created": now()
                }
                if mt_notes:
                    md["notes_file"] = mt_notes.name
                    md["notes_text"] = extract_file_text(mt_notes)
                upsert_data(mid, md)
                st.success(f"Meeting '{mt_title}' scheduled.")

        st.markdown("---")
        df_all = fetch_all()
        meetings = pd.DataFrame()
        if not df_all.empty:
            df_all["type"] = df_all.get("type", "").astype(str).str.replace('"', '').str.strip()
            meetings = df_all[df_all["type"].str.lower() == "meeting"]
        if meetings.empty:
            st.info("No meetings recorded.")
        else:
            search_meet = st.text_input("Search meetings by attendee or company (optional)").strip().lower()
            display_meet = meetings.copy()
            if search_meet:
                def match_row(row):
                    try:
                        attendees = json.loads(row.get("attendees", "[]"))
                    except Exception:
                        attendees = []
                    return (search_meet in str(row.get("company", "")).lower()) or any(search_meet in a for a in attendees)
                display_meet = display_meet[display_meet.apply(match_row, axis=1)]
            st.dataframe(display_meet[["meeting_title", "company", "meeting_date", "meeting_time", "attendees"]].fillna(""), width='stretch')

            sel = st.selectbox("Select meeting", display_meet["meeting_title"].unique())
            if sel:
                row = display_meet[display_meet["meeting_title"] == sel].iloc[0]
                notes_text = row.get("notes_text", "")
                st.text_area("Notes", notes_text, height=200)
                if st.button("Summarize notes (HuggingFace)"):
                    if HF_AVAILABLE and HUGGINGFACE_TOKEN:
                        try:
                            client = InferenceClient(token=HUGGINGFACE_TOKEN)
                            prompt = f"Summarize the meeting notes and extract action items:\n\n{notes_text[:4000]}"
                            res = client.text_generation(model="mistralai/Mixtral-8x7B-Instruct", inputs=prompt, max_new_tokens=250)
                            # res could be dict or list
                            if isinstance(res, dict):
                                out = res.get("generated_text") or res.get("output") or str(res)
                            elif isinstance(res, list) and res and isinstance(res[0], dict):
                                out = res[0].get("generated_text") or str(res[0])
                            else:
                                out = str(res)
                            st.subheader("AI Summary")
                            st.write(out)
                        except Exception as e:
                            st.error(f"AI summarization failed: {e}")
                    else:
                        st.warning("HuggingFace token not configured or Inference client unavailable.")

    # -------------------------
    # Overview
    # -------------------------
    with tabs[4]:
        st.subheader("Team Overview & Exports")
        df_all = fetch_all()
        if df_all.empty:
            st.info("No data.")
        else:
            df_all["type"] = df_all.get("type", "").astype(str).str.replace('"', '').str.strip()
            tasks = df_all[df_all["type"].str.lower() == "task"]
            if not tasks.empty:
                tasks["completion"] = pd.to_numeric(tasks.get("completion", 0), errors="coerce").fillna(0)
                pred = predict_ai(tasks)
                pred = pred.drop_duplicates(subset=["employee", "task"])
                st.dataframe(pred[["company", "employee", "department", "task", "completion", "predicted_marks", "predicted_status", "predicted_risk"]], width='stretch')
                csv = pred.to_csv(index=False).encode("utf-8")
                st.download_button("Download Tasks CSV", data=csv, file_name="tasks_export.csv", mime="text/csv")
            else:
                st.info("No task data available.")

# ============================================================
# TEAM MEMBER
# ============================================================
elif role == "Team Member":
    st.header("Team Member Dashboard — Update Progress & Apply Leave")
    name = st.text_input("Enter your name")
    if name:
        df_all = fetch_all()
        if df_all.empty:
            st.info("No data available.")
        else:
            df_all["type"] = df_all.get("type", "").astype(str).str.replace('"', '').str.strip()
            my_tasks = df_all[(df_all["type"].str.lower() == "task") & (df_all["employee"].astype(str).str.lower() == name.lower())]
            if my_tasks.empty:
                st.info("No tasks assigned to you.")
            else:
                for _, r in my_tasks.iterrows():
                    st.markdown(f"**{r.get('task', '')}** — Status: {r.get('status','')}")
                    curr = int(float(r.get("completion", 0) or 0))
                    val = st.slider(f"Progress: {r.get('task')}", 0, 100, curr, key=r.get("_id"))
                    attach = st.file_uploader(f"Optional upload for {r.get('task')}", type=["txt", "pdf", "csv", "xlsx"], key=f"file_{r.get('_id')}")
                    if st.button(f"Update {r.get('task')}", key=f"upd_{r.get('_id')}"):
                        r["completion"] = int(val)
                        r["marks"] = float(lin_reg.predict([[val]])[0])
                        r["status"] = "In Progress" if val < 100 else "Completed"
                        if attach:
                            r["attachment"] = extract_file_text(attach)
                        upsert_data(r.get("_id", str(uuid.uuid4())), r)
                        st.success("Updated successfully.")
                        st.experimental_rerun()

            st.markdown("---")
            st.subheader("Apply for Leave")
            lt = st.selectbox("Leave Type", ["Casual", "Sick", "Earned"])
            f = st.date_input("From")
            t = st.date_input("To")
            reason = st.text_area("Reason")
            if st.button("Submit Leave Request"):
                lid = str(uuid.uuid4())
                md = {"type": "Leave", "employee": name, "leave_type": lt, "from": str(f), "to": str(t), "reason": reason, "status": "Pending", "submitted": now()}
                upsert_data(lid, md)
                st.success("Leave requested.")

# ============================================================
# CLIENT
# ============================================================
elif role == "Client":
    st.header("Client Portal — Review & Feedback")
    company = st.text_input("Enter Company Name")
    if company:
        df_all = fetch_all()
        if df_all.empty:
            st.info("No data.")
        else:
            df_all["type"] = df_all.get("type", "").astype(str).str.replace('"', '').str.strip()
            df_client = df_all[df_all["company"].astype(str).str.lower() == company.lower()]
            if df_client.empty:
                st.info("No records for this company.")
            else:
                tasks = df_client[df_client["type"].str.lower() == "task"].copy()
                if not tasks.empty:
                    st.dataframe(tasks[["employee", "task", "completion", "marks", "status"]].fillna(""), width='stretch')
                else:
                    st.info("No tasks for this company.")

                st.markdown("---")
                st.subheader("Review Completed / Ready Tasks")
                reviewable = df_client[(df_client["type"] == "Task") & ((pd.to_numeric(df_client.get("completion", 0), errors="coerce") >= 75) | (df_client.get("status") == "Under Client Review"))]
                if reviewable.empty:
                    st.info("No tasks ready for review.")
                else:
                    sel = st.selectbox("Select Task to Review", reviewable["task"].unique())
                    if sel:
                        fb = st.text_area("Client Feedback")
                        rating = st.slider("Rating (1–5)", 1, 5, 4)
                        if st.button("Submit Client Feedback"):
                            rec = reviewable[reviewable["task"] == sel].iloc[0].to_dict()
                            rec["client_feedback"] = fb
                            rec["client_rating"] = rating
                            rec["client_reviewed_on"] = now()
                            rec["status"] = "Client Approved"
                            upsert_data(rec.get("_id", str(uuid.uuid4())), rec)
                            st.success("Client feedback saved.")

# ============================================================
# HR ADMIN
# ============================================================
elif role == "HR Administrator":
    st.header("HR Dashboard — Analytics & Export")
    df_all = fetch_all()
    if df_all.empty:
        st.info("No data.")
    else:
        df_all["type"] = df_all.get("type", "").astype(str).str.replace('"', '').str.strip()
        tasks = df_all[df_all["type"].str.lower() == "task"].copy()
        leaves = df_all[df_all["type"].str.lower() == "leave"].copy()

        tabs = st.tabs(["Performance Clusters", "Leave Tracker", "Export"])
        with tabs[0]:
            if tasks.empty:
                st.info("Not enough task data.")
            else:
                tasks["completion"] = pd.to_numeric(tasks.get("completion", 0), errors="coerce").fillna(0)
                if len(tasks) >= 2:
                    try:
                        km = KMeans(n_clusters=min(3, len(tasks)), random_state=42, n_init=10)
                        tasks["cluster"] = km.fit_predict(tasks[["completion", "marks"]].fillna(0))
                    except Exception:
                        tasks["cluster"] = 0
                else:
                    tasks["cluster"] = 0
                st.dataframe(tasks[["employee", "task", "completion", "marks", "cluster"]].fillna(""), width='stretch')

        with tabs[1]:
            if leaves.empty:
                st.info("No leave data.")
            else:
                st.dataframe(leaves[["employee", "leave_type", "from", "to", "status"]].fillna(""), width='stretch')

        with tabs[2]:
            # Export tasks csv
            if not tasks.empty:
                csv = tasks.to_csv(index=False).encode("utf-8")
                st.download_button("Download Tasks CSV", data=csv, file_name="hr_tasks.csv", mime="text/csv")
            else:
                st.info("No tasks to export.")

# ============================================================
# End of app
# ============================================================
