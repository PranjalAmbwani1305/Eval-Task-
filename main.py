# ============================================================
# main.py — Enterprise Workforce Performance Intelligence System (Stable + Enhancements)
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

# ----------------------------
# Secrets & constants
# ----------------------------
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
HF_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")
INDEX_NAME = "task"   # must be lowercase alphanumeric or '-'
DIMENSION = 1024

# ----------------------------
# Pinecone init (best-effort)
# ----------------------------
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

# ----------------------------
# Utilities
# ----------------------------
def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def rand_vec():
    return np.random.rand(DIMENSION).tolist()

def safe_meta(md: dict) -> dict:
    """Convert metadata to JSON-friendly primitives."""
    clean = {}
    for k, v in (md or {}).items():
        try:
            if isinstance(v, (datetime, date)):
                clean[k] = v.isoformat()
            elif isinstance(v, (dict, list)):
                # store as JSON string for safe retrieval
                clean[k] = json.dumps(v)
            elif pd.isna(v):
                clean[k] = ""
            else:
                clean[k] = v
        except Exception:
            clean[k] = str(v)
    return clean

def upsert_data(id_, md: dict) -> bool:
    """Upsert record to Pinecone or session-local storage if Pinecone not available.
       Accepts md as dict (safe to pass Series.to_dict())."""
    id_ = str(id_)
    md = dict(md)
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
    """Fetch all metadata records from Pinecone (or local storage). Returns DataFrame with _id column."""
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
        # try describe stats to avoid querying empty index (best-effort)
        try:
            stats = index.describe_index_stats()
            if stats and stats.get("total_vector_count", 0) == 0:
                return pd.DataFrame()
        except Exception:
            pass
        res = index.query(vector=rand_vec(), top_k=1000, include_metadata=True)
        matches = getattr(res, "matches", None) or res.get("matches", [])
        if not matches:
            return pd.DataFrame()
        rows = []
        for m in matches:
            # m may be object or dict depending on client
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
        st.warning(f"Fetch error: {e}")
        return pd.DataFrame()

# ----------------------------
# Hugging Face safe wrapper
# ----------------------------
def hf_text_generation(prompt: str, model: str = "mistralai/Mixtral-8x7B-Instruct", max_new_tokens: int = 200):
    if not HF_AVAILABLE or not HF_TOKEN:
        raise RuntimeError("Hugging Face client or token not available")
    client = InferenceClient(token=HF_TOKEN)
    try:
        # some client versions expect 'inputs', others 'prompt'
        res = client.text_generation(model=model, inputs=prompt, max_new_tokens=max_new_tokens)
    except TypeError:
        res = client.text_generation(model=model, prompt=prompt, max_new_tokens=max_new_tokens)
    # extract text safely
    if isinstance(res, dict):
        return res.get("generated_text") or res.get("output") or json.dumps(res)
    if isinstance(res, list) and res and isinstance(res[0], dict):
        return res[0].get("generated_text") or json.dumps(res[0])
    return str(res)

# ----------------------------
# Small ML helper for marks
# ----------------------------
lin_reg = LinearRegression().fit([[0], [50], [100]], [0, 2.5, 5])

# ----------------------------
# Helper: attendees parse (robust)
# ----------------------------
def parse_attendees_field(val):
    """
    Accepts:
      - list of strings
      - JSON string of list (double or single quotes)
      - raw comma-separated string
    Returns list of lowercased trimmed names.
    """
    if isinstance(val, list):
        return [a.strip().lower() for a in val if isinstance(a, str) and a.strip()]
    if isinstance(val, str):
        s = val.strip()
        # try JSON first (normalize single quotes)
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [a.strip().lower() for a in parsed if isinstance(a, str) and a.strip()]
            except Exception:
                # try replace single quotes with double quotes then parse
                try:
                    parsed = json.loads(s.replace("'", '"'))
                    if isinstance(parsed, list):
                        return [a.strip().lower() for a in parsed if isinstance(a, str) and a.strip()]
                except Exception:
                    # fallback to manual split
                    s2 = s.strip("[]").replace("'", "")
                    return [a.strip().lower() for a in s2.split(",") if a.strip()]
        # else treat as comma-separated
        return [a.strip().lower() for a in s.split(",") if a.strip()]
    return []

# ----------------------------
# Role selection
# ----------------------------
role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client", "HR (Admin)"])
current_month = datetime.now().strftime("%B %Y")

# ----------------------------
# MANAGER
# ----------------------------
if role == "Manager":
    st.header("Manager Dashboard")
    tabs = st.tabs(["Task Management", "Feedback & Review", "Meetings", "Leave Approvals", "Team Overview", "AI Insights"])

    # --- Task Management (assign + reassign) ----
    with tabs[0]:
        st.subheader("Assign Task")
        with st.form("assign_task"):
            company = st.text_input("Client Company Name")
            department = st.text_input("Department")
            employee = st.text_input("Employee Name")
            task_title = st.text_input("Task Title")
            description = st.text_area("Description")
            deadline = st.date_input("Deadline", value=date.today())
            submit = st.form_submit_button("Assign Task")
            if submit:
                if not employee or not task_title:
                    st.warning("Employee and Task Title required.")
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
                    upsert_data(tid, md)
                    st.success(f"Assigned '{task_title}' to {employee} for {company or '—'}")

        # quick tasks table and reassign block
        df_all = fetch_all()
        df_tasks = pd.DataFrame()
        if not df_all.empty:
            if "type" in df_all.columns:
                df_tasks = df_all[df_all["type"] == "Task"]
            else:
                df_tasks = df_all[df_all.get("task", "").astype(str) != ""]
        if df_tasks.empty:
            st.info("No tasks found.")
        else:
            # ensure columns
            for col in ["company", "employee", "department", "task", "status", "completion", "deadline", "created"]:
                if col not in df_tasks.columns:
                    df_tasks[col] = ""
            # ensure numeric
            df_tasks["completion"] = pd.to_numeric(df_tasks["completion"], errors="coerce").fillna(0)
            df_tasks["marks"] = pd.to_numeric(df_tasks.get("marks", 0), errors="coerce").fillna(0)
            st.dataframe(df_tasks[["company","employee","department","task","status","completion","deadline"]], use_container_width=True)

            st.markdown("---")
            st.subheader("Reassign Task")
            task_choice = st.selectbox("Select Task", df_tasks["task"].dropna().unique())
            new_emp = st.text_input("New Employee Name")
            reas = st.text_area("Reason for Reassignment")
            if st.button("Reassign Task"):
                rec = df_tasks[df_tasks["task"] == task_choice].iloc[0].to_dict()
                rec["employee"] = new_emp or rec.get("employee", "")
                rec["status"] = "Reassigned"
                rec["reassigned_reason"] = reas
                rec["reassigned_on"] = now()
                upsert_data(rec.get("_id") or str(uuid.uuid4()), rec)
                st.success("Task reassigned.")

    # --- Manager Feedback & marks (before client) ---
    with tabs[1]:
        st.subheader("Manager Feedback & Marks Review")
        df_all = fetch_all()
        if df_all.empty:
            st.info("No data available.")
        else:
            # identify completed tasks (robust)
            df_tasks = df_all[df_all.get("type") == "Task"] if "type" in df_all.columns else df_all[df_all.get("task", "").astype(str) != ""]
            if df_tasks.empty:
                st.info("No tasks found.")
            else:
                # normalize numeric
                df_tasks["completion"] = pd.to_numeric(df_tasks.get("completion", 0), errors="coerce").fillna(0)
                df_tasks["marks"] = pd.to_numeric(df_tasks.get("marks", 0), errors="coerce").fillna(0)

                pending_review = df_tasks[(df_tasks["completion"] >= 100) & (df_tasks.get("status") != "Under Client Review")]
                if pending_review.empty:
                    st.info("No completed tasks awaiting review.")
                else:
                    # search by employee optional
                    emp_search = st.text_input("Search by employee name (optional)", value="")
                    if emp_search:
                        pending_review = pending_review[pending_review["employee"].astype(str).str.lower().str.contains(emp_search.strip().lower())]

                    # unique tasks list
                    task_list = pending_review["task"].dropna().unique().tolist()
                    if not task_list:
                        st.info("No matching completed tasks.")
                    else:
                        task_sel = st.selectbox("Select completed task", task_list)
                        row = pending_review[pending_review["task"] == task_sel].iloc[0].to_dict()
                        st.write(f"Employee: {row.get('employee','')}  |  Company: {row.get('company','')}")
                        st.write(f"Task: {row.get('task','')}")
                        final_marks = st.slider("Final Marks (0–5)", 0.0, 5.0, float(row.get("marks", 0)))
                        manager_feedback = st.text_area("Manager Comments")
                        if st.button("Finalize & Send to Client"):
                            row["marks"] = final_marks
                            row["manager_feedback"] = manager_feedback
                            row["manager_reviewed_on"] = now()
                            row["status"] = "Under Client Review"
                            upsert_data(row.get("_id") or str(uuid.uuid4()), row)
                            st.success("Task reviewed and sent for client evaluation.")

    # --- Meetings (schedule + notes + attendees) ---
    with tabs[2]:
        st.subheader("Meeting Scheduler")
        with st.form("schedule_meeting"):
            meeting_title = st.text_input("Meeting Title")
            meeting_company = st.text_input("Client Company (optional)")
            meeting_date = st.date_input("Meeting Date", value=date.today())
            meeting_time = st.text_input("Meeting Time (HH:MM)", value="10:00")
            attendees_raw = st.text_area("Attendees (comma-separated names)")
            notes_file = st.file_uploader("Upload notes (txt/pdf/csv/xlsx)", type=["txt","pdf","csv","xlsx"])
            submit_meet = st.form_submit_button("Schedule Meeting")
            if submit_meet:
                mid = str(uuid.uuid4())
                attendees_list = [a.strip().lower() for a in attendees_raw.split(",") if a.strip()]
                md = {
                    "type": "Meeting",
                    "company": meeting_company or "",
                    "meeting_title": meeting_title or "",
                    "meeting_date": str(meeting_date),
                    "meeting_time": meeting_time or "",
                    # store attendees as JSON string for consistency
                    "attendees": json.dumps(attendees_list),
                    "created": now()
                }
                # extract text from uploaded file if any
                if notes_file:
                    fname = notes_file.name.lower()
                    try:
                        if fname.endswith(".txt"):
                            content = notes_file.read().decode("utf-8", errors="ignore")
                        elif fname.endswith(".pdf") and PDF_AVAILABLE:
                            reader = PyPDF2.PdfReader(notes_file)
                            content = "\n".join([p.extract_text() or "" for p in reader.pages])
                        elif fname.endswith(".csv"):
                            content = pd.read_csv(notes_file).to_csv(index=False)
                        elif fname.endswith(".xlsx"):
                            content = pd.read_excel(notes_file).to_csv(index=False)
                        else:
                            content = ""
                        md["notes_file"] = notes_file.name
                        md["notes_text"] = content[:20000]
                    except Exception as e:
                        md["notes_text"] = f"ERROR reading file: {e}"
                upsert_data(mid, md)
                st.success("Meeting scheduled.")

        st.markdown("---")
        st.subheader("Upcoming Meetings")
        df_all = fetch_all()
        meets = pd.DataFrame()
        if not df_all.empty:
            meets = df_all[df_all.get("type") == "Meeting"]
        if meets.empty:
            st.info("No meetings scheduled.")
        else:
            # ensure cols
            for c in ["meeting_title","company","meeting_date","meeting_time","attendees","notes_file"]:
                if c not in meets.columns:
                    meets[c] = ""
            # show basic meeting list
            display_meets = meets[["company","meeting_title","meeting_date","meeting_time","attendees","notes_file"]].fillna("")
            st.dataframe(display_meets, use_container_width=True)
            # allow manager to pick a meeting and summarize via HF
            sel = st.selectbox("Select meeting to view/summarize", meets["meeting_title"].dropna().unique())
            if sel:
                mrow = meets[meets["meeting_title"] == sel].iloc[0].to_dict()
                notes = mrow.get("notes_text","")
                st.text_area("Notes preview", notes, height=200)
                if st.button("AI Summarize Meeting Notes"):
                    if HF_AVAILABLE and HF_TOKEN:
                        try:
                            prompt = f"Summarize and extract actions from these meeting notes:\n\n{notes[:4000]}"
                            out = hf_text_generation(prompt=prompt, max_new_tokens=250)
                            st.subheader("AI Summary")
                            st.write(out)
                        except Exception as e:
                            st.error(f"AI summarization failed: {e}")
                    else:
                        st.warning("Hugging Face not configured.")

    # --- Leave approvals ---
    with tabs[3]:
        st.subheader("Leave Approvals")
        df_all = fetch_all()
        leaves = pd.DataFrame()
        if not df_all.empty:
            # detect leave rows robustly
            if "type" in df_all.columns:
                leaves = df_all[df_all["type"] == "Leave"]
            else:
                # fallback: rows with leave_type field or missing task
                leaves = df_all[df_all.get("leave_type").notna() | (df_all.get("task","") == "")]
        if leaves.empty:
            st.info("No leave requests.")
        else:
            # normalize status column
            if "status" not in leaves.columns:
                leaves["status"] = ""
            # work only on pending
            pending = leaves[leaves["status"].astype(str).str.strip().str.lower() == "pending"]
            if pending.empty:
                st.info("No pending leave requests.")
            else:
                for i, row in pending.iterrows():
                    emp = row.get("employee", "Unknown")
                    st.markdown(f"**{emp}** — {row.get('leave_type','Leave')} ({row.get('from','')} → {row.get('to','')})")
                    st.write(f"Reason: {row.get('reason','-')}")
                    decision = st.radio(f"Decision for {emp}", ["Approve","Reject"], key=f"lv_dec_{i}")
                    remarks = st.text_area(f"Manager Remarks for {emp}", key=f"lv_rem_{i}")
                    if st.button(f"Finalize decision for {emp}", key=f"lv_btn_{i}"):
                        updated = row.to_dict()
                        updated["_id"] = str(row.get("_id") or uuid.uuid4())
                        updated["status"] = "Approved" if decision == "Approve" else "Rejected"
                        updated["approved_by"] = "Manager"
                        updated["approved_on"] = now()
                        updated["manager_remarks"] = remarks
                        # append audit
                        existing_audit = updated.get("audit_log","")
                        audit_entry = f"{updated['status']} by Manager on {now()} | Remarks: {remarks}"
                        updated["audit_log"] = (existing_audit + "\n" + audit_entry) if existing_audit else audit_entry
                        success = upsert_data(updated["_id"], updated)
                        if success:
                            st.success(f"Leave {updated['status']} for {emp}")
                            time.sleep(0.8)
                            st.experimental_rerun()
                        else:
                            st.error("Failed to update leave. Try again.")

    # --- Team Overview + AI insights quick ---
    with tabs[4]:
        st.subheader("Team Overview")
        df_all = fetch_all()
        df_tasks = pd.DataFrame()
        if not df_all.empty:
            df_tasks = df_all[df_all.get("type") == "Task"]
        if df_tasks.empty:
            st.info("No tasks.")
        else:
            # normalize
            for col in ["company","department","employee","task","completion","marks","status","created"]:
                if col not in df_tasks.columns:
                    df_tasks[col] = ""
            df_tasks["completion"] = pd.to_numeric(df_tasks["completion"], errors="coerce").fillna(0)
            df_tasks["marks"] = pd.to_numeric(df_tasks.get("marks", 0), errors="coerce").fillna(0)
            st.dataframe(df_tasks[["company","department","employee","task","completion","status"]], use_container_width=True)
            # chart
            try:
                fig = None
                if "department" in df_tasks.columns and df_tasks["department"].notna().any():
                    fig = px.bar(df_tasks, x="employee", y="completion", color="department", title="Completion by Employee")
                else:
                    fig = px.bar(df_tasks, x="employee", y="completion", title="Completion by Employee")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

    # --- AI Insights tab (expanded) ---
    with tabs[5]:
        st.subheader("AI Insights")
        df_all = fetch_all()
        if df_all.empty:
            st.info("No data available.")
        else:
            q = st.text_input("Ask AI (e.g., 'Who is underperforming this week?')")
            if st.button("Generate Insight"):
                if HF_AVAILABLE and HF_TOKEN:
                    try:
                        summary = df_all.describe(include="all").to_dict()
                        prompt = f"You are an analyst. Given dataset summary:\n{summary}\nQuestion: {q}\nAnswer with concise, numbered insights and suggested actions."
                        out = hf_text_generation(prompt=prompt, max_new_tokens=200)
                        st.write(out)
                    except Exception as e:
                        st.error(f"AI query failed: {e}")
                else:
                    st.warning("Hugging Face not configured.")

# ----------------------------
# TEAM MEMBER
# ----------------------------
elif role == "Team Member":
    st.header("Team Member Dashboard")
    name = st.text_input("Enter your name")
    company = st.text_input("Enter your company name (if applicable)")

    if name:
        df_all = fetch_all()
        if df_all.empty:
            st.info("No data present.")
        else:
            # ensure 'type' field exists
            if "type" not in df_all.columns:
                df_all["type"] = df_all.apply(lambda r: "Task" if pd.notna(r.get("task")) and r.get("task") != "" else "Leave", axis=1)

            # filter tasks by name and optional company
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
                    # show manager and client feedback status
                    mfb = r.get("manager_feedback", "")
                    cfb = r.get("client_feedback", "")
                    cr = r.get("client_rating", "")
                    if mfb:
                        st.info(f"Manager Feedback: {mfb}")
                    if cfb:
                        st.success(f"Client Feedback: {cfb} (Rating: {cr})")

                    try:
                        curr = int(float(r.get("completion", 0) or 0))
                    except Exception:
                        curr = 0
                    comp = st.slider("Completion %", 0, 100, curr, key=r.get("_id"))
                    if st.button(f"Update {r.get('_id')}", key=f"upd_{r.get('_id')}"):
                        row = r.to_dict()
                        row["completion"] = comp
                        row["marks"] = float(lin_reg.predict([[comp]])[0])
                        row["status"] = "In Progress" if comp < 100 else "Completed"
                        upsert_data(row.get("_id") or str(uuid.uuid4()), row)
                        st.success("Updated task progress.")

            st.markdown("---")
            st.subheader("My Meetings")
            all_data = df_all
            meets = all_data[all_data.get("type") == "Meeting"] if not all_data.empty else pd.DataFrame()
            if meets.empty:
                st.info("No meetings scheduled.")
            else:
                # filter meetings where attendee list contains name
                def invited_to_meeting(row):
                    attendees_field = row.get("attendees", "")
                    attendees = parse_attendees_field(attendees_field)
                    return name.strip().lower() in attendees
                try:
                    invited_meets = meets[meets.apply(invited_to_meeting, axis=1)]
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
            # Show member's leave requests
            member_leaves = df_all[(df_all.get("type") == "Leave") & (df_all.get("employee","").astype(str).str.lower() == name.lower())]
            if member_leaves.empty:
                st.info("You have no leave requests on record.")
            else:
                # show table of leaves
                display_cols = ["leave_type","from","to","status","approved_by","approved_on","audit_log"]
                for c in display_cols:
                    if c not in member_leaves.columns:
                        member_leaves[c] = ""
                member_leaves["from"] = member_leaves.get("from","")
                member_leaves["to"] = member_leaves.get("to","")
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
    st.header("Client Dashboard")
    company = st.text_input("Enter your company name")
    if company:
        df_all = fetch_all()
        if df_all.empty:
            st.info("No data.")
        else:
            df_client = df_all[df_all.get("company","").astype(str).str.lower() == company.lower()]
            if df_client.empty:
                st.info("No records for this company.")
            else:
                st.subheader("Tasks")
                df_tasks = df_client[df_client.get("type") == "Task"]
                if df_tasks.empty:
                    st.info("No tasks for this company.")
                else:
                    # ensure numeric
                    df_tasks["completion"] = pd.to_numeric(df_tasks.get("completion", 0), errors="coerce").fillna(0)
                    df_tasks["marks"] = pd.to_numeric(df_tasks.get("marks", 0), errors="coerce").fillna(0)
                    st.dataframe(df_tasks[["employee","task","status","completion","marks"]].fillna(""), use_container_width=True)

                    st.markdown("---")
                    st.subheader("Provide feedback for Manager-reviewed tasks")
                    pending = df_tasks[(df_tasks.get("status") == "Under Client Review") & (df_tasks.get("client_reviewed") != True)]
                    if pending.empty:
                        st.info("No tasks pending client review.")
                    else:
                        # allow selection
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
                            # maintain status
                            rec["status"] = "Client Reviewed"
                            upsert_data(rec.get("_id") or str(uuid.uuid4()), rec)
                            st.success("Client feedback submitted.")

                st.markdown("---")
                st.subheader("Company Meetings")
                meets = df_client[df_client.get("type") == "Meeting"]
                if meets.empty:
                    st.info("No meetings for this company.")
                else:
                    # attendees stored as JSON string; parse for display
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
# HR (Admin)
# ----------------------------
elif role == "HR (Admin)":
    st.header("HR Dashboard — Performance Clustering & Leave Tracker")
    df_all = fetch_all()
    if df_all.empty:
        st.info("No records.")
    else:
        # ensure columns exist
        for c in ["company","employee","department","task","completion","marks","type","status","leave_type","from","to"]:
            if c not in df_all.columns:
                df_all[c] = ""
        df_all["completion"] = pd.to_numeric(df_all.get("completion", 0), errors="coerce").fillna(0)
        df_all["marks"] = pd.to_numeric(df_all.get("marks", 0), errors="coerce").fillna(0)

        df_tasks = df_all[df_all["type"] == "Task"]
        df_leaves = df_all[df_all["type"] == "Leave"]

        tabs_hr = st.tabs(["Performance Clustering", "Leave Tracker", "Company Summary"])

        with tabs_hr[0]:
            st.subheader("Performance Clustering (no risk computation)")
            if df_tasks.empty:
                st.info("No task data.")
            else:
                try:
                    data = df_tasks[["completion","marks"]].fillna(0)
                    if len(data) >= 3:
                        km = KMeans(n_clusters=3, random_state=42, n_init=10)
                        df_tasks["cluster"] = km.fit_predict(data)
                        centers = km.cluster_centers_
                        order = np.argsort(centers[:,0] + centers[:,1])
                        label_map = {order[2]:"High Performer", order[1]:"Average", order[0]:"Needs Improvement"}
                        df_tasks["Performance Group"] = df_tasks["cluster"].map(label_map)
                    else:
                        df_tasks["cluster"] = 0
                        df_tasks["Performance Group"] = "Insufficient data"
                    st.dataframe(df_tasks[["company","employee","department","completion","marks","Performance Group"]].fillna(""), use_container_width=True)
                    # scatter
                    try:
                        fig = px.scatter(df_tasks, x="completion", y="marks", color="Performance Group", hover_data=["employee","department"], title="Performance Clusters")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        pass
                    # department average
                    dp = df_tasks.groupby("department")["completion"].mean().reset_index()
                    if not dp.empty:
                        fig2 = px.bar(dp, x="department", y="completion", title="Avg Completion by Department")
                        st.plotly_chart(fig2, use_container_width=True)
                except Exception as e:
                    st.error(f"Clustering failed: {e}")

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
                st.markdown("---")
                st.dataframe(df_leaves[["company","employee","leave_type","from","to","reason","status"]].fillna(""), use_container_width=True)
                # leave per employee
                lc = df_leaves.groupby("employee").size().reset_index(name="Total Leaves")
                if not lc.empty:
                    fig3 = px.bar(lc, x="employee", y="Total Leaves", title="Leave requests per employee")
                    st.plotly_chart(fig3, use_container_width=True)

        with tabs_hr[2]:
            st.subheader("Company Summary")
            companies = df_all["company"].dropna().astype(str).unique().tolist()
            if not companies:
                st.info("No company data.")
            else:
                comp = st.selectbox("Select company", companies)
                comp_df = df_all[df_all.get("company","").astype(str) == comp]
                tasks = comp_df[comp_df["type"] == "Task"]
                meets = comp_df[comp_df["type"] == "Meeting"]
                leaves = comp_df[comp_df["type"] == "Leave"]
                st.metric("Total tasks", len(tasks))
                st.metric("Total meetings", len(meets))
                st.metric("Total leave requests", len(leaves))
                if not tasks.empty:
                    st.markdown("### Tasks (sample)")
                    st.dataframe(tasks[["employee","task","status","completion","marks"]].fillna(""), use_container_width=True)
                if not meets.empty:
                    st.markdown("### Meetings (sample)")
                    st.dataframe(meets[["meeting_title","meeting_date","meeting_time","attendees"]].fillna(""), use_container_width=True)
                if not leaves.empty:
                    st.markdown("### Leaves (sample)")
                    st.dataframe(leaves[["employee","leave_type","from","to","status"]].fillna(""), use_container_width=True)

# ----------------------------
# End of app
# ----------------------------
