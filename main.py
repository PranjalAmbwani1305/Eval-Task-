import streamlit as st
import pandas as pd
import numpy as np
import uuid
import os
from datetime import date, datetime, timedelta
from sklearn.cluster import KMeans
import plotly.express as px
import pathlib


st.title("🏢 AI Enterprise Workforce — Stable Edition (Assign / Upload / Meetings)")

# storage dir for uploaded files
FILE_STORE_DIR = os.path.join(os.getcwd(), "ai_workforce_files")
pathlib.Path(FILE_STORE_DIR).mkdir(parents=True, exist_ok=True)

# ------------- PINECONE (optional) -------------
USE_PINECONE = False
index = None
DIMENSION = 1024

try:
    if "PINECONE_API_KEY" in st.secrets:
        from pinecone import Pinecone
        pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
        index = pc.Index("task")  # if index missing, it may error; we catch below
        USE_PINECONE = True
except Exception:
    # no Pinecone — will use local session store
    USE_PINECONE = False
    index = None

# ------------- SESSION STORAGE INITIALIZATION -------------
if "store_df" not in st.session_state:
    # columns include general task fields; meeting records have record_type == "meeting"
    st.session_state.store_df = pd.DataFrame(
        columns=[
            "_id", "record_type", "company", "department", "employee", "task", "completion",
            "marks", "status", "deadline", "priority", "notes", "attachments",
            "submitted_on", "reviewed", "client_approved_on", "last_reassigned_on",
            # meeting-specific
            "meeting_title", "organizer", "participants", "meeting_datetime",
            "duration_min", "agenda", "attachment", "meeting_submissions"
        ]
    )

# convenience alias
def local_df():
    return st.session_state.store_df

def save_uploaded_file(uploaded_file, prefix="file"):
    if uploaded_file is None:
        return None
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    fname = f"{prefix}_{ts}_{uuid.uuid4().hex}_{uploaded_file.name}"
    out_path = os.path.join(FILE_STORE_DIR, fname)
    try:
        with open(out_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return out_path
    except Exception as e:
        st.error(f"Failed to save file: {e}")
        return None

# ------------- Data API: unified interface to Pinecone or local -------------
def fetch_all():
    """Return pandas DataFrame of all records. Uses Pinecone if configured, else local."""
    if USE_PINECONE and index is not None:
        try:
            res = index.query(vector=np.random.rand(DIMENSION).tolist(), top_k=1000, include_metadata=True)
            rows = []
            for m in res.matches:
                meta = m.metadata or {}
                meta["_id"] = m.id
                rows.append(meta)
            return pd.DataFrame(rows)
        except Exception:
            # fallback to local
            return local_df().copy()
    else:
        return local_df().copy()

def upsert_records(records):
    """
    records: list of dicts. If dict contains '_id' we update that record. Otherwise create.
    This updates local session store and, if Pinecone available, attempts Pinecone upsert.
    """
    df = local_df()
    changed = False
    for rec in records:
        rec = dict(rec)  # copy
        rec_id = rec.get("_id") or str(uuid.uuid4())
        rec["_id"] = rec_id
        # normalize fields: ensure columns exist
        for c in rec.keys():
            if c not in df.columns:
                df[c] = None
        # if exists in local df -> update row; else append
        if (df["_id"] == rec_id).any():
            df.loc[df["_id"] == rec_id, list(rec.keys())] = list(rec.values())
        else:
            df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True, sort=False)
        changed = True

    # clean up DataFrame index
    st.session_state.store_df = df.reset_index(drop=True)

    # attempt Pinecone upsert (best-effort, silent failure)
    if USE_PINECONE and index is not None:
        try:
            batch = []
            for rec in records:
                rec_copy = dict(rec)
                pc_id = rec_copy.pop("_id")
                vector = np.random.rand(DIMENSION).tolist()  # placeholder
                batch.append({"id": pc_id, "values": vector, "metadata": rec_copy})
            index.upsert(batch)
        except Exception:
            pass

    return changed

# initialize local data variable for display
df = fetch_all()

# ------------- Sidebar: role selection -------------
st.sidebar.header("Role")
role = st.sidebar.radio("Choose Role", ["Manager", "Team Member", "Client", "Admin"])

# ------------- Helper utilities -------------
def compute_marks_and_status(completion):
    try:
        comp = float(completion)
    except:
        comp = 0.0
    marks = round(comp * 0.05, 2)
    status = "On Track" if comp >= 60 else ("Delayed" if comp < 30 else "At Risk")
    return marks, status

# ------------- MANAGER UI -------------
if role == "Manager":
    st.header("👨‍💼 Manager Dashboard — Assign / Meetings")

    tabs = st.tabs(["Assign / Reassign", "Review Tasks", "Meetings", "Overview"])

    # ---------- Assign / Reassign ----------
    with tabs[0]:
        st.subheader("📝 Create or Update Task")
        mode = st.radio("Mode", ["Create New Task", "Update / Reassign Existing"], index=0)

        if mode == "Create New Task":
            company = st.text_input("Company")
            department = st.text_input("Department")
            employee = st.text_input("Assign To (Employee)")
            task_title = st.text_input("Task Title")
            completion = st.number_input("Initial Completion %", min_value=0, max_value=100, value=0)
            deadline = st.date_input("Deadline", value=date.today() + timedelta(days=7))
            priority = st.selectbox("Priority", ["Low", "Medium", "High"])
            file_upload = st.file_uploader("Optional: upload spec / file", type=None, key="mgr_create_file")
            if st.button("Create Task"):
                marks, status = compute_marks_and_status(completion)
                file_path = save_uploaded_file(file_upload, prefix="task") if file_upload else None
                rec = {
                    "_id": str(uuid.uuid4()),
                    "record_type": "task",
                    "company": company,
                    "department": department,
                    "employee": employee,
                    "task": task_title,
                    "completion": completion,
                    "marks": marks,
                    "status": status,
                    "deadline": str(deadline),
                    "priority": priority,
                    "notes": "",
                    "attachments": file_path,
                    "submitted_on": str(datetime.now()),
                    "reviewed": False
                }
                upsert_records([rec])
                st.success(f"Task '{task_title}' created and assigned to {employee}")
                df = fetch_all()

        else:
            st.write("Select a task to update")
            df_tasks = df[df.get("record_type") == "task"] if not df.empty else pd.DataFrame()
            if df_tasks.empty:
                st.info("No tasks available to update.")
            else:
                df_tasks = df_tasks.reset_index(drop=True)
                df_tasks["_display"] = df_tasks.apply(lambda r: f"{r.get('employee','')} — {r.get('task','')} — {r.get('company','')}", axis=1)
                sel = st.selectbox("Select task", options=df_tasks["_display"].tolist())
                sel_row = df_tasks[df_tasks["_display"] == sel].iloc[0]
                comp = st.number_input("Completion %", min_value=0, max_value=100, value=int(sel_row.get("completion") or 0))
                emp = st.text_input("Assign To", value=sel_row.get("employee") or "")
                title = st.text_input("Task Title", value=sel_row.get("task") or "")
                dept = st.text_input("Department", value=sel_row.get("department") or "")
                dl = st.date_input("Deadline", value=pd.to_datetime(sel_row.get("deadline")).date() if sel_row.get("deadline") else date.today())
                status = st.selectbox("Status", ["On Track", "At Risk", "Delayed"], index=0)
                file_upload = st.file_uploader("Upload attachment (optional)", type=None, key="mgr_upd_file")
                if st.button("Update Task"):
                    marks, _ = compute_marks_and_status(comp)
                    file_path = save_uploaded_file(file_upload, prefix="task") if file_upload else sel_row.get("attachments")
                    updated = {
                        "_id": sel_row.get("_id"),
                        "record_type": "task",
                        "company": sel_row.get("company"),
                        "department": dept,
                        "employee": emp,
                        "task": title,
                        "completion": comp,
                        "marks": marks,
                        "status": status,
                        "deadline": str(dl),
                        "attachments": file_path,
                        "last_reassigned_on": str(datetime.now())
                    }
                    upsert_records([updated])
                    st.success("Task updated")
                    df = fetch_all()

    # ---------- Review Tasks ----------
    with tabs[1]:
        st.subheader("🧾 Review Pending Tasks")
        df_tasks = df[df.get("record_type") == "task"] if not df.empty else pd.DataFrame()
        if df_tasks.empty:
            st.info("No tasks.")
        else:
            pending = df_tasks[df_tasks.get("reviewed") != True]
            if pending.empty:
                st.success("No pending tasks.")
            else:
                for i, r in pending.iterrows():
                    st.markdown(f"**{r.get('task','Untitled')}** — {r.get('employee','')}")
                    new_comp = st.slider(f"Completion ({r.get('employee')})", 0, 100, int(r.get("completion") or 0), key=f"rev_{i}")
                    comment = st.text_area("Manager comment", key=f"mgr_comment_{i}")
                    if st.button(f"Approve {r.get('task','')}", key=f"approve_{i}"):
                        marks, status = compute_marks_and_status(new_comp)
                        updated = dict(r)
                        updated["completion"] = new_comp
                        updated["marks"] = marks
                        updated["status"] = status
                        updated["reviewed"] = True
                        updated["manager_comment"] = comment
                        upsert_records([updated])
                        st.success("Reviewed & approved")
                        df = fetch_all()
                        st.experimental_rerun()

    # ---------- Meetings ----------
    with tabs[2]:
        st.subheader("📅 Meetings (create / view / cancel)")
        mode_meet = st.radio("Mode", ["Schedule New Meeting", "View Meetings"], index=0)
        if mode_meet == "Schedule New Meeting":
            title = st.text_input("Meeting Title")
            organizer = st.text_input("Organizer")
            participants = st.text_area("Participants (comma-separated)")
            meet_date = st.date_input("Date", value=date.today())
            meet_time = st.time_input("Time", value=datetime.now().time())
            duration_min = st.number_input("Duration (minutes)", min_value=15, max_value=480, value=60)
            agenda = st.text_area("Agenda")
            meet_file = st.file_uploader("Attach file (optional)", key="mgr_meet_file")
            if st.button("Schedule Meeting"):
                dt = datetime.combine(meet_date, meet_time)
                file_path = save_uploaded_file(meet_file, prefix="meeting") if meet_file else None
                meeting = {
                    "_id": str(uuid.uuid4()),
                    "record_type": "meeting",
                    "meeting_title": title,
                    "organizer": organizer,
                    "participants": participants,
                    "meeting_datetime": dt.isoformat(),
                    "duration_min": int(duration_min),
                    "agenda": agenda,
                    "attachment": file_path,
                    "created_on": str(datetime.now())
                }
                upsert_records([meeting])
                st.success("Meeting scheduled")
                df = fetch_all()
        else:
            meetings = df[df.get("record_type") == "meeting"] if not df.empty else pd.DataFrame()
            if meetings.empty:
                st.info("No meetings scheduled")
            else:
                for i, m in meetings.iterrows():
                    st.markdown(f"**{m.get('meeting_title')}** — {m.get('meeting_datetime')}")
                    st.write(f"Organizer: {m.get('organizer')}")
                    st.write(f"Participants: {m.get('participants')}")
                    st.write(f"Agenda: {m.get('agenda')}")
                    if m.get("attachment"):
                        st.write(f"Attachment path: `{m.get('attachment')}`")
                    if st.button(f"Cancel Meeting: {m.get('meeting_title')}", key=f"cancel_{i}"):
                        updated = dict(m)
                        updated["status"] = "Canceled"
                        upsert_records([updated])
                        st.success("Meeting canceled")
                        df = fetch_all()
                        st.experimental_rerun()

    # ---------- Overview ----------
    with tabs[3]:
        st.subheader("📊 Quick Overview")
        df_tasks = df[df.get("record_type") == "task"] if not df.empty else pd.DataFrame()
        if df_tasks.empty:
            st.info("No tasks to show")
        else:
            # simple metrics
            st.metric("Total Tasks", len(df_tasks))
            st.metric("Pending Review", int((df_tasks.get("reviewed") != True).sum()))
            # cluster plot if enough rows
            try:
                num_df = df_tasks.dropna(subset=["marks", "completion"]).copy()
                num_df["marks"] = pd.to_numeric(num_df["marks"], errors="coerce")
                num_df["completion"] = pd.to_numeric(num_df["completion"], errors="coerce")
                if len(num_df) >= 3:
                    km = KMeans(n_clusters=3, random_state=42).fit(num_df[["marks", "completion"]])
                    num_df["cluster"] = km.labels_
                    fig = px.scatter(num_df, x="completion", y="marks", color="cluster", hover_data=["employee", "task"])
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Need 3+ tasks for clustering visualization")
            except Exception:
                st.info("Visualization currently not available")

# ------------- TEAM MEMBER UI -------------
elif role == "Team Member":
    st.header("👷 Team Member — Update / Upload / Meetings")

    tabs = st.tabs(["My Tasks", "Update & Upload Work", "My Meetings"])
    # My Tasks
    with tabs[0]:
        st.subheader("📋 Your Tasks")
        if df.empty or "employee" not in df.columns:
            st.info("No tasks available.")
        else:
            emp_list = sorted(df["employee"].dropna().unique())
            emp_name = st.selectbox("Select your name", options=emp_list)
            if emp_name:
                my_tasks = df[(df.get("record_type") == "task") & (df.get("employee") == emp_name)]
                if my_tasks.empty:
                    st.info("No tasks assigned.")
                else:
                    cols = [c for c in ["company", "task", "completion", "marks", "status", "deadline", "attachments"] if c in my_tasks.columns]
                    st.dataframe(my_tasks[cols])

    # Update & Upload Work
    with tabs[1]:
        st.subheader("✅ Update Progress & Upload Work")
        emp = st.text_input("Your name")
        task_name = st.text_input("Task title (existing or new)")
        completion = st.number_input("Completion %", min_value=0, max_value=100, value=0)
        comment = st.text_area("Progress comment")
        upload = st.file_uploader("Upload work (optional)", key="tm_upload")
        if st.button("Submit Update"):
            marks, status = compute_marks_and_status(completion)
            file_path = save_uploaded_file(upload, prefix="work") if upload else None
            rec = {
                "_id": str(uuid.uuid4()),
                "record_type": "task",
                "employee": emp,
                "task": task_name,
                "completion": completion,
                "marks": marks,
                "status": status,
                "progress_comment": comment,
                "work_attachment": file_path,
                "submitted_on": str(datetime.now()),
                "reviewed": False
            }
            upsert_records([rec])
            st.success("Update submitted")
            df = fetch_all()

    # My Meetings
    with tabs[2]:
        st.subheader("📅 Meetings you're in")
        my_name = st.text_input("Enter your name to filter meetings (optional)", key="tm_meet_filter")
        meetings = df[df.get("record_type") == "meeting"] if not df.empty else pd.DataFrame()
        if meetings.empty:
            st.info("No meetings scheduled")
        else:
            if my_name:
                meetings = meetings[meetings.get("participants", "").str.lower().str.contains(my_name.lower(), na=False) |
                                     meetings.get("organizer", "").str.lower().str.contains(my_name.lower(), na=False)]
            for i, m in meetings.iterrows():
                st.markdown(f"**{m.get('meeting_title')}** — {m.get('meeting_datetime')}")
                st.write(f"Organizer: {m.get('organizer')}")
                st.write(f"Participants: {m.get('participants')}")
                st.write(f"Agenda: {m.get('agenda')}")
                if m.get("attachment"):
                    st.write(f"Attachment: `{m.get('attachment')}`")
                submit_file = st.file_uploader(f"Upload work for meeting '{m.get('meeting_title')}'", key=f"mt_{i}")
                if submit_file:
                    fp = save_uploaded_file(submit_file, prefix="meeting_work")
                    subs = m.get("meeting_submissions") or []
                    if not isinstance(subs, list):
                        subs = []
                    subs.append({"submitted_by": my_name or "unknown", "file": fp, "submitted_on": str(datetime.now())})
                    updated = dict(m)
                    updated["meeting_submissions"] = subs
                    upsert_records([updated])
                    st.success("Work attached to meeting")
                    df = fetch_all()

# ------------- CLIENT UI -------------
elif role == "Client":
    st.header("🧾 Client Portal — Approve / View")

    tabs = st.tabs(["Project Overview", "Approve Tasks", "Meetings"])
    with tabs[0]:
        st.subheader("Project Overview")
        df_tasks = df[df.get("record_type") == "task"] if not df.empty else pd.DataFrame()
        if df_tasks.empty:
            st.info("No task data")
        else:
            comps = sorted(df_tasks["company"].dropna().unique())
            comp = st.selectbox("Select Company", options=comps)
            comp_df = df_tasks[df_tasks["company"] == comp]
            cols = [c for c in ["employee", "task", "completion", "marks", "status", "work_attachment"] if c in comp_df.columns]
            st.dataframe(comp_df[cols])
    with tabs[1]:
        st.subheader("Approve Reviewed Tasks")
        df_tasks = df[df.get("record_type") == "task"] if not df.empty else pd.DataFrame()
        ready = df_tasks[df_tasks.get("reviewed") == True]
        if ready.empty:
            st.info("No tasks ready for client approval")
        else:
            for i, r in ready.iterrows():
                st.markdown(f"**{r.get('task')}** — {r.get('employee')}")
                if st.button(f"Approve {r.get('task')}", key=f"cl_app_{i}"):
                    updated = dict(r)
                    updated["client_approved_on"] = str(datetime.now())
                    upsert_records([updated])
                    st.success("Approved")
                    df = fetch_all()
    with tabs[2]:
        st.subheader("Meetings")
        meetings = df[df.get("record_type") == "meeting"] if not df.empty else pd.DataFrame()
        if meetings.empty:
            st.info("No meetings")
        else:
            st.dataframe(meetings[["meeting_title", "organizer", "meeting_datetime", "participants", "attachment"]])

# ------------- ADMIN UI -------------
elif role == "Admin":
    st.header("🧠 Admin — Global Overview")
    df_tasks = df[df.get("record_type") == "task"] if not df.empty else pd.DataFrame()
    st.metric("Total Records", len(df))
    st.metric("Total Tasks", len(df_tasks))
    if not df_tasks.empty and {"marks","completion","employee"} <= set(df_tasks.columns):
        try:
            leaderboard = df_tasks.groupby("employee")[["marks", "completion"]].mean().sort_values("marks", ascending=False)
            st.subheader("Leaderboard")
            st.dataframe(leaderboard)
            fig = px.bar(leaderboard.reset_index(), x="employee", y="marks", title="Avg Marks by Employee")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("Leaderboard unavailable")

# ------------- FOOTER -------------
st.markdown("---")
st.caption("Stable edition: meeting scheduling + file upload + assign/reassign. If you still get errors, paste the traceback here and I will fix it immediately.")
