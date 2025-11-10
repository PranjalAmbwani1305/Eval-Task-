# main.py — AI Workforce Intelligence Platform (Enterprise + Meeting Management)
# Requirements:
# pip install streamlit pinecone-client scikit-learn plotly huggingface-hub pandas openpyxl PyPDF2

import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import pandas as pd
import uuid, json, time
from datetime import datetime, date
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from huggingface_hub import InferenceClient

# Optional PDF reader
try:
    import PyPDF2
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

# -------------------------
# App config
# -------------------------
st.set_page_config(page_title="AI Workforce Intelligence Platform", layout="wide")
st.title("AI Workforce Intelligence Platform")

# -------------------------
# Secrets / constants
# -------------------------
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
HF_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")
INDEX_NAME = "task"   # must be lowercase
DIMENSION = 1024

# -------------------------
# Init Pinecone (best-effort)
# -------------------------
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
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            # wait (best-effort)
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

# -------------------------
# Utilities
# -------------------------
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
                if isinstance(v, (str, int, float, bool)) or v is None:
                    clean[k] = v
                else:
                    clean[k] = str(v)
        except Exception:
            clean[k] = str(v)
    return clean

def upsert_data(id_, md: dict) -> bool:
    if not index:
        loc = st.session_state.setdefault("LOCAL_DATA", {})
        loc[str(id_)] = md
        return True
    try:
        index.upsert([{"id": str(id_), "values": rand_vec(), "metadata": safe_meta(md)}])
        return True
    except Exception as e:
        st.warning(f"Pinecone upsert failed: {e}")
        return False

def fetch_all() -> pd.DataFrame:
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
        # avoid querying empty index
        try:
            stats = index.describe_index_stats()
            if stats and stats.get("total_vector_count", 0) == 0:
                return pd.DataFrame()
        except Exception:
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

# Hugging Face safe wrapper
def hf_text_generation(prompt: str, model: str = "mistralai/Mixtral-8x7B-Instruct", max_new_tokens: int = 200):
    if not HF_TOKEN:
        raise RuntimeError("Hugging Face token not configured.")
    client = InferenceClient(token=HF_TOKEN)
    try:
        res = client.text_generation(model=model, inputs=prompt, max_new_tokens=max_new_tokens)
    except TypeError:
        res = client.text_generation(model=model, prompt=prompt, max_new_tokens=max_new_tokens)
    # extract text
    if isinstance(res, dict):
        return res.get("generated_text") or res.get("output") or json.dumps(res)
    if isinstance(res, list) and res and isinstance(res[0], dict):
        return res[0].get("generated_text") or json.dumps(res[0])
    return str(res)

# -------------------------
# Simple ML helper
# -------------------------
lin_reg = LinearRegression().fit([[0], [50], [100]], [0, 2.5, 5])

# -------------------------
# Role selection
# -------------------------
role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client", "HR (Admin)"])
current_month = datetime.now().strftime("%B %Y")

# -------------------------
# MANAGER
# -------------------------
if role == "Manager":
    st.header("Manager Dashboard")
    tabs = st.tabs(["Task Management", "Feedback & Review", "Meetings", "Leave Approvals", "Team Overview", "AI Insights"])

    # ----- Task Management -----
    with tabs[0]:
        st.subheader("Assign Task")
        with st.form("assign_task"):
            company = st.text_input("Company")
            department = st.text_input("Department")
            employee = st.text_input("Employee")
            task = st.text_input("Task Title")
            desc = st.text_area("Description")
            deadline = st.date_input("Deadline", value=date.today())
            submit = st.form_submit_button("Assign Task")
            if submit:
                if not employee or not task:
                    st.warning("Employee and Task Title required.")
                else:
                    tid = str(uuid.uuid4())
                    md = {
                        "type": "Task",
                        "company": company or "",
                        "department": department or "General",
                        "employee": employee,
                        "task": task,
                        "description": desc or "",
                        "deadline": deadline.isoformat(),
                        "completion": 0,
                        "marks": 0,
                        "status": "Assigned",
                        "created": now()
                    }
                    if upsert_data(tid, md):
                        st.success(f"Assigned '{task}' to {employee}.")

        # quick view + reassign
        df_all = fetch_all()
        df_tasks = pd.DataFrame()
        if not df_all.empty:
            df_tasks = df_all[df_all.get("type") == "Task"]
        if df_tasks.empty:
            st.info("No tasks yet.")
        else:
            # ensure columns
            for c in ["employee", "department", "task", "completion", "status", "deadline", "created"]:
                if c not in df_tasks.columns:
                    df_tasks[c] = ""
            df_tasks["completion"] = pd.to_numeric(df_tasks["completion"], errors="coerce").fillna(0)
            st.dataframe(df_tasks[["employee","department","task","completion","status","deadline"]], use_container_width=True)

            st.markdown("---")
            st.subheader("Reassign Task")
            task_choice = st.selectbox("Select Task", df_tasks["task"].dropna().unique())
            new_emp = st.text_input("New Employee Name")
            reason = st.text_area("Reason for Reassignment")
            if st.button("Reassign Task"):
                rec = df_tasks[df_tasks["task"] == task_choice].iloc[0].to_dict()
                rec["employee"] = new_emp or rec.get("employee", "")
                rec["status"] = "Reassigned"
                rec["reassigned_reason"] = reason
                rec["reassigned_on"] = now()
                upsert_data(rec.get("_id") or str(uuid.uuid4()), rec)
                st.success("Task reassigned.")

    # ----- Feedback & Review -----
    with tabs[1]:
        st.subheader("Manager Feedback & Marks Review")
        df = fetch_all()
        if df.empty:
            st.info("No data available.")
        else:
            # identify completed tasks >=100 completion (tolerant to older data)
            df_tasks = df[df.get("type") == "Task"] if "type" in df.columns else df[df.get("task").notna()]
            if df_tasks.empty:
                st.info("No tasks found.")
            else:
                # ensure numeric
                df_tasks["completion"] = pd.to_numeric(df_tasks.get("completion", 0), errors="coerce").fillna(0)
                completed = df_tasks[(df_tasks["completion"] >= 100) & (df_tasks.get("status") != "Under Client Review")]
                if completed.empty:
                    st.info("No completed tasks awaiting review.")
                else:
                    selected = st.selectbox("Select Task to review", completed["task"].dropna().unique())
                    row = completed[completed["task"] == selected].iloc[0].to_dict()
                    st.write(f"Employee: {row.get('employee','')}, Department: {row.get('department','')}")
                    st.write(f"Description: {row.get('description','')}")
                    marks = st.slider("Final Marks (0–5)", 0.0, 5.0, float(row.get("marks", 0.0)))
                    feedback = st.text_area("Manager Feedback")
                    if st.button("Finalize Review"):
                        row["marks"] = marks
                        row["manager_feedback"] = feedback
                        row["manager_reviewed_on"] = now()
                        row["status"] = "Under Client Review"
                        upsert_data(row.get("_id") or str(uuid.uuid4()), row)
                        st.success("Review saved and sent to client for evaluation.")

    # ----- Meetings (NEW) -----
    with tabs[2]:
        st.subheader("Meetings — Schedule & Notes")
        # schedule meeting
        with st.form("schedule_meeting"):
            m_title = st.text_input("Meeting Title")
            m_date = st.date_input("Meeting Date", value=date.today())
            m_time = st.text_input("Meeting Time (HH:MM)", value="10:00")
            attendees = st.text_area("Attendees (comma separated)")
            m_notes_file = st.file_uploader("Upload meeting notes (txt/pdf/csv/xlsx)", type=["txt","pdf","csv","xlsx"])
            submit_meet = st.form_submit_button("Schedule Meeting")
            if submit_meet:
                mid = str(uuid.uuid4())
                meeting_md = {
                    "type": "Meeting",
                    "meeting_title": m_title or "",
                    "meeting_date": str(m_date),
                    "meeting_time": m_time or "",
                    "attendees": attendees or "",
                    "created": now()
                }
                # extract uploaded content if present
                if m_notes_file:
                    fname = m_notes_file.name.lower()
                    try:
                        if fname.endswith(".txt"):
                            content = m_notes_file.read().decode("utf-8", errors="ignore")
                        elif fname.endswith(".pdf") and PDF_AVAILABLE:
                            reader = PyPDF2.PdfReader(m_notes_file)
                            pages = [p.extract_text() or "" for p in reader.pages]
                            content = "\n".join(pages)
                        elif fname.endswith(".csv"):
                            content = pd.read_csv(m_notes_file).to_csv(index=False)
                        elif fname.endswith(".xlsx"):
                            content = pd.read_excel(m_notes_file).to_csv(index=False)
                        else:
                            content = ""
                        meeting_md["notes_file"] = m_notes_file.name
                        meeting_md["notes_text"] = content[:30000]
                    except Exception as e:
                        meeting_md["notes_text"] = f"ERROR reading file: {e}"
                upsert_data(mid, meeting_md)
                st.success("Meeting scheduled.")

        st.markdown("---")
        st.subheader("Upcoming Meetings & Notes")
        all_data = fetch_all()
        meets = all_data[all_data.get("type") == "Meeting"] if not all_data.empty else pd.DataFrame()
        if meets.empty:
            st.info("No meetings scheduled.")
        else:
            # ensure columns
            for col in ["meeting_title","meeting_date","meeting_time","attendees","notes_file"]:
                if col not in meets.columns:
                    meets[col] = ""
            st.dataframe(meets[["meeting_title","meeting_date","meeting_time","attendees","notes_file"]], use_container_width=True)

            # allow summarization using HF
            sel = st.selectbox("Select meeting to summarize", meets["meeting_title"].dropna().unique())
            if sel:
                meeting_row = meets[meets["meeting_title"] == sel].iloc[0].to_dict()
                notes = meeting_row.get("notes_text","")
                st.text_area("Notes (preview)", notes, height=200)
                if st.button("AI Summarize Meeting Notes"):
                    if not HF_TOKEN:
                        st.warning("Hugging Face token not configured.")
                    else:
                        prompt = f"Summarize the following meeting notes and list decisions, owners and next actions:\n\n{notes[:4000]}"
                        try:
                            summary = hf_text_generation(prompt=prompt, max_new_tokens=250)
                            st.subheader("AI Summary")
                            st.write(summary)
                        except Exception as e:
                            st.error(f"AI summarization failed: {e}")

    # ----- Leave Approvals -----
    with tabs[3]:
        st.subheader("Leave Approvals")
        df_all = fetch_all()
        leaves = pd.DataFrame()
        if not df_all.empty:
            # flexible detection of leave rows
            if "type" in df_all.columns:
                leaves = df_all[df_all["type"] == "Leave"]
            else:
                # fallback: rows without task but with status pending etc.
                leaves = df_all[df_all.get("task","") == ""]
        if leaves.empty:
            st.info("No leave requests pending.")
        else:
            for i, row in leaves.iterrows():
                if str(row.get("status","")).lower() == "pending":
                    emp = row.get("employee","Unknown")
                    lt = row.get("leave_type","Leave")
                    st.markdown(f"**{emp}** requested **{lt}** ({row.get('from','')} → {row.get('to','')})")
                    st.write(f"Reason: {row.get('reason','-')}")
                    decision = st.radio(f"Decision for {emp}", ["Approve","Reject"], key=f"dec_{i}")
                    if st.button(f"Finalize Decision for {emp}", key=f"btn_{i}"):
                        updated = dict(row)
                        updated["_id"] = str(row.get("_id") or uuid.uuid4())
                        updated["status"] = "Approved" if decision == "Approve" else "Rejected"
                        updated["approved_on"] = now()
                        upsert_data(updated["_id"], updated)
                        st.success(f"Leave {updated['status']} for {emp}")

    # ----- Team Overview -----
    with tabs[4]:
        st.markdown("## Team Performance Overview")
        df_all = fetch_all()
        df_tasks = pd.DataFrame()
        if not df_all.empty:
            # filter only tasks
            df_tasks = df_all[df_all.get("type") == "Task"]
        if df_tasks.empty:
            st.info("No task data available.")
        else:
            # ensure required columns
            required_cols = ["employee","department","task","completion","status","created"]
            for col in required_cols:
                if col not in df_tasks.columns:
                    df_tasks[col] = ""
            df_tasks["completion"] = pd.to_numeric(df_tasks["completion"], errors="coerce").fillna(0)
            st.dataframe(df_tasks[required_cols], use_container_width=True)
            # plot (department optional)
            try:
                if "department" in df_tasks.columns and df_tasks["department"].notna().any():
                    fig = px.bar(df_tasks, x="employee", y="completion", color="department", title="Completion by Employee")
                else:
                    fig = px.bar(df_tasks, x="employee", y="completion", title="Completion by Employee")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

    # ----- AI Insights -----
    with tabs[5]:
        st.subheader("AI Insights Center (Team)")
        df_all = fetch_all()
        if df_all.empty:
            st.info("No data available to analyze.")
        else:
            q = st.text_input("Ask AI (e.g., 'Who is underperforming today?')")
            if st.button("Generate Insight"):
                if not HF_TOKEN:
                    st.warning("Hugging Face token not configured.")
                else:
                    # create a compact summary
                    summary = df_all.describe(include="all").to_dict()
                    prompt = f"You're an AI analyst. Given this dataset summary:\n{summary}\nQuestion: {q}\nProvide concise, actionable management insights."
                    try:
                        out = hf_text_generation(prompt=prompt, max_new_tokens=200)
                        st.write(out)
                    except Exception as e:
                        st.error(f"AI query failed: {e}")

# -------------------------
# TEAM MEMBER
# -------------------------
elif role == "Team Member":
    st.header("Team Member Dashboard")
    name = st.text_input("Enter your name")
    if name:
        df_all = fetch_all()
        if df_all.empty:
            st.info("No records found.")
        else:
            # infer types if missing
            if "type" not in df_all.columns:
                df_all["type"] = df_all.apply(lambda r: "Task" if pd.notna(r.get("task")) and r.get("task") != "" else "Leave" if pd.notna(r.get("leave_type")) else "Task", axis=1)

            my_tasks = df_all[(df_all.get("employee","").astype(str).str.lower() == name.lower()) & (df_all["type"] == "Task")]
            my_leaves = df_all[(df_all.get("employee","").astype(str).str.lower() == name.lower()) & (df_all["type"] == "Leave")]

            t1, t2, t3 = st.tabs(["My Tasks", "My Leaves", "Meetings"])
            with t1:
                st.subheader("My Tasks")
                if my_tasks.empty:
                    st.info("No tasks assigned.")
                else:
                    for _, r in my_tasks.iterrows():
                        st.markdown(f"**{r.get('task','Untitled')}** — Status: {r.get('status','')}")
                        try:
                            current = int(float(r.get("completion", 0) or 0))
                        except Exception:
                            current = 0
                        comp = st.slider("Completion %", 0, 100, current, key=r.get("_id"))
                        if st.button(f"Update {r.get('_id')}", key=f"u_{r.get('_id')}"):
                            r["completion"] = comp
                            r["marks"] = float(lin_reg.predict([[comp]])[0])
                            r["status"] = "In Progress" if comp < 100 else "Completed"
                            upsert_data(r.get("_id") or str(uuid.uuid4()), r)
                            st.success("Progress updated.")

            with t2:
                st.subheader("Leave Requests")
                st.write("Submit leave request:")
                lt = st.selectbox("Leave Type", ["Casual","Sick","Earned"])
                f = st.date_input("From")
                tdate = st.date_input("To")
                reason = st.text_area("Reason")
                if st.button("Submit Leave"):
                    lid = str(uuid.uuid4())
                    md = {
                        "type": "Leave",
                        "employee": name,
                        "leave_type": lt,
                        "from": str(f),
                        "to": str(tdate),
                        "reason": reason,
                        "status": "Pending",
                        "submitted": now()
                    }
                    upsert_data(lid, md)
                    st.success("Leave submitted.")
                if not my_leaves.empty:
                    st.markdown("### My leave history")
                    st.dataframe(my_leaves[["leave_type","from","to","reason","status"]].fillna(""), use_container_width=True)

            with t3:
                st.subheader("Meetings I'm invited to")
                all_data = fetch_all()
                meets = all_data[all_data.get("type") == "Meeting"] if not all_data.empty else pd.DataFrame()
                if meets.empty:
                    st.info("No meetings scheduled.")
                else:
                    # show meetings where user listed as attendee
                    meets["att_list"] = meets.get("attendees","").astype(str).str.lower()
                    my_meets = meets[meets["att_list"].str.contains(name.lower())]
                    if my_meets.empty:
                        st.info("No meetings with you as attendee.")
                    else:
                        st.dataframe(my_meets[["meeting_title","meeting_date","meeting_time","notes_file"]].fillna(""), use_container_width=True)

# -------------------------
# CLIENT
# -------------------------
elif role == "Client":
    st.header("Client Review Center")
    company = st.text_input("Company Name")
    if company:
        df = fetch_all()
        df_client = pd.DataFrame()
        if not df.empty:
            df_client = df[(df.get("company","").astype(str).str.lower() == company.lower()) & (df.get("type") == "Task")]
        if df_client.empty:
            st.info("No tasks for this company.")
        else:
            tab1, tab2 = st.tabs(["Initial Review","Re-Evaluation"])
            with tab1:
                st.subheader("Pending Reviews")
                pending = df_client[(df_client.get("client_reviewed") != True) & (df_client.get("status") == "Under Client Review")]
                if pending.empty:
                    st.info("No pending reviews.")
                else:
                    choice = st.selectbox("Select task to review", pending["task"].unique())
                    fb = st.text_area("Feedback")
                    rating = st.slider("Rating (1-5)", 1, 5, 3)
                    if st.button("Submit Feedback"):
                        rec = pending[pending["task"] == choice].iloc[0].to_dict()
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
                    st.info("No reviewed tasks.")
                else:
                    choice = st.selectbox("Select task to re-evaluate", reviewed["task"].unique())
                    fb = st.text_area("Updated Feedback")
                    rating = st.slider("Re-evaluation rating", 1, 5, 3)
                    if st.button("Submit Re-evaluation"):
                        rec = reviewed[reviewed["task"] == choice].iloc[0].to_dict()
                        rec["client_reval_feedback"] = fb
                        rec["client_reval_rating"] = rating
                        rec["client_revaluated"] = True
                        rec["client_reval_on"] = now()
                        upsert_data(rec.get("_id") or str(uuid.uuid4()), rec)
                        st.success("Re-evaluation submitted.")

# -------------------------
# HR (Admin)
# -------------------------
elif role == "HR (Admin)":
    st.header("HR Dashboard — Performance & Leave")
    df_all = fetch_all()
    if df_all.empty:
        st.info("No data available.")
    else:
        # ensure columns
        for c in ["employee","department","task","completion","marks","type","status","leave_type","from","to"]:
            if c not in df_all.columns:
                df_all[c] = ""
        df_all["completion"] = pd.to_numeric(df_all.get("completion", 0), errors="coerce").fillna(0)
        df_all["marks"] = pd.to_numeric(df_all.get("marks", 0), errors="coerce").fillna(0)

        df_tasks = df_all[df_all["type"] == "Task"]
        df_leaves = df_all[df_all["type"] == "Leave"]

        tabs_hr = st.tabs(["Performance Clustering", "Leave Tracker"])

        with tabs_hr[0]:
            st.subheader("Performance Clustering")
            if df_tasks.empty:
                st.info("No task data.")
            else:
                try:
                    # KMeans clustering on completion & marks
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
                    st.dataframe(df_tasks[["employee","department","completion","marks","Performance Group"]].fillna(""), use_container_width=True)
                    # scatter
                    fig = None
                    try:
                        fig = px.scatter(df_tasks, x="completion", y="marks", color="Performance Group", hover_data=["employee","department"], title="Employee Performance Clusters")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        pass
                    # dept bar
                    dp = df_tasks.groupby("department")["completion"].mean().reset_index()
                    if not dp.empty:
                        fig2 = px.bar(dp, x="department", y="completion", title="Average Completion by Department")
                        st.plotly_chart(fig2, use_container_width=True)
                except Exception as e:
                    st.error(f"Clustering failed: {e}")

        with tabs_hr[1]:
            st.subheader("Leave Tracker")
            if df_leaves.empty:
                st.info("No leave data.")
            else:
                total = len(df_leaves)
                pending = (df_leaves["status"] == "Pending").sum()
                approved = (df_leaves["status"] == "Approved").sum()
                rejected = (df_leaves["status"] == "Rejected").sum()
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Total requests", total)
                c2.metric("Pending", int(pending))
                c3.metric("Approved", int(approved))
                c4.metric("Rejected", int(rejected))
                st.markdown("---")
                st.dataframe(df_leaves[["employee","leave_type","from","to","reason","status"]].fillna(""), use_container_width=True)
                # leave per employee
                lc = df_leaves.groupby("employee").size().reset_index(name="Total Leaves")
                if not lc.empty:
                    fig3 = px.bar(lc, x="employee", y="Total Leaves", title="Leave requests per employee")
                    st.plotly_chart(fig3, use_container_width=True)
