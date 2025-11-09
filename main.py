# main.py
"""
Enterprise AI Workforce Intelligence — with Meetings, File Upload & HR Intelligence Layer
Dependencies:
  pip install streamlit pinecone-client scikit-learn plotly huggingface-hub PyPDF2 openpyxl pandas
Notes:
  - Put PINECONE_API_KEY and HUGGINGFACEHUB_API_TOKEN in .streamlit/secrets.toml
  - This file is additive: it retains task/feedback/leave features and extends with meeting + HR analytics.
"""

import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import pandas as pd
import uuid, json, logging, time, os
from datetime import datetime, date
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import plotly.express as px

# Optional HF client
try:
    from huggingface_hub import InferenceClient
    HF_HUB_AVAILABLE = True
except Exception:
    HF_HUB_AVAILABLE = False

# Optional PDF parsing
try:
    import PyPDF2
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

# --------------------------
# App configuration
# --------------------------
st.set_page_config(page_title="AI Workforce Intelligence", layout="wide")
st.title("AI Workforce Intelligence Platform")

PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
HF_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")
INDEX_NAME = "tasks"
MEETING_INDEX = "tasks"  # separate index for meetings (optional)
DIMENSION = 1024  # keep consistent for vectors (if using embeddings)

# --------------------------
# Logging and helpers
# --------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("workforce_ai")

def safe_rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

def now_ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def rand_vec(dim=DIMENSION):
    return np.random.rand(dim).tolist()

def safe_meta(md: dict):
    clean = {}
    for k, v in md.items():
        try:
            if v is None:
                v = ""
            elif isinstance(v, (datetime, date)):
                v = v.isoformat()
            elif isinstance(v, (np.generic, np.number)):
                v = float(v)
            elif isinstance(v, (list, dict)):
                v = json.dumps(v)
            # else keep primitive types (str, int, float, bool)
        except Exception:
            v = str(v)
        clean[k] = v
    return clean

# --------------------------
# Pinecone safe init
# --------------------------
index = None
meeting_index = None
if PINECONE_API_KEY:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        existing = [i["name"] for i in pc.list_indexes()]
        # create main index if missing
        if INDEX_NAME not in existing:
            pc.create_index(name=INDEX_NAME, dimension=DIMENSION, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
            # wait until ready
            while True:
                try:
                    desc = pc.describe_index(INDEX_NAME)
                    ready = desc.get("status", {}).get("ready", False)
                    if ready:
                        break
                except Exception:
                    pass
                time.sleep(2)
        index = pc.Index(INDEX_NAME)
        # create meeting index
        if MEETING_INDEX not in existing:
            if MEETING_INDEX not in [i["name"] for i in pc.list_indexes()]:
                pc.create_index(name=MEETING_INDEX, dimension=DIMENSION, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
                while True:
                    try:
                        desc = pc.describe_index(MEETING_INDEX)
                        ready = desc.get("status", {}).get("ready", False)
                        if ready:
                            break
                    except Exception:
                        pass
                    time.sleep(2)
        meeting_index = pc.Index(MEETING_INDEX)
        st.success("Connected to Pinecone indices.")
    except Exception as e:
        st.warning(f"Pinecone init failed; running local only. ({e})")
        index = None
        meeting_index = None
else:
    st.warning("Pinecone key not set. Running in local mode (non-persistent).")

# --------------------------
# Local fallback stores (session)
# --------------------------
if "LOCAL_TASKS" not in st.session_state:
    st.session_state.LOCAL_TASKS = {}     # id -> metadata
if "LOCAL_MEETINGS" not in st.session_state:
    st.session_state.LOCAL_MEETINGS = {}  # id -> metadata
if "LOCAL_LEAVES" not in st.session_state:
    st.session_state.LOCAL_LEAVES = pd.DataFrame(columns=["Employee","Type","From","To","Reason","Status"])
if "LOCAL_FEEDBACK" not in st.session_state:
    st.session_state.LOCAL_FEEDBACK = []  # list of dicts

# --------------------------
# Upsert & fetch helpers
# --------------------------
def upsert_task(id_, md):
    mdc = safe_meta(md)
    if index:
        try:
            index.upsert([{"id": id_, "values": rand_vec(), "metadata": mdc}])
            return True
        except Exception as e:
            st.warning(f"Pinecone task upsert error: {e}")
            return False
    else:
        st.session_state.LOCAL_TASKS[id_] = mdc
        return True

def upsert_meeting(id_, md):
    mdc = safe_meta(md)
    if meeting_index:
        try:
            meeting_index.upsert([{"id": id_, "values": rand_vec(), "metadata": mdc}])
            return True
        except Exception as e:
            st.warning(f"Pinecone meeting upsert error: {e}")
            return False
    else:
        st.session_state.LOCAL_MEETINGS[id_] = mdc
        return True

def fetch_tasks():
    if index:
        try:
            stats = index.describe_index_stats()
            if stats.get("total_vector_count", 0) == 0:
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
            st.warning(f"Pinecone fetch error: {e}")
            return pd.DataFrame()
    else:
        return pd.DataFrame(list(st.session_state.LOCAL_TASKS.values()))

def fetch_meetings():
    if meeting_index:
        try:
            stats = meeting_index.describe_index_stats()
            if stats.get("total_vector_count", 0) == 0:
                return pd.DataFrame()
            res = meeting_index.query(vector=rand_vec(), top_k=1000, include_metadata=True)
            if not getattr(res, "matches", None):
                return pd.DataFrame()
            rows = []
            for m in res.matches:
                md = m.metadata or {}
                md["_id"] = m.id
                rows.append(md)
            return pd.DataFrame(rows)
        except Exception as e:
            st.warning(f"Pinecone meeting fetch error: {e}")
            return pd.DataFrame()
    else:
        return pd.DataFrame(list(st.session_state.LOCAL_MEETINGS.values()))

# --------------------------
# Basic ML & NLP models (toy baseline)
# --------------------------
lin_reg = LinearRegression().fit([[0],[50],[100]], [0,2.5,5])
vectorizer = CountVectorizer()
svm = SVC()
try:
    X = vectorizer.fit_transform(["excellent", "good", "bad", "poor", "average"])
    svm.fit(X, [1,1,0,0,0])
except Exception:
    pass

# --------------------------
# Role selection
# --------------------------
role = st.sidebar.selectbox("Role", ["Manager","Team Member","Client","Admin"])
current_month = datetime.now().strftime("%B %Y")

# --------------------------
# Manager interface
# --------------------------
if role == "Manager":
    st.header("Manager Control Center")
    tabs = st.tabs(["Assign Task","Review Submissions","AI Insights","Task Reassignment","Leave Management","Meetings & Feedback","360 Overview"])
    # -- Assign Task (unchanged)
    with tabs[0]:
        st.subheader("Assign Task")
        with st.form("assign"):
            company = st.text_input("Company")
            department = st.text_input("Department")
            employee = st.text_input("Employee")
            task = st.text_input("Task Title")
            desc = st.text_area("Description")
            deadline = st.date_input("Deadline", value=date.today())
            submit = st.form_submit_button("Assign")
            if submit and company and employee and task:
                tid = str(uuid.uuid4())
                md = {
                    "company": company, "department": department, "employee": employee,
                    "task": task, "description": desc, "deadline": deadline.isoformat(),
                    "completion": 0, "marks": 0, "status": "Assigned", "assigned_on": now_ts()
                }
                ok = upsert_task(tid, md)
                if ok:
                    st.success("Task assigned.")

    # -- Review Submissions
    with tabs[1]:
        st.subheader("Review Submissions")
        df = fetch_tasks()
        if df.empty:
            st.info("No task data.")
        else:
            df["completion"] = pd.to_numeric(df.get("completion",0), errors="coerce").fillna(0)
            for _, r in df.iterrows():
                st.markdown(f"### {r.get('task')}")
                st.write(f"Employee: {r.get('employee','')}")
                st.write(f"Completion: {r.get('completion')}%")
                adjusted = st.slider(f"Adjust completion for {r.get('task')}", 0,100,int(r.get('completion',0)), key=f"adj_{r.get('_id')}")
                comments = st.text_area(f"Manager comments for {r.get('task')}", key=f"cm_{r.get('_id')}")
                approve = st.radio(f"Approve {r.get('task')}?", ["Yes","No"], key=f"ap_{r.get('_id')}")
                if st.button(f"Finalize {r.get('task')}", key=f"fin_{r.get('_id')}"):
                    sentiment = "Positive" if (svm.predict(vectorizer.transform([comments]))[0] == 1) else "Negative" if comments else "Neutral"
                    md = {**r, "completion": adjusted, "marks": float(lin_reg.predict([[adjusted]])[0]), "manager_comments": comments, "reviewed": True, "sentiment": sentiment, "approved_by_boss": approve=="Yes", "reviewed_on": now_ts()}
                    upsert_task(r.get("_id") or str(uuid.uuid4()), md)
                    st.success("Review finalized.")
                    safe_rerun()

    # -- AI Insights
    with tabs[2]:
        st.subheader("AI Insights")
        df = fetch_tasks()
        if df.empty:
            st.info("No data for insights.")
        else:
            df["completion"] = pd.to_numeric(df.get("completion",0), errors="coerce").fillna(0)
            df["marks"] = pd.to_numeric(df.get("marks",0), errors="coerce").fillna(0)
            c1,c2,c3 = st.columns(3)
            c1.metric("Average Completion", f"{df['completion'].mean():.1f}%")
            c2.metric("Average Marks", f"{df['marks'].mean():.2f}")
            c3.metric("Employees", df['employee'].nunique())
            if "department" in df.columns:
                st.plotly_chart(px.bar(df.groupby("department")["completion"].mean().reset_index(), x="department", y="completion"), use_container_width=True)
            st.subheader("Ask AI")
            q = st.text_input("Enter question")
            if st.button("Run AI"):
                if HF_HUB_AVAILABLE and HF_TOKEN:
                    try:
                        hf = InferenceClient(token=HF_TOKEN)
                        avg = df["completion"].mean()
                        low_count = int((df["completion"] < avg - 10).sum())
                        context = f"Average completion: {avg:.1f}%. Employees below average: {low_count}."
                        prompt = f"Context: {context}\nQuestion: {q}\nProvide brief, actionable insights."
                        try:
                            resp = hf.text_generation(model="mistralai/Mixtral-8x7B-Instruct", prompt=prompt, max_new_tokens=200)
                        except TypeError:
                            resp = hf.text_generation(model="mistralai/Mixtral-8x7B-Instruct", inputs=prompt, max_new_tokens=200)
                        if isinstance(resp, dict) and "generated_text" in resp:
                            out = resp["generated_text"]
                        elif isinstance(resp, list) and "generated_text" in resp[0]:
                            out = resp[0]["generated_text"]
                        else:
                            out = str(resp)
                        st.write(out)
                    except Exception as e:
                        st.error(f"AI query error: {e}")
                else:
                    st.warning("Hugging Face token not configured.")

    # -- Task Reassignment
    with tabs[3]:
        st.subheader("Task Reassignment")
        df = fetch_tasks()
        if df.empty:
            st.info("No tasks.")
        else:
            task_choice = st.selectbox("Select task", df["task"].unique())
            new_owner = st.text_input("New owner")
            reason = st.text_area("Reason for reassignment")
            if st.button("Reassign"):
                row = df[df["task"]==task_choice].iloc[0].to_dict()
                row["employee"] = new_owner
                row["status"] = "Reassigned"
                row["reassign_reason"] = reason
                row["reassigned_on"] = now_ts()
                upsert_task(row.get("_id") or str(uuid.uuid4()), row)
                st.success("Task reassigned.")

    # -- Leave Management
    with tabs[4]:
        st.subheader("Leave Management")
        # show existing leaves (local store)
        leaves = st.session_state.LOCAL_LEAVES
        if leaves.empty:
            st.info("No leave requests.")
        else:
            st.dataframe(leaves)
        with st.form("leave_form"):
            le_emp = st.text_input("Employee")
            le_type = st.selectbox("Type", ["Casual","Sick","Earned"])
            le_from = st.date_input("From")
            le_to = st.date_input("To")
            le_reason = st.text_area("Reason")
            submit_leave = st.form_submit_button("Submit Leave Request")
            if submit_leave:
                new = {"Employee": le_emp, "Type": le_type, "From": str(le_from), "To": str(le_to), "Reason": le_reason, "Status": "Pending"}
                leaves = pd.concat([leaves, pd.DataFrame([new])], ignore_index=True)
                st.session_state.LOCAL_LEAVES = leaves
                st.success("Leave requested.")

        # Approve/Reject
        if not leaves.empty:
            sel = st.selectbox("Select leave index to update", leaves.index)
            action = st.radio("Action", ["Approve","Reject"])
            if st.button("Update Leave"):
                st.session_state.LOCAL_LEAVES.loc[sel,"Status"] = action
                st.success("Leave updated.")
                safe_rerun()

    # -- Meetings & Feedback
    with tabs[5]:
        st.subheader("Meetings & Feedback")
        # Create meeting
        with st.form("create_meeting"):
            m_title = st.text_input("Meeting title")
            m_date = st.date_input("Meeting date", value=date.today())
            m_time = st.time_input("Meeting time", value=datetime.now().time())
            m_attendees = st.text_area("Attendees (comma separated)")
            m_agenda = st.text_area("Agenda / Notes (optional)")
            m_upload = st.file_uploader("Upload meeting minutes or report (pdf/csv/xlsx/txt)", type=["pdf","csv","xlsx","txt"])
            submit_meet = st.form_submit_button("Create Meeting")
            if submit_meet and m_title:
                mid = str(uuid.uuid4())
                meeting_md = {
                    "title": m_title, "date": str(m_date), "time": str(m_time),
                    "attendees": m_attendees, "agenda": m_agenda, "created_on": now_ts()
                }
                # process uploaded file
                if m_upload is not None:
                    fname = m_upload.name
                    # read text for csv/xlsx/txt; for pdf try PyPDF2
                    extracted = ""
                    try:
                        if fname.lower().endswith(".csv"):
                            dfu = pd.read_csv(m_upload)
                            extracted = dfu.to_csv(index=False)
                        elif fname.lower().endswith(".xlsx") or fname.lower().endswith(".xls"):
                            dfu = pd.read_excel(m_upload)
                            extracted = dfu.to_csv(index=False)
                        elif fname.lower().endswith(".txt"):
                            extracted = m_upload.getvalue().decode(errors="ignore")
                        elif fname.lower().endswith(".pdf") and PDF_AVAILABLE:
                            reader = PyPDF2.PdfReader(m_upload)
                            pages = [p.extract_text() or "" for p in reader.pages]
                            extracted = "\n".join(pages)
                        else:
                            # save binary file in session for download / audit
                            raw_bytes = m_upload.getvalue()
                            storage_path = os.path.join("uploaded_files", f"{mid}_{fname}")
                            os.makedirs("uploaded_files", exist_ok=True)
                            with open(storage_path, "wb") as f:
                                f.write(raw_bytes)
                            meeting_md["file_path"] = storage_path
                            extracted = ""
                    except Exception as e:
                        st.warning(f"File processing issue: {e}")
                    meeting_md["file_name"] = fname
                    meeting_md["file_text"] = extracted[:30000]  # truncate to 30k chars to avoid huge metadata
                upsert_meeting(mid, meeting_md)
                st.success("Meeting created and stored.")

        # List meetings
        meetings = fetch_meetings()
        if meetings.empty:
            st.info("No meetings scheduled.")
        else:
            st.dataframe(meetings[["title","date","time","attendees","file_name"]].fillna(""))
            sel = st.selectbox("Select meeting title to add feedback", meetings["title"].unique())
            mrow = meetings[meetings["title"]==sel].iloc[0].to_dict()
            st.write("Selected:", mrow.get("title"), mrow.get("date"))
            fb_text = st.text_area("Add feedback / minutes summary for this meeting")
            fb_rating = st.slider("Rating (1-5)", 1,5,4)
            if st.button("Submit Meeting Feedback"):
                # store feedback record locally and in meeting metadata
                fb = {"meeting_id": mrow.get("_id"), "title": mrow.get("title"), "feedback": fb_text, "rating": int(fb_rating), "submitted_on": now_ts()}
                st.session_state.LOCAL_FEEDBACK.append(fb)
                # attach to meeting metadata
                mrow.setdefault("feedbacks", [])
                mrow["feedbacks"].append(fb)
                upsert_meeting(mrow.get("_id") or str(uuid.uuid4()), mrow)
                st.success("Meeting feedback stored.")

    # -- 360 Overview
    with tabs[6]:
        st.subheader("360 Overview")
        df = fetch_tasks()
        if df.empty:
            st.info("No task data.")
        else:
            emp = st.selectbox("Select employee", df["employee"].unique())
            emp_df = df[df["employee"]==emp]
            st.write("Recent tasks:")
            st.dataframe(emp_df[["task","completion","marks","status"]].sort_values("completion",ascending=False))
            # consolidated feedback from meeting feedbacks and local feedback
            meeting_df = fetch_meetings()
            fb_collect = [fb for fb in st.session_state.LOCAL_FEEDBACK if fb.get("title") and emp in (fb.get("title") or "")]
            st.write("Meeting feedback count:", len(fb_collect))
            avg_completion = emp_df["completion"].mean()
            st.write(f"Average completion: {avg_completion:.1f}%")
            # AI summary (optional)
            if HF_HUB_AVAILABLE and HF_TOKEN:
                try:
                    hf = InferenceClient(token=HF_TOKEN)
                    prompt = f"Employee: {emp}\nTasks: {emp_df.to_dict(orient='records')}\nMeetings feedback (sample): {fb_collect[:5]}\nProvide concise performance summary and recommendations."
                    try:
                        resp = hf.text_generation(model="mistralai/Mixtral-8x7B-Instruct", prompt=prompt, max_new_tokens=200)
                    except TypeError:
                        resp = hf.text_generation(model="mistralai/Mixtral-8x7B-Instruct", inputs=prompt, max_new_tokens=200)
                    out = resp.get("generated_text") if isinstance(resp, dict) else (resp[0].get("generated_text") if isinstance(resp, list) else str(resp))
                    st.subheader("AI Summary")
                    st.write(out)
                except Exception as e:
                    st.warning(f"AI summary error: {e}")

# --------------------------
# Team Member portal
# --------------------------
elif role == "Team Member":
    st.header("Team Member Portal")
    st.subheader("My Tasks")
    name = st.text_input("Enter your name")
    if st.button("Load My Tasks"):
        df = fetch_tasks()
        my = df[df["employee"]==name] if not df.empty else pd.DataFrame()
        if my.empty:
            st.info("No tasks assigned.")
        else:
            for _, r in my.iterrows():
                st.markdown(f"**{r.get('task')}**")
                st.write(r.get("description",""))
                new = st.slider(f"Completion for {r.get('task')}", 0,100,int(r.get("completion",0)), key=f"tm_{r.get('_id')}")
                if st.button(f"Submit {r.get('task')}", key=f"sub_{r.get('_id')}"):
                    r["completion"] = new
                    r["marks"] = float(lin_reg.predict([[new]])[0])
                    r["status"] = "Completed" if new>=100 else "In Progress"
                    upsert_task(r.get("_id") or str(uuid.uuid4()), r.to_dict())
                    st.success("Progress submitted.")

    st.subheader("Upload Meeting Minutes / Personal Report")
    uploaded = st.file_uploader("Upload file (txt/csv/xlsx/pdf)", type=["txt","csv","xlsx","pdf"])
    if uploaded is not None:
        fname = uploaded.name
        try:
            text = ""
            if fname.lower().endswith(".txt"):
                text = uploaded.getvalue().decode(errors="ignore")
            elif fname.lower().endswith(".csv"):
                dfup = pd.read_csv(uploaded)
                text = dfup.to_csv(index=False)
            elif fname.lower().endswith(".xlsx") or fname.lower().endswith(".xls"):
                dfup = pd.read_excel(uploaded)
                text = dfup.to_csv(index=False)
            elif fname.lower().endswith(".pdf") and PDF_AVAILABLE:
                reader = PyPDF2.PdfReader(uploaded)
                pages = [p.extract_text() or "" for p in reader.pages]
                text = "\n".join(pages)
            else:
                # save binary
                path = os.path.join("uploaded_files", f"{uuid.uuid4()}_{fname}")
                os.makedirs("uploaded_files", exist_ok=True)
                with open(path, "wb") as f:
                    f.write(uploaded.getvalue())
                st.success(f"File saved to {path}")
                text = ""
            # store as feedback record
            rec = {"uploader": name, "filename": fname, "text": (text[:30000] if text else ""), "uploaded_on": now_ts()}
            st.session_state.LOCAL_FEEDBACK.append(rec)
            st.success("File processed and stored in local feedback.")
        except Exception as e:
            st.error(f"Upload processing failed: {e}")

# --------------------------
# Client portal
# --------------------------
elif role == "Client":
    st.header("Client Dashboard")
    company = st.text_input("Enter your company to view project status")
    if company:
        df = fetch_tasks()
        if df.empty:
            st.info("No project data.")
        else:
            dfc = df[df.get("company","").str.lower()==company.lower()] if "company" in df.columns else pd.DataFrame()
            if dfc.empty:
                st.info("No tasks for this company.")
            else:
                dfc["completion"] = pd.to_numeric(dfc.get("completion",0), errors="coerce").fillna(0)
                st.metric("Project Completion", f"{dfc['completion'].mean():.1f}%")
                st.dataframe(dfc[["task","employee","deadline","completion","status"]])

                # feedback on completed deliverables
                completed = dfc[dfc["completion"]>=80]
                for _, row in completed.iterrows():
                    st.subheader(row["task"])
                    fb = st.text_area(f"Feedback for {row['task']}", key=f"cf_{row['_id']}")
                    rating = st.slider(f"Rating (1-5) for {row['task']}", 1,5,4, key=f"cr_{row['_id']}")
                    if st.button(f"Submit feedback for {row['task']}", key=f"csub_{row['_id']}"):
                        # attach to task metadata
                        row_md = {**row, "client_feedback": fb, "client_rating": int(rating), "client_feedback_on": now_ts()}
                        upsert_task(row["_id"], row_md)
                        st.success("Feedback submitted.")

# --------------------------
# Admin portal + HR Intelligence Layer (not a separate tab)
# --------------------------
elif role == "Admin":
    st.header("Admin Control Panel")
    df = fetch_tasks()
    st.subheader("Organization Summary")
    if df.empty:
        st.info("No data available.")
    else:
        df["completion"] = pd.to_numeric(df.get("completion",0), errors="coerce").fillna(0)
        df["marks"] = pd.to_numeric(df.get("marks",0), errors="coerce").fillna(0)
        st.metric("Average Completion", f"{df['completion'].mean():.1f}%")
        st.metric("Average Marks", f"{df['marks'].mean():.2f}")
        st.dataframe(df[["employee","department","task","completion","marks","status"]])

        # HR Intelligence Layer (heuristic risk scoring)
        st.subheader("HR Intelligence")
        # features: avg completion per employee, std dev, open task count, recent leaves count, meeting attendance
        emp_stats = df.groupby("employee").agg(avg_completion=("completion","mean"), task_count=("task","count"), avg_marks=("marks","mean")).reset_index()
        # leave counts from local
        leaves = st.session_state.LOCAL_LEAVES
        if not leaves.empty:
            leave_counts = leaves.groupby("Employee").size().rename("leave_count").reset_index()
            emp_stats = emp_stats.merge(leave_counts, left_on="employee", right_on="Employee", how="left").fillna(0).drop(columns=["Employee"])
        else:
            emp_stats["leave_count"] = 0
        # meeting attendance heuristic: count meetings where employee listed in attendees
        meetings = fetch_meetings()
        attendance = {}
        if not meetings.empty and "attendees" in meetings.columns:
            for _, m in meetings.iterrows():
                atts = str(m.get("attendees",""))
                for emp in emp_stats["employee"].tolist():
                    if emp and emp.lower() in atts.lower():
                        attendance[emp] = attendance.get(emp,0) + 1
        emp_stats["meeting_count"] = emp_stats["employee"].apply(lambda e: attendance.get(e,0))
        # compute attrition risk score (weighted heuristic)
        def attrition_score(row):
            score = 0.0
            # low completion increases risk
            score += max(0, (80 - row["avg_completion"]) / 80) * 0.6
            # low marks increases risk
            score += max(0, (3 - row.get("avg_marks",0)) / 3) * 0.2
            # many leaves increases risk
            score += min(1, row.get("leave_count",0) / 5) * 0.1
            # low meeting count slightly increases risk
            if row.get("meeting_count",0) == 0:
                score += 0.1
            return round(score, 3)
        emp_stats["attrition_score"] = emp_stats.apply(attrition_score, axis=1)
        emp_stats = emp_stats.sort_values("attrition_score", ascending=False)
        st.dataframe(emp_stats[["employee","avg_completion","avg_marks","task_count","leave_count","meeting_count","attrition_score"]].head(50))

        st.subheader("Top attrition risks (explainability)")
        top = emp_stats.head(10)
        for _, r in top.iterrows():
            reasons = []
            if r["avg_completion"] < 60:
                reasons.append(f"Low completion ({r['avg_completion']:.0f}%)")
            if r["avg_marks"] < 2:
                reasons.append(f"Low marks ({r['avg_marks']:.1f})")
            if r["leave_count"] > 2:
                reasons.append(f"Multiple leaves ({int(r['leave_count'])})")
            if r["meeting_count"] == 0:
                reasons.append("No meeting attendance recorded")
            st.write(f"{r['employee']} — risk {r['attrition_score']}: " + (", ".join(reasons) if reasons else "Monitor"))

    st.subheader("Meetings Index")
    mdf = fetch_meetings()
    if mdf.empty:
        st.info("No meetings stored.")
    else:
        st.dataframe(mdf[["title","date","attendees","file_name"]].fillna(""))

# End of file
