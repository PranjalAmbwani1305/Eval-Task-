# app.py
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import pandas as pd
import uuid
from datetime import datetime, date
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Optional libraries
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except Exception:
    TEXTBLOB_AVAILABLE = False

try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except Exception:
    TWILIO_AVAILABLE = False

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Enterprise AI Task System", layout="wide")
st.title("Enterprise AI Task & Performance Management")

# -----------------------------
# SAFE RERUN (compat)
# -----------------------------
def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

# -----------------------------
# PINECONE INIT (expects secret)
# -----------------------------
if "PINECONE_API_KEY" not in st.secrets:
    st.error("Add PINECONE_API_KEY to .streamlit/secrets.toml and restart.")
    st.stop()

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = st.secrets.get("PINECONE_INDEX", "task")
DIMENSION = 1024

# Create index if not exists
try:
    if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    index = pc.Index(INDEX_NAME)
except Exception as e:
    st.error(f"Pinecone init failed: {e}")
    st.stop()

# -----------------------------
# HELPERS: time, stable vector, safe metadata/upsert
# -----------------------------
def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def stable_vec(text: str):
    """Deterministic pseudo-embedding for task text (1024 dims)."""
    # Use a seeded RNG so same text yields same vector.
    h = abs(hash(text)) % (2**32)
    rng = np.random.default_rng(h)
    return rng.random(DIMENSION).tolist()

def safe_meta(md: dict):
    """Convert dates and non-JSON-safe values to serializable ones."""
    clean = {}
    for k, v in md.items():
        if isinstance(v, (datetime, date)):
            clean[k] = v.isoformat()
        elif v is None:
            clean[k] = ""
        else:
            # ensure simple types
            if isinstance(v, (str, int, float, bool, list, dict)):
                clean[k] = v
            else:
                clean[k] = str(v)
    return clean

def safe_upsert(index, md: dict):
    """Wrap Pinecone upsert with checks: ensures id, vector dimension, JSON-safe metadata."""
    try:
        if "_id" not in md or not md["_id"]:
            md["_id"] = str(uuid.uuid4())
        vec = stable_vec(str(md.get("task", md["_id"])))
        if len(vec) != DIMENSION:
            st.error("Embedding dimension mismatch.")
            return False
        meta = safe_meta(md)
        index.upsert([{"id": str(md["_id"]), "values": vec, "metadata": meta}])
        return True
    except Exception as e:
        st.error(f"Pinecone upsert failed: {e}")
        return False

def to_df_from_matches(matches):
    rows = []
    for m in matches:
        md = m.metadata or {}
        md["_id"] = m.id
        rows.append(md)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # normalize column names to lowercase
    df.columns = [c.lower() for c in df.columns]
    return df

def fetch_filtered(filter_dict=None, top_k=10000):
    try:
        res = index.query(vector=stable_vec("query"), top_k=top_k, include_metadata=True, filter=filter_dict or {})
        return to_df_from_matches(res.matches or [])
    except Exception as e:
        st.warning(f"Fetch failed: {e}")
        return pd.DataFrame()

def fetch_all(top_k=10000):
    try:
        res = index.query(vector=stable_vec("all"), top_k=top_k, include_metadata=True)
        return to_df_from_matches(res.matches or [])
    except Exception as e:
        st.warning(f"Fetch all failed: {e}")
        return pd.DataFrame()

# -----------------------------
# SIMPLE AI MODELS + NLP
# -----------------------------
lin_reg = LinearRegression()
lin_reg.fit([[0], [50], [100]], [0, 2.5, 5])

log_reg = LogisticRegression()
log_reg.fit([[0], [40], [80], [100]], [0, 0, 1, 1])

# small textset SVM fallback for sentiment
small_comments = ["excellent work", "needs improvement", "bad performance", "great job", "average"]
small_labels = [1, 0, 0, 1, 0]
vectorizer = CountVectorizer()
X_small = vectorizer.fit_transform(small_comments)
svm_sent = SVC()
svm_sent.fit(X_small, small_labels)

rf = RandomForestClassifier()
rf.fit(np.array([[10, 2], [50, 1], [90, 0], [100, 0]]), [0, 1, 0, 0])

def analyze_sentiment(text: str):
    t = (text or "").strip()
    if not t:
        return {"label": "Neutral", "polarity": 0.0, "subjectivity": 0.0}
    if TEXTBLOB_AVAILABLE:
        tb = TextBlob(t)
        p = tb.sentiment.polarity
        s = tb.sentiment.subjectivity
        if p > 0.35:
            label = "Positive"
        elif p < -0.25:
            label = "Negative"
        else:
            label = "Neutral"
        return {"label": label, "polarity": p, "subjectivity": s}
    else:
        try:
            pred = int(svm_sent.predict(vectorizer.transform([t]))[0])
            return {"label": "Positive" if pred == 1 else "Negative", "polarity": 0.5 if pred == 1 else -0.3, "subjectivity": 0.5}
        except Exception:
            return {"label": "Neutral", "polarity": 0.0, "subjectivity": 0.0}

# -----------------------------
# Notifications (SMTP + Twilio optional)
# -----------------------------
EMAIL_SENDER = st.secrets.get("EMAIL_SENDER", "")
EMAIL_PASSWORD = st.secrets.get("EMAIL_PASSWORD", "")
TWILIO_SID = st.secrets.get("TWILIO_SID", "")
TWILIO_TOKEN = st.secrets.get("TWILIO_TOKEN", "")
TWILIO_FROM = st.secrets.get("TWILIO_FROM", "")

def send_email(to_email: str, subject: str, body: str):
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        st.info("Email not configured in secrets. Skipping email.")
        return False
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_SENDER
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.starttls()
            s.login(EMAIL_SENDER, EMAIL_PASSWORD)
            s.send_message(msg)
        return True
    except Exception as e:
        st.warning(f"Email send failed: {e}")
        return False

def send_sms(to_number: str, body: str):
    if not (TWILIO_SID and TWILIO_TOKEN and TWILIO_FROM and TWILIO_AVAILABLE):
        st.info("Twilio not configured or available; skipping SMS.")
        return False
    try:
        client = TwilioClient(TWILIO_SID, TWILIO_TOKEN)
        client.messages.create(body=body, from_=TWILIO_FROM, to=to_number)
        return True
    except Exception as e:
        st.warning(f"SMS send failed: {e}")
        return False

def notify(email=None, phone=None, subject="Update", message=""):
    ok = True
    if email:
        ok = send_email(email, subject, message) or ok
    if phone:
        ok = send_sms(phone, message) or ok
    return ok

# -----------------------------
# UI: Role selection
# -----------------------------
role = st.sidebar.selectbox("Login as", ["Admin", "Manager", "Team Member", "Client"])
current_month = datetime.now().strftime("%B %Y")

# -----------------------------
# Admin Dashboard (enterprise-level)
# -----------------------------
if role == "Admin":
    st.header("Admin — Enterprise Dashboard")
    tab_overview, tab_users, tab_audit = st.tabs(["Overview & Analytics", "Companies & Users", "Audit & Export"])

    # Overview & Analytics
    with tab_overview:
        st.subheader("Global Task Analytics")
        df = fetch_all()
        if df.empty:
            st.info("No tasks available in the system.")
        else:
            # normalize numeric columns
            df["marks"] = pd.to_numeric(df.get("marks", 0), errors="coerce").fillna(0)
            df["completion"] = pd.to_numeric(df.get("completion", 0), errors="coerce").fillna(0)
            # Top employees
            st.markdown("### Top Employees by Average Marks")
            top = df.groupby("employee")["marks"].mean().reset_index().sort_values("marks", ascending=False).head(10)
            st.dataframe(top)

            # charts
            st.markdown("### Performance Charts")
            col1, col2 = st.columns(2)
            with col1:
                fig_bar = px.bar(top, x="employee", y="marks", title="Top Employees by Marks")
                st.plotly_chart(fig_bar, use_container_width=True)
            with col2:
                if len(df) >= 2:
                    # kmeans clustering
                    n_clusters = min(3, len(df))
                    km = KMeans(n_clusters=n_clusters, n_init=10).fit(df[["completion", "marks"]].fillna(0))
                    df["cluster"] = km.labels_
                    fig_scatter = px.scatter(df, x="completion", y="marks", color=df["cluster"].astype(str),
                                             hover_data=["employee", "task"], title="Completion vs Marks Clusters")
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.info("Not enough records for clustering.")

            # Summary metrics
            avg_marks = df["marks"].mean()
            avg_comp = df["completion"].mean()
            st.info(f"Average marks: {avg_marks:.2f} • Average completion: {avg_comp:.1f}%")
            # Export
            st.download_button("Download All Tasks (CSV)", df.to_csv(index=False), file_name="tasks_all.csv", mime="text/csv")

    # Companies & Users (lightweight management)
    with tab_users:
        st.subheader("Companies & Quick User Management")
        # For simplicity, management is stored as Pinecone metadata; in prod you'd use a proper DB.
        st.info("This panel allows quick admin-level actions on tasks and user lookup.")
        company_filter = st.text_input("Filter tasks by company (optional)")
        if st.button("Load Tasks"):
            if company_filter:
                df = fetch_filtered({"company": {"$eq": company_filter}})
            else:
                df = fetch_all()
            if df.empty:
                st.info("No matching tasks.")
            else:
                st.dataframe(df[["company", "employee", "task", "completion", "marks", "status"]])

    with tab_audit:
        st.subheader("Audit & Action Logs")
        st.info("Action logs are kept in task metadata under 'manager_actions'.")
        df = fetch_all()
        if df.empty:
            st.info("No data.")
        else:
            # extract manager actions
            logs = []
            for _, r in df.iterrows():
                ma = r.get("manager_actions", []) or []
                for act in ma:
                    logs.append({
                        "task": r.get("task"), "employee": r.get("employee"),
                        "type": act.get("type"), "note": act.get("note"), "by": act.get("by"), "on": act.get("on")
                    })
            if logs:
                logs_df = pd.DataFrame(logs).sort_values("on", ascending=False)
                st.dataframe(logs_df)
                st.download_button("Download Audit CSV", logs_df.to_csv(index=False), file_name="audit.csv", mime="text/csv")
            else:
                st.info("No manager actions recorded yet.")

# -----------------------------
# Manager Dashboard
# -----------------------------
elif role == "Manager":
    st.header("Manager — Assign, Review, 360°")
    tab_assign, tab_review, tab_360 = st.tabs(["Assign Task", "Boss Review & Adjustment", "Manager 360°"])

    # Assign
    with tab_assign:
        st.subheader("Assign Task")
        with st.form("assign_form"):
            company = st.text_input("Company")
            employee = st.text_input("Employee")
            email = st.text_input("Employee Email (optional)")
            phone = st.text_input("Employee Phone (optional)")
            task = st.text_input("Task Title")
            desc = st.text_area("Description")
            deadline = st.date_input("Deadline", value=date.today())
            submit = st.form_submit_button("Assign")
            if submit:
                if not (company and employee and task):
                    st.warning("Company, Employee, and Task are required.")
                else:
                    tid = str(uuid.uuid4())
                    md = {
                        "_id": tid, "company": company, "employee": employee,
                        "email": email, "phone": phone, "task": task, "description": desc,
                        "deadline": deadline.isoformat(), "month": current_month,
                        "completion": 0, "marks": 0, "status": "Assigned", "reviewed": False,
                        "client_reviewed": False, "assigned_on": now(), "manager_actions": []
                    }
                    ok = safe_upsert(index, md)
                    if ok:
                        notify_msg = f"You were assigned a new task: {task} (Company: {company}). Deadline: {deadline}"
                        notify(email if email else None, phone if phone else None, subject=f"New Task: {task}", message=notify_msg)
                        st.success("Task assigned.")
                        safe_rerun()

    # Review & Adjustment
    with tab_review:
        st.subheader("Review Team Submissions")
        # filter by company optional
        company_filter = st.text_input("Filter by Company (optional)")
        if company_filter:
            df = fetch_filtered({"company": {"$eq": company_filter}})
        else:
            df = fetch_all()
        if df.empty:
            st.info("No tasks to show.")
        else:
            df = df.fillna("")
            # Show only tasks with some completion (team submitted)
            submitted = df[pd.to_numeric(df["completion"], errors="coerce").fillna(0) > 0]
            if submitted.empty:
                st.info("No submitted tasks yet.")
            else:
                for _, r in submitted.iterrows():
                    st.markdown("----")
                    st.markdown(f"### {r.get('task')}")
                    left, right = st.columns([2,1])
                    with left:
                        st.write(f"Employee: {r.get('employee')} • Company: {r.get('company')}")
                        st.write(f"Reported Completion: {r.get('completion')}%")
                        st.write(f"Current Marks: {r.get('marks')}")
                        st.write(f"Deadline Risk: {r.get('deadline_risk','N/A')}")
                        st.write("Manager actions:")
                        for a in (r.get("manager_actions") or []):
                            st.write(f"- {a.get('type')} on {a.get('on')}: {a.get('note')}")
                    with right:
                        adj = st.slider("Adjust Completion", 0, 100, int(r.get("completion", 0)), key=f"adj_{r['_id']}")
                        marks_calc = float(lin_reg.predict([[adj]])[0])
                        comments = st.text_area("Comments", key=f"com_{r['_id']}")
                        approve = st.selectbox("Approve?", ["No", "Yes"], key=f"app_{r['_id']}")
                        reassign_to = st.text_input("Reassign to (optional)", key=f"re_{r['_id']}")
                        escalate = st.checkbox("Escalate", key=f"esc_{r['_id']}")
                        note_type = st.selectbox("Note type", ["None","Appreciation","Warning","Suggestion"], key=f"nt_{r['_id']}")
                        note_text = st.text_area("Note text", key=f"ntxt_{r['_id']}")
                        if st.button("Finalize Review", key=f"fin_{r['_id']}"):
                            sinfo = analyze_sentiment(comments)
                            md = dict(r)
                            md.update({
                                "completion": int(adj),
                                "marks": marks_calc,
                                "manager_comments": comments,
                                "reviewed": True,
                                "sentiment": sinfo["label"],
                                "approved_by_boss": (approve == "Yes"),
                                "reviewed_on": now()
                            })
                            # reassign
                            if reassign_to:
                                old = md.get("employee")
                                md["employee"] = reassign_to
                                md.setdefault("manager_actions", []).append({"type":"Reassigned","note":f"{old} -> {reassign_to}","by":"Manager","on":now()})
                                # notify
                                notify(md.get("email"), md.get("phone"), subject="Task Reassigned", message=f"Task '{md.get('task')}' reassigned to {reassign_to}")
                            # escalate
                            if escalate:
                                md["escalated"] = True
                                md.setdefault("manager_actions", []).append({"type":"Escalated","note":note_text or "Escalated","by":"Manager","on":now()})
                                # admin notification if set
                                admin_email = st.secrets.get("ADMIN_EMAIL")
                                if admin_email:
                                    send_email(admin_email, f"Task Escalated: {md.get('task')}", f"Task escalated by manager: {note_text}")
                            # note
                            if note_type and note_type != "None":
                                md.setdefault("manager_actions", []).append({"type":note_type,"note":note_text,"by":"Manager","on":now()})
                            safe_upsert(index, md)
                            notify(md.get("email"), md.get("phone"), subject=f"Task Reviewed: {md.get('task')}",
                                   message=f"Your task '{md.get('task')}' reviewed. Completion set to {md.get('completion')}%.")
                            st.success("Review finalized.")
                            safe_rerun()

    # Manager 360
    with tab_360:
        st.subheader("360° Performance Summaries")
        company_360 = st.text_input("Company (optional)")
        period = st.text_input("Period (e.g., 'November 2025')", value=current_month)
        if st.button("Generate 360° Summaries"):
            if company_360:
                df = fetch_filtered({"company": {"$eq": company_360}})
            else:
                df = fetch_all()
            if df.empty:
                st.info("No data for 360° generation.")
            else:
                df["marks"] = pd.to_numeric(df.get("marks",0), errors="coerce")
                df["completion"] = pd.to_numeric(df.get("completion",0), errors="coerce")
                reps = []
                for emp, group in df.groupby("employee"):
                    avg_marks = float(group["marks"].mean() or 0)
                    avg_comp = float(group["completion"].mean() or 0)
                    high_quality = len(group[group["completion"]>=90])
                    reviewed_count = len(group[group.get("reviewed","") == True])
                    pos = len(group[group.get("sentiment","") == "Positive"])
                    neg = len(group[group.get("sentiment","") == "Negative"])
                    tone = "Balanced"
                    if pos > neg + 1:
                        tone = "Strong positive feedback"
                    elif neg > pos + 1:
                        tone = "Concerns flagged"
                    summary_text = (f"{emp}: Avg Marks {avg_marks:.2f}, Avg Completion {avg_comp:.1f}%. High-quality tasks: {high_quality}. Reviewed: {reviewed_count}. Tone: {tone}")
                    reps.append({"employee": emp, "avg_marks": avg_marks, "avg_comp": avg_comp, "high_quality": high_quality, "reviewed": reviewed_count, "pos": pos, "neg": neg, "summary": summary_text})
                rep_df = pd.DataFrame(reps).sort_values("avg_marks", ascending=False)
                st.dataframe(rep_df)
                st.download_button("Download 360 CSV", rep_df.to_csv(index=False), file_name="360_summary.csv", mime="text/csv")

# -----------------------------
# Team Member Dashboard
# -----------------------------
elif role == "Team Member":
    st.header("Team Member — Update Progress (AI Feedback)")
    company = st.text_input("Company")
    employee = st.text_input("Your name")
    if st.button("Load My Tasks"):
        if not (company and employee):
            st.warning("Enter company and your name.")
        else:
            try:
                res = index.query(vector=stable_vec(employee), top_k=500, include_metadata=True, filter={"company": {"$eq": company}, "employee": {"$eq": employee}})
                st.session_state["tasks"] = [(m.id, m.metadata) for m in res.matches or []]
                st.success(f"Loaded {len(st.session_state['tasks'])} tasks.")
            except Exception as e:
                st.error(f"Error loading tasks: {e}")

    tasks = st.session_state.get("tasks", [])
    if isinstance(tasks, list) and all(isinstance(x, (tuple, list)) and len(x) == 2 for x in tasks):
        for tid, md in tasks:
            if not isinstance(md, dict):
                continue
            st.markdown("----")
            st.subheader(md.get("task", "Unnamed"))
            st.write(md.get("description", ""))
            curr = float(md.get("completion", 0) or 0)
            new = st.slider("Completion %", 0, 100, int(curr), key=f"tm_{tid}")
            overtime = st.number_input("Overtime hours (optional)", min_value=0.0, value=float(md.get("overtime", 0) or 0.0), step=0.5, key=f"ot_{tid}")
            request_help = st.checkbox("Request help/escalation", key=f"help_{tid}")
            if st.button("Submit Update", key=f"submit_{tid}"):
                marks = float(lin_reg.predict([[new]])[0])
                status = "On Track" if log_reg.predict([[new]])[0] == 1 else "Delayed"
                risk_pred = rf.predict([[new, 0]])[0]
                md2 = dict(md)
                md2.update({
                    "completion": int(new),
                    "marks": marks,
                    "status": status,
                    "deadline_risk": "High" if risk_pred else "Low",
                    "overtime": float(overtime),
                    "help_requested": bool(request_help),
                    "submitted_on": now()
                })
                # sentiment from any quick note (optional)
                safe_upsert(index, md2)
                # notify manager if configured in secrets
                mgr_email = st.secrets.get("MANAGER_EMAIL")
                if mgr_email:
                    notify(mgr_email, None, subject=f"Update: {md2.get('task')}", message=f"{employee} updated '{md2.get('task')}' to {new}%")
                st.success(f"Update submitted — {status} (Risk: {md2['deadline_risk']})")
                safe_rerun()
    else:
        st.info("No tasks loaded. Click 'Load My Tasks' to fetch assignments.")

# -----------------------------
# Client Dashboard
# -----------------------------
elif role == "Client":
    st.header("Client — Review & Approve")
    company = st.text_input("Company Name")
    if st.button("Load Reviewed Tasks"):
        if not company:
            st.warning("Enter company name.")
        else:
            try:
                res = index.query(vector=stable_vec(company), top_k=500, include_metadata=True, filter={"company": {"$eq": company}, "reviewed": {"$eq": True}})
                st.session_state["ctasks"] = [(m.id, m.metadata) for m in res.matches or []]
                st.success(f"Loaded {len(st.session_state['ctasks'])} tasks.")
            except Exception as e:
                st.error(f"Load failed: {e}")

    ctasks = st.session_state.get("ctasks", [])
    if isinstance(ctasks, list) and all(isinstance(x, (tuple, list)) and len(x) == 2 for x in ctasks):
        for tid, md in ctasks:
            if not isinstance(md, dict):
                continue
            st.markdown("----")
            st.subheader(md.get("task", "Unnamed"))
            st.write(f"Employee: {md.get('employee', '')}")
            st.write(f"Completion: {md.get('completion', 0)}% • Marks: {md.get('marks', 0)}")
            client_feedback = st.text_area("Client Feedback (optional)", key=f"cf_{tid}")
            decision = st.selectbox("Decision", ["Approve Deliverable", "Approve with changes", "Reject"], key=f"cd_{tid}")
            if st.button("Submit Decision", key=f"cdbtn_{tid}"):
                md2 = dict(md)
                md2.update({
                    "client_reviewed": True,
                    "client_comments": client_feedback,
                    "client_decision": decision,
                    "client_approved_on": now()
                })
                safe_upsert(index, md2)
                # notify manager / employee
                notify(md2.get("email"), md2.get("phone"), subject=f"Client Decision: {md2.get('task')}", message=f"Decision: {decision}\nFeedback: {client_feedback}")
                st.success("Client decision submitted.")
                safe_rerun()
    else:
        st.info("No reviewed tasks loaded yet. Click 'Load Reviewed Tasks'.")

# -----------------------------
# END
# -----------------------------
