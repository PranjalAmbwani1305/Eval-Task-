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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Optional advanced NLP
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except Exception:
    TEXTBLOB_AVAILABLE = False

# Optional Twilio for SMS
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except Exception:
    TWILIO_AVAILABLE = False

# -----------------------------
# APP CONFIG
# -----------------------------
st.set_page_config(page_title="AI Task System — Automated", layout="wide")
st.title("AI-Powered Automated Task & Manager System")

# -----------------------------
# PINECONE INIT (expects secrets.PINECONE_API_KEY)
# -----------------------------
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
if not PINECONE_API_KEY:
    st.error("PINECONE_API_KEY missing in st.secrets. Add it in .streamlit/secrets.toml")
    st.stop()

pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "task"
DIMENSION = 1024

# create index if missing
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
    st.error(f"Pinecone init error: {e}")
    st.stop()

# -----------------------------
# HELPERS
# -----------------------------
def now(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def stable_vec(seed_text):
    """Deterministic vector based on seed_text so updates stay near same spot."""
    rng = np.random.default_rng(abs(hash(seed_text)) % (2**32))
    return rng.random(DIMENSION).tolist()

def safe_meta(md):
    clean = {}
    for k, v in md.items():
        if v is None:
            v = ""
        elif isinstance(v, (datetime, date)):
            v = v.isoformat()
        clean[k] = v
    return clean

def to_df_from_matches(matches):
    rows = []
    for m in matches:
        md = m.metadata or {}
        md["_id"] = m.id
        rows.append(md)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.columns = [c.lower() for c in df.columns]
    return df

def fetch_filtered(filter_dict=None, top_k=10000):
    """Fetch filtered results from Pinecone deterministically."""
    try:
        res = index.query(
            vector=np.zeros(DIMENSION).tolist(),
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict or {}
        )
        return to_df_from_matches(res.matches or [])
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def fetch_all(top_k=10000):
    """Fetch all results (admin use)."""
    try:
        res = index.query(
            vector=np.zeros(DIMENSION).tolist(),
            top_k=top_k,
            include_metadata=True
        )
        return to_df_from_matches(res.matches or [])
    except Exception as e:
        st.error(f"Error fetching all data: {e}")
        return pd.DataFrame()

# -----------------------------
# SIMPLE AI MODELS (fallbacks)
# -----------------------------
lin_reg = LinearRegression()
lin_reg.fit([[0], [50], [100]], [0, 2.5, 5])

log_reg = LogisticRegression()
log_reg.fit([[0], [40], [80], [100]], [0, 0, 1, 1])

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
# NOTIFICATION UTILITIES (secure via st.secrets)
# -----------------------------
def send_email_smtp(to_email, subject, body):
    """Send email via SMTP - credentials read from st.secrets['email']"""
    email_conf = st.secrets.get("email", {})
    if not email_conf:
        st.warning("Email config missing in secrets; cannot send email.")
        return False
    try:
        msg = MIMEMultipart()
        msg["From"] = email_conf["SENDER_EMAIL"]
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP(email_conf.get("SMTP_SERVER","smtp.gmail.com"), int(email_conf.get("SMTP_PORT",587))) as server:
            server.starttls()
            server.login(email_conf["SENDER_EMAIL"], email_conf["APP_PASSWORD"])
            server.send_message(msg)
        return True
    except Exception as e:
        st.warning(f"Email send failed: {e}")
        return False

def send_email_sendgrid(to_email, subject, body):
    """Optional SendGrid path (if you prefer API). Provide SENDGRID_API_KEY in secrets."""
    import requests
    key = st.secrets.get("SENDGRID_API_KEY")
    if not key:
        st.warning("SENDGRID_API_KEY missing; cannot use SendGrid.")
        return False
    url = "https://api.sendgrid.com/v3/mail/send"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    data = {
        "personalizations": [{"to":[{"email": to_email}]}],
        "from": {"email": st.secrets.get("SENDGRID_FROM","noreply@example.com")},
        "subject": subject,
        "content": [{"type":"text/plain","value": body}]
    }
    r = requests.post(url, headers=headers, json=data)
    return r.status_code in (200,202)

def send_sms_twilio(to_number, body):
    tw = st.secrets.get("twilio", {})
    if not TWILIO_AVAILABLE or not tw:
        st.warning("Twilio not available or missing config; cannot send SMS.")
        return False
    try:
        client = TwilioClient(tw["TWILIO_ACCOUNT_SID"], tw["TWILIO_AUTH_TOKEN"])
        client.messages.create(body=body, from_=tw["TWILIO_FROM"], to=to_number)
        return True
    except Exception as e:
        st.warning(f"Twilio send failed: {e}")
        return False

# Wrapper to prefer SMTP, fallback to SendGrid
def send_email(to_email, subject, body):
    ok = send_email_smtp(to_email, subject, body)
    if not ok:
        ok = send_email_sendgrid(to_email, subject, body)
    if ok:
        st.success(f"Email queued to {to_email}")
    else:
        st.warning("Email not sent via any provider.")

def send_sms(to_number, body):
    ok = send_sms_twilio(to_number, body)
    if ok:
        st.success(f"SMS queued to {to_number}")

# -----------------------------
# FEEDBACK / SENTIMENT
# -----------------------------
def analyze_comment_advanced(text):
    text = (text or "").strip()
    if not text:
        return {"polarity": 0.0, "subjectivity": 0.0, "label": "Neutral", "emoji": "🟡", "level":"No comment"}
    if TEXTBLOB_AVAILABLE:
        tb = TextBlob(text)
        p = tb.sentiment.polarity
        s = tb.sentiment.subjectivity
        if p > 0.4:
            return {"polarity": p, "subjectivity": s, "label":"Excellent", "emoji":"🟢", "level":"Outstanding"}
        elif p < -0.25:
            return {"polarity": p, "subjectivity": s, "label":"Needs Improvement", "emoji":"🔴", "level":"Below expectations"}
        elif p > 0.1:
            return {"polarity": p, "subjectivity": s, "label":"Good", "emoji":"🟢", "level":"Good"}
        else:
            return {"polarity": p, "subjectivity": s, "label":"Neutral", "emoji":"🟡", "level":"Satisfactory"}
    else:
        try:
            pred = int(svm_clf.predict(vectorizer.transform([text]))[0])
            if pred == 1:
                return {"polarity": 0.6, "subjectivity":0.5, "label":"Excellent", "emoji":"🟢", "level":"Positive (fallback)"}
            else:
                return {"polarity": -0.3, "subjectivity":0.6, "label":"Needs Improvement", "emoji":"🔴", "level":"Negative (fallback)"}
        except Exception:
            return {"polarity":0.0,"subjectivity":0.0,"label":"Neutral","emoji":"🟡","level":"Unknown"}

# -----------------------------
# SHARED UI UTIL
# -----------------------------
def show_quick_stats(df):
    if df.empty:
        st.info("No tasks found for current selection.")
        return
    avg_marks = pd.to_numeric(df.get("marks",0)).mean()
    avg_comp = pd.to_numeric(df.get("completion",0)).mean()
    counts = len(df)
    c1,c2,c3 = st.columns(3)
    c1.metric("Tasks", counts)
    c2.metric("Avg Completion", f"{avg_comp:.1f}%")
    c3.metric("Avg Marks", f"{avg_marks:.2f}")

# -----------------------------
# ROLE SELECTION
# -----------------------------
role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client", "Admin"])
current_month = datetime.now().strftime("%B %Y")

# -----------------------------
# MANAGER DASHBOARD (MASTER)
# -----------------------------
if role == "Manager":
    st.header("Manager — Actions & 360°")
    tab1, tab2, tab3 = st.tabs(["Assign Task", "Review & Adjust", "Manager Actions & 360°"])

    # Assign Task
    with tab1:
        st.subheader("Assign New Task")
        with st.form("assign_task"):
            company = st.text_input("Company")
            employee = st.text_input("Employee Name")
            task = st.text_input("Task Title")
            desc = st.text_area("Description")
            deadline = st.date_input("Deadline", value=date.today())
            month = st.text_input("Month", value=current_month)
            submit = st.form_submit_button("Create Task")
            if submit:
                if not (company and employee and task):
                    st.warning("Company, employee and task title required.")
                else:
                    tid = str(uuid.uuid4())
                    md = safe_meta({
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
                        "assigned_on": now(),
                        "escalated": False,
                        "approvals": {},
                        "manager_actions": []
                    })
                    index.upsert([{"id": tid, "values": stable_vec(task), "metadata": md}])
                    st.success(f"Task '{task}' assigned to {employee} at {company}.")

    # Review & Adjust
    with tab2:
        st.subheader("Boss Review & Adjustment")
        company_filter = st.text_input("Filter by Company (optional)")
        if company_filter:
            df = fetch_filtered({"company": {"$eq": company_filter}})
        else:
            df = pd.DataFrame()
        show_quick_stats(df)
        if df.empty:
            st.info("No tasks loaded for review.")
        else:
            for _, r in df.iterrows():
                st.markdown("----")
                st.markdown(f"### {r.get('task')}")
                left, right = st.columns([2,1])
                with left:
                    st.write(f"Employee: {r.get('employee')}")
                    st.write(f"Completion: {r.get('completion',0)}%")
                    st.write(f"Marks: {r.get('marks',0)}")
                    st.write(f"Deadline Risk: {r.get('deadline_risk','N/A')}")
                    st.write("Manager Actions:")
                    for a in r.get("manager_actions", []) or []:
                        st.write(f"- {a.get('type')} ({a.get('on')}): {a.get('note')}")
                with right:
                    adjusted_completion = st.slider("Adjust Completion", 0, 100, int(r.get("completion",0)), key=f"adj_{r.get('_id')}")
                    adjusted_marks = float(lin_reg.predict([[adjusted_completion]])[0])
                    comments = st.text_area("Boss Comments", key=f"c_{r.get('_id')}")
                    approve = st.selectbox("Approve?", ["No","Yes"], key=f"app_{r.get('_id')}")
                    reassign_to = st.text_input("Reassign to (optional)", key=f"re_{r.get('_id')}")
                    escalate = st.checkbox("Escalate", key=f"esc_{r.get('_id')}")
                    note_type = st.selectbox("Send Note Type", ["None","Appreciation","Warning","Suggestion"], key=f"nt_{r.get('_id')}")
                    note_text = st.text_area("Note text (optional)", key=f"ntxt_{r.get('_id')}", height=80)
                    if st.button("Finalize Review", key=f"final_{r.get('_id')}"):
                        sinfo = analyze_comment_advanced(comments)
                        md = dict(r)
                        md.update({
                            "completion": int(adjusted_completion),
                            "marks": float(adjusted_marks),
                            "manager_comments": comments,
                            "reviewed": True,
                            "sentiment": sinfo["label"],
                            "sentiment_pol": float(sinfo.get("polarity",0.0)),
                            "approved_by_boss": (approve == "Yes"),
                            "reviewed_on": now()
                        })
                        # reassign
                        if reassign_to:
                            old = md.get("employee")
                            md["employee"] = reassign_to
                            md.setdefault("manager_actions", []).append({"type":"Reassigned","note":f"From {old} to {reassign_to}","by":"Manager","on":now()})
                            # notify HR and new assignee if contact provided in secrets
                            hr_email = st.secrets.get("notifications",{}).get("HR_EMAIL")
                            if hr_email:
                                send_email(hr_email, f"Task reassigned: {md.get('task')}", f"Task '{md.get('task')}' reassigned from {old} to {reassign_to}.")
                        # escalate
                        if escalate:
                            md["escalated"] = True
                            md.setdefault("manager_actions", []).append({"type":"Escalated","note":note_text or "Escalated by manager","by":"Manager","on":now()})
                            boss_email = st.secrets.get("notifications",{}).get("BOSS_EMAIL")
                            if boss_email:
                                send_email(boss_email, f"Task escalated: {md.get('task')}", f"Task '{md.get('task')}' escalated.")
                        # send note
                        if note_type and note_type != "None":
                            md.setdefault("manager_actions", []).append({"type":note_type,"note":note_text or note_type,"by":"Manager","on":now()})
                        # upsert
                        index.upsert([{"id": md["_id"], "values": stable_vec(md.get("task","")), "metadata": safe_meta(md)}])
                        # notify employee if configured
                        notify_email = st.secrets.get("notifications",{}).get("EMPLOYEE_EMAIL")
                        notify_phone = st.secrets.get("notifications",{}).get("EMPLOYEE_PHONE")
                        summary_msg = f"Task '{md.get('task')}' reviewed by manager. Completion set {md.get('completion')}%."
                        if notify_email:
                            send_email(notify_email, f"Task reviewed: {md.get('task')}", summary_msg)
                        if notify_phone:
                            send_sms(notify_phone, summary_msg)
                        st.success(f"Review finalized for {md.get('task')} — {sinfo['label']}")
                        st.experimental_rerun()

    # Manager Actions & 360°
    with tab3:
        st.subheader("Quick Managerial Actions")
        q_task = st.text_input("Task Title (optional)")
        q_emp = st.text_input("Employee name (optional)")
        if st.button("Load for Action"):
            if q_task:
                df_load = fetch_filtered({"task": {"$eq": q_task}})
            elif q_emp:
                df_load = fetch_filtered({"employee": {"$eq": q_emp}})
            else:
                st.warning("Enter task title or employee.")
                df_load = pd.DataFrame()
            if df_load.empty:
                st.info("No matching tasks found.")
            else:
                st.session_state["action_tasks"] = df_load.to_dict("records")
                st.success(f"Loaded {len(df_load)} tasks for action.")
        for r in st.session_state.get("action_tasks", []):
            st.markdown("----")
            st.markdown(f"### {r.get('task')} — {r.get('employee')} ({r.get('company')})")
            col1,col2,col3 = st.columns(3)
            with col1:
                new_emp = st.text_input("Reassign to", key=f"r2_{r.get('_id')}")
                if st.button("Reassign Now", key=f"rbtn_{r.get('_id')}"):
                    md = dict(r)
                    old = md.get("employee")
                    md["employee"] = new_emp
                    md.setdefault("manager_actions", []).append({"type":"Reassigned","note":f"From {old} to {new_emp}","by":"Manager","on":now()})
                    index.upsert([{"id": md["_id"], "values": stable_vec(md.get("task","")), "metadata": safe_meta(md)}])
                    # notifications (HR & new emp if mapped)
                    hr = st.secrets.get("notifications",{}).get("HR_EMAIL")
                    if hr:
                        send_email(hr, "Task Reassigned", f"{md.get('task')} reassigned {old} -> {new_emp}")
                    st.success("Reassigned.")
                    st.experimental_rerun()
            with col2:
                if st.button("Escalate", key=f"esc2_{r.get('_id')}"):
                    md = dict(r)
                    md["escalated"] = True
                    md.setdefault("manager_actions", []).append({"type":"Escalated","note":"Escalated by Manager","by":"Manager","on":now()})
                    index.upsert([{"id": md["_id"], "values": stable_vec(md.get("task","")), "metadata": safe_meta(md)}])
                    boss = st.secrets.get("notifications",{}).get("BOSS_EMAIL")
                    if boss:
                        send_email(boss, "Task Escalated", f"{md.get('task')} escalated by manager.")
                    st.success("Escalated.")
                    st.experimental_rerun()
            with col3:
                note_t = st.selectbox("Note type", ["Appreciation","Warning","Suggestion"], key=f"nt3_{r.get('_id')}")
                note_txt = st.text_input("Note text", key=f"nt3txt_{r.get('_id')}")
                if st.button("Send Note", key=f"sn_{r.get('_id')}"):
                    md = dict(r)
                    md.setdefault("manager_actions", []).append({"type":note_t,"note":note_txt,"by":"Manager","on":now()})
                    index.upsert([{"id": md["_id"], "values": stable_vec(md.get("task","")), "metadata": safe_meta(md)}])
                    st.success("Note added.")
                    st.experimental_rerun()

        st.markdown("----")
        st.subheader("Generate 360° Performance Summaries")
        period = st.text_input("Month/Period (e.g., 'November 2025')", value=current_month)
        company_for_360 = st.text_input("Company for 360° (optional)")
        if st.button("Generate 360° Summaries"):
            if company_for_360:
                df_all = fetch_filtered({"company": {"$eq": company_for_360}})
            else:
                df_all = fetch_all()
            if df_all.empty:
                st.info("No data for 360° generation.")
            else:
                df_all["marks"] = pd.to_numeric(df_all.get("marks",0), errors="coerce")
                df_all["completion"] = pd.to_numeric(df_all.get("completion",0), errors="coerce")
                summaries = []
                for emp, group in df_all.groupby("employee"):
                    avg_marks = float(group["marks"].mean() or 0)
                    avg_comp = float(group["completion"].mean() or 0)
                    completed_high = len(group[group["completion"]>=90])
                    reviewed = len(group[group.get("reviewed","") == True])
                    pos = len(group[group.get("sentiment","") == "Excellent"])
                    neg = len(group[group.get("sentiment","") == "Needs Improvement"])
                    tone = "Balanced"
                    if pos > neg + 1: tone = "Strong positive feedback"
                    elif neg > pos + 1: tone = "Concerns flagged"
                    summary_text = (f"{emp} — Avg Marks: {avg_marks:.2f}, Avg Completion: {avg_comp:.1f}%. "
                                    f"High-quality tasks: {completed_high}. Reviewed: {reviewed}. Tone: {tone}.")
                    summaries.append({"employee":emp,"avg_marks":avg_marks,"avg_comp":avg_comp,
                                      "high_quality":completed_high,"reviewed":reviewed,
                                      "pos_feedback":pos,"neg_feedback":neg,"summary":summary_text})
                s_df = pd.DataFrame(summaries).sort_values("avg_marks", ascending=False)
                st.dataframe(s_df)
                st.download_button("Download 360° CSV", s_df.to_csv(index=False), "360_summaries.csv", "text/csv")

# -----------------------------
# TEAM MEMBER
# -----------------------------
elif role == "Team Member":
    st.header("Team Member — Progress Update")
    company = st.text_input("Company")
    employee = st.text_input("Your Name")
    if st.button("Load Tasks"):
        if not (company and employee):
            st.warning("Enter company and your name.")
        else:
            df = fetch_filtered({"company": {"$eq": company}, "employee": {"$eq": employee}})
            if df.empty:
                st.info("No tasks found.")
            else:
                st.session_state["tasks"] = df.to_dict("records")
                st.success(f"Loaded {len(df)} tasks.")
    for r in st.session_state.get("tasks", []):
        st.markdown("----")
        st.subheader(r.get("task"))
        st.write(r.get("description"))
        curr = float(r.get("completion",0) or 0)
        new = st.slider("Completion", 0, 100, int(curr), key=f"tm_{r.get('_id')}")
        hours_over = st.number_input("Overtime hours", min_value=0.0, value=float(r.get("overtime",0) or 0), step=0.5, key=f"ot_{r.get('_id')}")
        leave_flag = st.checkbox("Request leave related to this task", key=f"lv_{r.get('_id')}")
        if st.button("Submit Update", key=f"sub_{r.get('_id')}"):
            marks = float(lin_reg.predict([[new]])[0])
            track = "On Track" if log_reg.predict([[new]])[0] == 1 else "Delayed"
            miss = rf.predict([[new, 0]])[0]
            md = dict(r)
            md.update({
                "completion": int(new),
                "marks": float(marks),
                "status": track,
                "deadline_risk": "High" if miss else "Low",
                "submitted_on": now(),
                "client_reviewed": False,
                "overtime": float(hours_over),
                "leave_requested": bool(leave_flag)
            })
            index.upsert([{"id": md["_id"], "values": stable_vec(md.get("task","")), "metadata": safe_meta(md)}])
            st.success("Update submitted.")
            st.experimental_rerun()

# -----------------------------
# CLIENT
# -----------------------------
elif role == "Client":
    st.header("Client — Review Deliverables")
    company = st.text_input("Company Name")
    if st.button("Load Reviewed Tasks"):
        if not company:
            st.warning("Enter company name.")
        else:
            df = fetch_filtered({"company": {"$eq": company}, "reviewed": {"$eq": True}})
            if df.empty:
                st.info("No reviewed tasks.")
            else:
                st.session_state["ctasks"] = df.to_dict("records")
                st.success(f"Loaded {len(df)} reviewed tasks.")
    for r in st.session_state.get("ctasks", []):
        st.markdown("----")
        st.subheader(r.get("task"))
        st.write(f"Employee: {r.get('employee')}")
        st.write(f"Completion: {r.get('completion')}%")
        st.write(f"Marks: {r.get('marks')}")
        client_comment = st.text_area("Feedback")
        decision = st.selectbox("Decision", ["Approve Deliverable","Approve with changes","Reject"], key=f"cd_{r.get('_id')}")
        if st.button("Submit Client Decision", key=f"csub_{r.get('_id')}"):
            md = dict(r)
            md.update({
                "client_reviewed": True,
                "client_comments": client_comment,
                "client_approved_on": now(),
                "client_decision": decision
            })
            index.upsert([{"id": md["_id"], "values": stable_vec(md.get("task","")), "metadata": safe_meta(md)}])
            st.success("Client decision submitted.")
            st.experimental_rerun()

# -----------------------------
# ADMIN
# -----------------------------
elif role == "Admin":
    st.header("Admin — Full Data & Analytics")
    df = fetch_all()
    if df.empty:
        st.warning("No data found.")
    else:
        df["marks"] = pd.to_numeric(df.get("marks",0), errors="coerce")
        df["completion"] = pd.to_numeric(df.get("completion",0), errors="coerce")
        st.subheader("Top Employees by Marks")
        top = df.groupby("employee")["marks"].mean().reset_index().sort_values("marks", ascending=False).head(10)
        st.dataframe(top)
        st.subheader("K-Means Clustering")
        if len(df) > 2:
            n_clusters = min(3, len(df))
            km = KMeans(n_clusters=n_clusters, n_init=10).fit(df[["completion","marks"]].fillna(0))
            df["cluster"] = km.labels_
            st.plotly_chart(px.scatter(df, x="completion", y="marks", color=df["cluster"].astype(str),
                                       hover_data=["employee","task"], title="Task Clusters"))
        else:
            st.info("Not enough data for clustering.")
        st.subheader("All tasks")
        st.dataframe(df.sort_values("assigned_on", ascending=False))
        st.download_button("Download All Tasks (CSV)", df.to_csv(index=False), "tasks_full.csv", "text/csv")

# -----------------------------
# END
# -----------------------------
