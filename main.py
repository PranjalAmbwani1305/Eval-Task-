import streamlit as st
import pandas as pd
import numpy as np
import pinecone
from datetime import datetime, date
from textblob import TextBlob
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import random
import string

# ----------------------- INITIAL SETUP -----------------------

st.set_page_config("AI Task Management", layout="wide")
st.title("🤖 AI-Driven Employee Management System")

# Connect to Pinecone
pinecone.init(api_key="YOUR_PINECONE_API_KEY", environment="gcp-starter")
index_name = "ai-task-system"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=128)
index = pinecone.Index(index_name)

# ----------------------- UTILITIES -----------------------

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def rand_vec():
    return np.random.rand(128).tolist()

def safe_upsert(idx, metadata):
    try:
        idx.upsert([(metadata["_id"], rand_vec(), metadata)])
    except Exception as e:
        st.error(f"Error saving record: {e}")

def send_notification(email, phone, subject, msg):
    st.info(f"📧 Notification sent to {email}: {subject}")

def generate_id():
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

# ----------------------- ML MODELS -----------------------

lin_reg = LinearRegression()
lin_reg.fit([[0], [50], [100]], [0, 2.5, 5])

log_reg = LogisticRegression()
log_reg.fit([[0], [40], [80], [100]], [0, 0, 1, 1])

rf = RandomForestClassifier()
rf.fit(np.array([[10, 2], [50, 1], [90, 0], [100, 0]]), [1, 0, 0, 0])

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(["excellent work", "needs improvement", "bad performance", "great job", "average"])
svm_clf = SVC()
svm_clf.fit(X_train, [1, 0, 0, 1, 0])

# ----------------------- SIDEBAR MENU -----------------------

role = st.sidebar.radio("Select your role:", ["Manager", "Team Member", "Client", "Admin"])

# ============================================================
# ======================= MANAGER ============================
# ============================================================

if role == "Manager":
    st.header("👨‍💼 Manager Dashboard")
    tabs = st.tabs(["Assign Task", "Review Tasks", "Leave Requests", "360° Overview"])

    # ---------- ASSIGN TASK ----------
    with tabs[0]:
        st.subheader("🗂 Assign Task")
        company = st.text_input("🏢 Company Name")
        employee = st.text_input("👤 Employee Name")
        email = st.text_input("📧 Employee Email")
        phone = st.text_input("📞 Phone Number")
        task = st.text_input("📝 Task Title")
        desc = st.text_area("📘 Description")
        deadline = st.date_input("📅 Deadline")
        month = date.today().strftime("%B")

        if st.button("✅ Assign Task"):
            tid = generate_id()
            md = {
                "_id": tid,
                "type": "task",
                "company": company,
                "employee": employee,
                "email": email,
                "phone": phone,
                "task": task,
                "description": desc,
                "deadline": deadline.isoformat(),
                "month": month,
                "completion": 0,
                "marks": 0,
                "status": "Assigned",
                "reviewed": False,
                "assigned_on": now()
            }
            safe_upsert(index, md)
            send_notification(email, phone, "New Task Assigned", f"{task} has been assigned.")
            st.success(f"✅ Task '{task}' assigned to {employee}")

    # ---------- REVIEW TASKS ----------
    with tabs[1]:
        st.subheader("🧾 Review Tasks")
        company = st.text_input("🏢 Company")
        res = index.query(vector=rand_vec(), top_k=500, include_metadata=True, filter={"company": {"$eq": company}})
        tasks = [m.metadata for m in res.matches if m.metadata.get("type") == "task"]
        df = pd.DataFrame(tasks)
        if not df.empty:
            for _, r in df.iterrows():
                st.markdown(f"### 🧩 {r.get('task')}")
                st.write(f"Assigned to: {r.get('employee')} | Deadline: {r.get('deadline')}")
                completion = st.slider(f"Completion % for {r.get('task')}", 0, 100, int(r.get("completion", 0)))
                marks = float(lin_reg.predict([[completion]])[0])
                approve = st.radio("Approve?", ["Yes", "No"], key=r["_id"])
                comments = st.text_area("💬 Manager Comments", key=f"c_{r['_id']}")
                if st.button(f"Finalize Review {r.get('task')}", key=f"f_{r['_id']}"):
                    sentiment_val = int(svm_clf.predict(vectorizer.transform([comments]))[0])
                    sentiment = "Positive" if sentiment_val == 1 else "Negative"
                    md = {
                        **r,
                        "completion": completion,
                        "marks": marks,
                        "reviewed": True,
                        "comments": comments,
                        "sentiment": sentiment,
                        "approved_by_boss": approve == "Yes",
                        "reviewed_on": now()
                    }
                    safe_upsert(index, md)
                    send_notification(r["email"], r["phone"], f"Task Review: {r['task']}", f"Completion: {completion}%, Marks: {marks:.2f}, Sentiment: {sentiment}")
                    st.success(f"✅ Review finalized for {r['task']} ({sentiment})")

    # ---------- LEAVE REQUESTS ----------
    with tabs[2]:
        st.subheader("📝 Leave Requests")
        res = index.query(vector=rand_vec(), top_k=500, include_metadata=True, filter={"type": {"$eq": "leave"}})
        df = pd.DataFrame([m.metadata for m in res.matches])
        if df.empty:
            st.info("No leave requests yet.")
        else:
            for _, r in df.iterrows():
                st.markdown(f"**{r['employee']}** | {r['reason']}")
                decision = st.radio("Approve?", ["Pending", "Approved", "Rejected"], key=f"leave_{r['_id']}")
                if st.button(f"Update {r['employee']} Leave", key=f"ul_{r['_id']}"):
                    r["status"] = decision
                    safe_upsert(index, r)
                    st.success(f"Updated leave: {decision}")

    # ---------- 360° OVERVIEW ----------
    with tabs[3]:
        st.subheader("📊 360° Overview")
        res = index.query(vector=rand_vec(), top_k=500, include_metadata=True)
        df = pd.DataFrame([m.metadata for m in res.matches])
        for c in ["marks", "completion"]:
            if c not in df.columns:
                df[c] = np.nan
        df["marks"] = pd.to_numeric(df["marks"], errors="coerce")
        df["completion"] = pd.to_numeric(df["completion"], errors="coerce")
        df = df.dropna(subset=["marks", "completion"])
        if df.empty:
            st.warning("No task data available.")
        else:
            kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
            df["cluster"] = kmeans.fit_predict(df[["completion", "marks"]])
            st.scatter_chart(df, x="completion", y="marks", color="cluster")
            st.dataframe(df[["employee", "marks", "completion", "cluster"]])

# ============================================================
# ======================= TEAM MEMBER ========================
# ============================================================

if role == "Team Member":
    st.header("👩‍💻 Team Member Portal")
    tabs = st.tabs(["My Tasks", "AI Feedback Summary", "Submit Leave"])

    # ---------- MY TASKS ----------
    with tabs[0]:
        company = st.text_input("🏢 Company")
        employee = st.text_input("👤 Your Name")
        res = index.query(vector=rand_vec(), top_k=500, include_metadata=True, filter={"company": {"$eq": company}, "employee": {"$eq": employee}})
        df = pd.DataFrame([m.metadata for m in res.matches])
        if df.empty:
            st.info("No tasks found.")
        else:
            st.dataframe(df[["task", "completion", "marks", "status"]])

    # ---------- AI FEEDBACK SUMMARY ----------
    with tabs[1]:
        st.subheader("🧠 AI Feedback Summarization (Automatic)")
        company = st.text_input("🏢 Company Name (for summary)", key="fb_company")
        employee = st.text_input("👤 Your Name (for summary)", key="fb_employee")
        if st.button("🔍 Load & Analyze Feedback"):
            res = index.query(vector=rand_vec(), top_k=500, include_metadata=True, filter={"company": {"$eq": company}, "employee": {"$eq": employee}, "reviewed": {"$eq": True}})
            records = []
            for m in res.matches:
                md = m.metadata
                records.append({
                    "task": md.get("task", "Unnamed"),
                    "manager_comments": md.get("comments", ""),
                    "client_comments": md.get("client_comments", "")
                })
            df_fb = pd.DataFrame(records)
            if df_fb.empty:
                st.warning("No feedback found.")
            else:
                feedback_combined = " ".join(df_fb["manager_comments"].fillna("") + " " + df_fb["client_comments"].fillna(""))
                blob = TextBlob(feedback_combined)
                polarity = blob.sentiment.polarity
                sentiment = "Positive" if polarity > 0.2 else "Negative" if polarity < -0.2 else "Neutral"
                st.metric("Overall Sentiment", sentiment)
                st.progress((polarity + 1) / 2)
                st.write("### Key Feedback Themes")
                st.write(", ".join(blob.noun_phrases) or "No clear themes.")
                st.success(f"Feedback summary for {employee} analyzed successfully!")

    # ---------- SUBMIT LEAVE ----------
    with tabs[2]:
        st.subheader("📝 Submit Leave")
        company = st.text_input("🏢 Company Name")
        employee = st.text_input("👤 Employee Name")
        reason = st.text_area("🗒 Reason for Leave")
        if st.button("📩 Submit Leave"):
            md = {"_id": generate_id(), "type": "leave", "company": company, "employee": employee, "reason": reason, "status": "Pending", "submitted_on": now()}
            safe_upsert(index, md)
            st.success("Leave request submitted.")

# ============================================================
# ======================= CLIENT =============================
# ============================================================

if role == "Client":
    st.header("💼 Client Dashboard")
    company = st.text_input("🏢 Company")
    res = index.query(vector=rand_vec(), top_k=500, include_metadata=True, filter={"company": {"$eq": company}, "type": {"$eq": "task"}})
    df = pd.DataFrame([m.metadata for m in res.matches])
    if df.empty:
        st.info("No tasks yet.")
    else:
        for _, r in df.iterrows():
            st.markdown(f"### 🧩 {r['task']} ({r['employee']})")
            comments = st.text_area("💬 Client Feedback", key=f"cl_{r['_id']}")
            approve = st.radio("Approve Task?", ["Yes", "No"], key=f"ca_{r['_id']}")
            if st.button(f"Submit Feedback {r['_id']}"):
                r["client_comments"] = comments
                r["client_approved"] = approve
                safe_upsert(index, r)
                st.success("Feedback submitted successfully!")

# ============================================================
# ======================= ADMIN ==============================
# ============================================================

if role == "Admin":
    st.header("🏛️ Admin Dashboard")
    res = index.query(vector=rand_vec(), top_k=500, include_metadata=True)
    df = pd.DataFrame([m.metadata for m in res.matches])
    if df.empty:
        st.info("No data found.")
    else:
        st.subheader("🏢 Department-wise Performance")
        if "department" not in df.columns:
            df["department"] = np.random.choice(["HR", "Tech", "Finance", "Marketing"], len(df))
