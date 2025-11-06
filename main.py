import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import pandas as pd
from datetime import datetime, date
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from textblob import TextBlob
import plotly.express as px
import uuid

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Task Management System", layout="wide")
st.title("🚀 AI-Powered Task Management System")

# -----------------------------
# INITIALIZATION
# -----------------------------
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "task"
DIMENSION = 1024

if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(INDEX_NAME)

def now(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def rand_vec(): return np.random.rand(DIMENSION).tolist()

# -----------------------------
# SAFETY HELPERS
# -----------------------------
def safe_upsert(index, md):
    try:
        index.upsert([{
            "id": str(md.get("_id", uuid.uuid4())),
            "values": rand_vec(),
            "metadata": md
        }])
    except Exception as e:
        st.error(f"Pinecone upsert failed: {e}")

def fetch_all():
    try:
        res = index.query(vector=rand_vec(), top_k=1000, include_metadata=True)
        rows = []
        for m in res.matches:
            md = m.metadata or {}
            md["_id"] = m.id
            rows.append(md)
        return pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"⚠️ Unable to fetch data: {e}")
        return pd.DataFrame()

def fetch_by_filter(filter_obj):
    try:
        res = index.query(vector=rand_vec(), top_k=1000, include_metadata=True, filter=filter_obj)
        rows = []
        for m in res.matches:
            md = m.metadata or {}
            md["_id"] = m.id
            rows.append(md)
        return pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"⚠️ Unable to fetch data: {e}")
        return pd.DataFrame()

# -----------------------------
# SIMPLE MODELS
# -----------------------------
lin_reg = LinearRegression()
lin_reg.fit([[0], [50], [100]], [0, 2.5, 5])
log_reg = LogisticRegression()
log_reg.fit([[0], [40], [80], [100]], [0, 0, 1, 1])
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(["excellent work", "needs improvement", "bad performance", "great job", "average"])
svm_clf = SVC()
svm_clf.fit(X_train, [1, 0, 0, 1, 0])
rf = RandomForestClassifier()
rf.fit(np.array([[10, 2], [50, 1], [90, 0], [100, 0]]), [0, 1, 0, 0])

# -----------------------------
# ROLE SELECTION
# -----------------------------
role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client", "Admin"])
current_month = datetime.now().strftime("%B %Y")

# ======================================================
# MANAGER DASHBOARD
# ======================================================
if role == "Manager":
    st.header("👨‍💼 Manager Dashboard")
    tab1, tab2, tab3, tab4 = st.tabs(["Assign Task", "Review Tasks", "Leave Requests", "360° Overview"])

    # --- Assign Task ---
    with tab1:
        with st.form("assign"):
            company = st.text_input("Company Name")
            department = st.selectbox("Department", ["Engineering", "Product", "Design", "HR", "Sales", "Marketing"], index=0)
            employees = st.text_input("Employee Names (comma separated) — multiple members allowed")
            task = st.text_input("Task Title")
            desc = st.text_area("Description")
            deadline = st.date_input("Deadline", value=date.today())
            submit = st.form_submit_button("Assign Task")

            if submit and company and employees and task:
                emp_list = [e.strip() for e in employees.split(",") if e.strip()]
                tid = str(uuid.uuid4())
                # Create one record per employee assigned (keeps history per-employee) and also create a parent project record
                for emp in emp_list:
                    md = {
                        "_id": str(uuid.uuid4()),
                        "company": company,
                        "department": department,
                        "employee": emp,
                        "task": task,
                        "description": desc,
                        "deadline": deadline.isoformat(),
                        "month": current_month,
                        "completion": 0,
                        "marks": 0,
                        "status": "Assigned",
                        "reviewed": False,
                        "assigned_on": now(),
                        "type": "task",
                        "project_id": tid
                    }
                    safe_upsert(index, md)
                st.success(f"✅ Task '{task}' assigned to {len(emp_list)} member(s) in {department}.")

    # --- Review Tasks ---
    with tab2:
        st.subheader("📝 Review Tasks")
        df = fetch_all()
        if df.empty:
            st.warning("No tasks found.")
        else:
            company_f = st.text_input("Filter - Company (leave blank for all)")
            employee_f = st.text_input("Filter - Employee (leave blank for all)")
            if company_f or employee_f:
                df = df[
                    (df.get("company","").str.contains(company_f, case=False, na=False)) &
                    (df.get("employee","").str.contains(employee_f, case=False, na=False))
                ]

            for _, r in df[df.get("type")=="task"].iterrows():
                st.markdown(f"### {r.get('task', 'Unnamed Task')}  —  {r.get('employee')}")
                st.write(f"Department: {r.get('department','N/A')} | Deadline: {r.get('deadline','')}")
                adj = st.slider(f"Completion ({r.get('employee', '')})", 0, 100, int(r.get("completion", 0)), key=f"sl_{r.get('_id')}")
                adj_marks = float(lin_reg.predict([[adj]])[0])
                comments = st.text_area(f"Manager Comments ({r.get('task', '')})", key=f"c_{r.get('_id')}")
                approve = st.radio(f"Approve {r.get('task', '')}?", ["Yes", "No"], key=f"a_{r.get('_id')}")

                if st.button(f"Finalize Review {r.get('task', '')}", key=f"f_{r.get('_id')}"):
                    sentiment_val = int(svm_clf.predict(vectorizer.transform([comments]))[0])
                    sentiment = "Positive" if sentiment_val == 1 else "Negative"
                    md = {**r,
                          "completion": adj,
                          "marks": adj_marks,
                          "reviewed": True,
                          "comments": comments,
                          "sentiment": sentiment,
                          "approved_by_boss": approve == "Yes",
                          "reviewed_on": now()}
                    safe_upsert(index, md)
                    st.success(f"✅ Review finalized for {r.get('employee')} ({sentiment})")

    # --- Leave Requests (Manager Approval Panel) ---
    with tab3:
        st.subheader("🛎️ Leave Requests")
        company_f = st.text_input("Filter Leaves - Company (leave blank for all)", key="leave_comp_filter")
        employee_f = st.text_input("Filter Leaves - Employee (leave blank for all)", key="leave_emp_filter")
        # Query leaves (type == leave)
        try:
            filter_obj = {"type": {"$eq": "leave"}}
            if company_f:
                filter_obj["company"] = {"$eq": company_f}
            if employee_f:
                filter_obj["employee"] = {"$eq": employee_f}
            res = index.query(vector=rand_vec(), top_k=1000, include_metadata=True, filter=filter_obj)
            leaves = [(m.id, m.metadata) for m in res.matches or []]
            if not leaves:
                st.info("No leave requests found for the selected filters.")
            else:
                for lid, md in leaves:
                    status = md.get("status", "Pending")
                    st.markdown(f"### {md.get('employee')} — {md.get('company')}")
                    st.write(f"From: {md.get('start_date')}  To: {md.get('end_date')}")
                    st.write(f"Reason: {md.get('reason')}")
                    st.write(f"Status: **{status}**")
                    manager_comment = st.text_area(f"Manager Comment ({lid})", key=f"mc_{lid}")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"Approve Leave {lid}", key=f"approve_leave_{lid}"):
                            md2 = {**md,
                                   "status": "Approved",
                                   "approved_by_manager": True,
                                   "manager_comment": manager_comment,
                                   "approved_on": now()}
                            safe_upsert(index, md2)
                            st.success("✅ Leave approved.")
                    with col2:
                        if st.button(f"Reject Leave {lid}", key=f"reject_leave_{lid}"):
                            md2 = {**md,
                                   "status": "Rejected",
                                   "approved_by_manager": False,
                                   "manager_comment": manager_comment,
                                   "rejected_on": now()}
                            safe_upsert(index, md2)
                            st.warning("❌ Leave rejected.")
        except Exception as e:
            st.error(f"Error loading leaves: {e}")

    # --- 360° Overview ---
    with tab4:
        st.subheader("📊 360° Overview & Performance Summaries")
        df = fetch_all()
        if df.empty:
            st.info("No data yet.")
        else:
            # department heatmap & department-wise averages
            df_tasks = df[df.get("type")=="task"].copy()
            df_tasks["marks"] = pd.to_numeric(df_tasks.get("marks", 0), errors="coerce").fillna(0)
            df_tasks["completion"] = pd.to_numeric(df_tasks.get("completion", 0), errors="coerce").fillna(0)
            if not df_tasks.empty:
                dept_summary = df_tasks.groupby("department").agg({
                    "marks": "mean",
                    "completion": "mean",
                    "task": "count"
                }).reset_index().rename(columns={"task": "num_tasks"})
                st.markdown("#### Department Summary")
                st.dataframe(dept_summary)

                # Bar chart
                fig = px.bar(dept_summary, x="department", y=["marks", "completion"], barmode="group",
                             title="Department Average Marks & Completion")
                st.plotly_chart(fig, use_container_width=True)

                # Top employees by consistency (not just marks)
                st.markdown("#### Top Employees (Consistency + Sentiment)")
                emp_stats = df_tasks.groupby("employee").agg({
                    "marks": ["mean", "std"],
                    "completion": "mean"
                })
                emp_stats.columns = ["marks_mean", "marks_std", "completion_mean"]
                emp_stats = emp_stats.reset_index()
                # score = high mean marks, low std (consistent), high completion
                emp_stats["score"] = emp_stats["marks_mean"] * 0.5 + (1 / (1 + emp_stats["marks_std"])) * 0.3 + emp_stats["completion_mean"] * 0.2
                top_emps = emp_stats.sort_values("score", ascending=False).head(10)
                st.dataframe(top_emps[["employee", "marks_mean", "marks_std", "completion_mean", "score"]])

# ======================================================
# TEAM MEMBER PORTAL
# ======================================================
elif role == "Team Member":
    st.header("👷 Team Member Portal")
    tab1, tab2, tab3 = st.tabs(["My Tasks", "AI Feedback Summarization", "Submit Leave"])

    # --- My Tasks ---
    with tab1:
        company = st.text_input("Company")
        employee = st.text_input("Your Name")
        if st.button("Load Tasks"):
            try:
                res = index.query(vector=rand_vec(), top_k=500, include_metadata=True,
                                  filter={"company": {"$eq": company}, "employee": {"$eq": employee}, "type": {"$eq": "task"}})
                st.session_state["tasks"] = [(m.id, m.metadata) for m in res.matches or []]
                st.success(f"Loaded {len(st.session_state['tasks'])} tasks.")
            except Exception as e:
                st.error(f"Error loading tasks: {e}")

        for tid, md in st.session_state.get("tasks", []):
            st.subheader(md.get("task"))
            st.write(f"Department: {md.get('department','N/A')}")
            st.write(f"Status: {md.get('status', 'N/A')}")
            st.write(f"Completion: {md.get('completion', 0)}%")
            st.write(f"Deadline: {md.get('deadline', '')}")

    # --- AI Feedback Summarization ---
    with tab2:
        st.subheader("🧠 AI Feedback Summarization (Automatic)")

        company_fb = st.text_input("🏢 Company Name (for summary)", key="fb_company")
        employee_fb = st.text_input("👤 Your Name (for summary)", key="fb_employee")

        if st.button("🔍 Load & Analyze Feedback", key="analyze_feedback"):
            if not company_fb or not employee_fb:
                st.warning("⚠️ Please enter both company and employee name.")
            else:
                try:
                    res = index.query(
                        vector=rand_vec(),
                        top_k=500,
                        include_metadata=True,
                        filter={
                            "company": {"$eq": company_fb},
                            "employee": {"$eq": employee_fb},
                            "reviewed": {"$eq": True},
                            "type": {"$eq": "task"}
                        }
                    )
                    records = []
                    for m in res.matches or []:
                        md = m.metadata
                        records.append({
                            "task": md.get("task", "Unnamed Task"),
                            "manager_comments": md.get("comments", ""),
                            "client_comments": md.get("client_comments", "")
                        })
                    df = pd.DataFrame(records)
                    if df.empty:
                        st.warning("No feedback found for this employee yet.")
                    else:
                        feedback_combined = ""
                        for _, row in df.iterrows():
                            st.markdown(f"### 🧾 {row['task']}")
                            if row["manager_comments"]:
                                st.write(f"**Manager Feedback:** {row['manager_comments']}")
                            if row["client_comments"]:
                                st.write(f"**Client Feedback:** {row['client_comments']}")
                            feedback_combined += f"{row['manager_comments']} {row['client_comments']} "

                        blob = TextBlob(feedback_combined)
                        polarity = blob.sentiment.polarity
                        subjectivity = blob.sentiment.subjectivity

                        if polarity > 0.2:
                            sentiment = "🌟 Overall Positive Feedback"
                            suggestion = "Keep up the great work!"
                        elif polarity < -0.2:
                            sentiment = "⚠️ Overall Negative Feedback"
                            suggestion = "Focus on improvement areas mentioned."
                        else:
                            sentiment = "💬 Neutral Feedback"
                            suggestion = "Continue improving communication and delivery."

                        st.markdown("---")
                        st.markdown(f"### 📊 Sentiment Summary")
                        st.markdown(f"**Sentiment:** {sentiment}")
                        st.markdown(f"**Polarity:** `{polarity:.2f}` | **Subjectivity:** `{subjectivity:.2f}`")
                        st.progress((polarity + 1) / 2)
                        st.markdown(f"### 💡 AI Suggestion")
                        st.info(suggestion)
                        st.markdown("### 🧠 Key Themes")
                        st.write(", ".join(blob.noun_phrases))
                except Exception as e:
                    st.error(f"Error fetching feedback: {e}")

    # --- Submit Leave ---
    with tab3:
        st.subheader("🛌 Submit Leave Request")
        with st.form("submit_leave"):
            company_l = st.text_input("Company", key="leave_company")
            employee_l = st.text_input("Your Name", key="leave_employee")
            start_date = st.date_input("Start Date", value=date.today(), key="leave_start")
            end_date = st.date_input("End Date", value=date.today(), key="leave_end")
            leave_type = st.selectbox("Leave Type", ["Paid Leave", "Sick Leave", "Unpaid Leave", "Work From Home"])
            reason = st.text_area("Reason for Leave")
            submit_leave = st.form_submit_button("Submit Leave Request")

            if submit_leave:
                if not company_l or not employee_l or not reason:
                    st.warning("Please fill Company, Your Name and Reason.")
                else:
                    leave_id = str(uuid.uuid4())
                    md = {
                        "_id": leave_id,
                        "type": "leave",
                        "company": company_l,
                        "employee": employee_l,
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "leave_type": leave_type,
                        "reason": reason,
                        "status": "Pending",
                        "submitted_on": now()
                    }
                    safe_upsert(index, md)
                    st.success("✅ Leave request submitted. Manager will review it shortly.")

# ======================================================
# CLIENT DASHBOARD
# ======================================================
elif role == "Client":
    st.header("🧾 Client Review")
    company = st.text_input("Company Name")
    if st.button("Load Completed Tasks"):
        try:
            res = index.query(vector=rand_vec(), top_k=500, include_metadata=True,
                              filter={"company": {"$eq": company}, "reviewed": {"$eq": True}, "type": {"$eq": "task"}})
            st.session_state["ctasks"] = [(m.id, m.metadata) for m in res.matches or []]
            st.success(f"Loaded {len(st.session_state['ctasks'])} tasks.")
        except Exception as e:
            st.error(f"Error loading tasks: {e}")

    for tid, md in st.session_state.get("ctasks", []):
        st.subheader(md.get("task"))
        st.write(f"Employee: {md.get('employee')}")
        st.write(f"Final Completion: {md.get('completion')}%")
        st.write(f"Marks: {md.get('marks')}")
        comment = st.text_area(f"Client Feedback ({md.get('task')})", key=f"cf_{tid}")
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"Approve {md.get('task')}", key=f"app_{tid}"):
                md2 = {**md,
                       "client_reviewed": True,
                       "client_comments": comment,
                       "client_approved_on": now()}
                safe_upsert(index, md2)
                st.success(f"✅ Approved {md.get('task')}")
        with col2:
            if st.button(f"Request Revision {md.get('task')}", key=f"rev_{tid}"):
                md2 = {**md,
                       "client_reviewed": False,
                       "client_comments": comment,
                       "client_requested_revision": True,
                       "client_requested_on": now()}
                safe_upsert(index, md2)
                st.warning(f"🔁 Revision requested for {md.get('task')}")

# ======================================================
# ADMIN DASHBOARD (basic overview + department-wise performance)
# ======================================================
elif role == "Admin":
    st.header("🧑‍⚖️ Admin / HR Dashboard")
    st.subheader("Organization Overview")
    df = fetch_all()
    if df.empty:
        st.info("No data yet.")
    else:
        # Department-wise performance (tasks only)
        df_tasks = df[df.get("type")=="task"].copy()
        df_tasks["marks"] = pd.to_numeric(df_tasks.get("marks", 0), errors="coerce").fillna(0)
        df_tasks["completion"] = pd.to_numeric(df_tasks.get("completion", 0), errors="coerce").fillna(0)
        if not df_tasks.empty:
            dept_summary = df_tasks.groupby("department").agg({
                "marks": "mean",
                "completion": "mean",
                "task": "count"
            }).reset_index().rename(columns={"task": "num_tasks"})
            st.markdown("### Department-wise Performance")
            st.dataframe(dept_summary)

            # Visuals
            fig1 = px.bar(dept_summary, x="department", y=["marks", "completion"], barmode="group",
                          title="Department Average Marks & Completion")
            st.plotly_chart(fig1, use_container_width=True)

            # Heatmap of departments vs completion
            pivot = df_tasks.pivot_table(values="completion", index="department", columns="employee", aggfunc="mean").fillna(0)
            st.markdown("### Department × Employee Completion Heatmap (sample)")
            st.dataframe(pivot)

        # Leave summary
        df_leaves = df[df.get("type")=="leave"].copy()
        if not df_leaves.empty:
            st.markdown("### Leave Requests Summary")
            leave_stats = df_leaves.groupby(["company", "status"]).size().unstack(fill_value=0)
            st.dataframe(leave_stats)

            # Pending leaves table
            st.markdown("#### Pending Leaves")
            st.dataframe(df_leaves[df_leaves.get("status")=="Pending"][["company","employee","start_date","end_date","leave_type","reason","submitted_on"]])

    st.markdown("---")
    st.info("Use Manager role to approve leaves and review individual tasks.")

# End of app
