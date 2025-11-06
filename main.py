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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from twilio.rest import Client
import uuid
import warnings

# Suppress warnings from scikit-learn about n_init
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.cluster._kmeans")

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="🚀 AI-Driven Employee & Project Management", layout="wide")
st.title("🚀 AI-Driven Employee & Project Management System")
st.markdown("---")

# -----------------------------
# INITIALIZATION
# -----------------------------
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY")
EMAIL_SENDER = st.secrets.get("EMAIL_SENDER", "")
EMAIL_PASSWORD = st.secrets.get("EMAIL_PASSWORD", "")
SMS_SID = st.secrets.get("TWILIO_ACCOUNT_SID", "")
SMS_AUTH = st.secrets.get("TWILIO_AUTH_TOKEN", "")
SMS_FROM = st.secrets.get("TWILIO_PHONE_NUMBER", "")

if not PINECONE_API_KEY:
    st.error("PINECONE_API_KEY not found in Streamlit secrets. Please add it to your .streamlit/secrets.toml file.")
    st.stop()

pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "project-task-management" # Updated index name for broader scope
DIMENSION = 1024

if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    try:
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        st.success(f"Pinecone index '{INDEX_NAME}' created.")
    except Exception as e:
        st.error(f"Failed to create Pinecone index: {e}")
        st.stop()

index = pc.Index(INDEX_NAME)

def now(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def rand_vec(): return np.random.rand(DIMENSION).tolist()

# -----------------------------
# HELPERS
# -----------------------------
def safe_upsert(md_list):
    """
    Safely upserts a list of dictionaries (metadata) to Pinecone.
    Each dictionary must contain an 'id' key or one will be generated.
    """
    upsert_data = []
    for md in md_list:
        doc_id = str(md.get("_id", uuid.uuid4()))
        upsert_data.append({
            "id": doc_id,
            "values": rand_vec(), # In a real app, replace with actual embeddings
            "metadata": {k: v for k, v in md.items() if not k.startswith('_')} # Exclude internal keys like _id
        })
    try:
        index.upsert(vectors=upsert_data)
        return True
    except Exception as e:
        st.error(f"Pinecone upsert failed: {e}")
        return False

@st.cache_data(ttl=60) # Cache data for 60 seconds to reduce Pinecone calls
def fetch_all():
    """Fetches all documents (up to top_k) from Pinecone."""
    try:
        res = index.query(vector=rand_vec(), top_k=10000, include_metadata=True)
        rows = []
        for m in res.matches:
            md = m.metadata or {}
            md["_id"] = m.id
            rows.append(md)
        return pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"⚠️ Unable to fetch data from Pinecone: {e}")
        return pd.DataFrame()

def safe_rerun():
    st.rerun()

# -----------------------------
# NOTIFICATIONS
# -----------------------------
def send_email(to, subject, message):
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        st.warning("Email sender credentials not configured. Skipping email.")
        return False
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_SENDER
        msg["To"] = to
        msg["Subject"] = subject
        msg.attach(MIMEText(message, "plain"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        st.warning(f"Email failed: {e}. Check sender email, password, and 'less secure app access' settings if using Gmail.")
        return False

def send_sms(to, message):
    if not SMS_SID or not SMS_AUTH or not SMS_FROM:
        st.warning("Twilio SMS credentials not configured. Skipping SMS.")
        return False
    try:
        client = Client(SMS_SID, SMS_AUTH)
        if not to.startswith('+'):
            st.warning(f"SMS recipient number '{to}' should start with a '+' and country code.")
            return False
        client.messages.create(body=message, from_=SMS_FROM, to=to)
        return True
    except Exception as e:
        st.warning(f"SMS failed: {e}. Ensure Twilio credentials and 'to' number format are correct.")
        return False

def send_notification(email=None, phone=None, subject="Update", msg="Task update"):
    email_sent = False
    sms_sent = False
    if email:
        email_sent = send_email(email, subject, msg)
    if phone:
        sms_sent = send_sms(phone, msg)
    return email_sent or sms_sent

# -----------------------------
# SIMPLE MODELS (Cached for performance)
# -----------------------------
@st.cache_resource
def get_models():
    lin_reg_model = LinearRegression()
    lin_reg_model.fit(np.array([[0], [50], [100]]), np.array([0, 2.5, 5]))

    log_reg_model = LogisticRegression(solver='liblinear')
    log_reg_model.fit(np.array([[0], [40], [80], [100]]), np.array([0, 0, 1, 1]))

    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(np.array([[10, 2], [50, 1], [90, 0], [100, 0]]), [0, 1, 0, 0])

    vectorizer_model = CountVectorizer()
    X_train = vectorizer_model.fit_transform(["excellent work", "needs improvement", "bad performance", "great job", "average"])
    svm_clf_model = SVC(kernel='linear', random_state=42)
    svm_clf_model.fit(X_train, [1, 0, 0, 1, 0])

    return lin_reg_model, log_reg_model, rf_model, vectorizer_model, svm_clf_model

lin_reg, log_reg, rf, vectorizer, svm_clf = get_models()

# -----------------------------
# GLOBAL EMPLOYEE/DEPARTMENT DATA (For multi-person & reassign)
# This would ideally come from a separate employee database/Pinecone index
# For demonstration, we'll simulate it.
# -----------------------------
@st.cache_data
def get_mock_employees_data():
    # Simulate a small database of employees with emails, phones, departments
    # In a real system, you'd fetch this from another Pinecone index or DB
    employees = [
        {"name": "Alice Smith", "email": "alice@example.com", "phone": "+15551234001", "department": "Tech", "company": "Acme Inc."},
        {"name": "Bob Johnson", "email": "bob@example.com", "phone": "+15551234002", "department": "Design", "company": "Acme Inc."},
        {"name": "Charlie Brown", "email": "charlie@example.com", "phone": "+15551234003", "department": "HR", "company": "Acme Inc."},
        {"name": "Diana Prince", "email": "diana@example.com", "phone": "+15551234004", "department": "Sales", "company": "Acme Inc."},
        {"name": "Eve Adams", "email": "eve@example.com", "phone": "+15551234005", "department": "Tech", "company": "Acme Inc."},
        {"name": "Frank White", "email": "frank@example.com", "phone": "+15551234006", "department": "Design", "company": "Acme Inc."},
        {"name": "Grace Hopper", "email": "grace@example.com", "phone": "+15551234007", "department": "Tech", "company": "Globex Corp."},
        {"name": "Harry Potter", "email": "harry@example.com", "phone": "+15551234008", "department": "Marketing", "company": "Globex Corp."},
    ]
    return pd.DataFrame(employees)

mock_employees_df = get_mock_employees_data()

# -----------------------------
# ROLE SELECTION
# -----------------------------
role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client", "Admin"])
current_month = datetime.now().strftime("%B %Y")
st.sidebar.markdown(f"Current Month: **{current_month}**")
st.sidebar.markdown("---")
st.sidebar.info("💡 **Startup Pitch Highlight:** This system leverages AI for performance insights and Pinecone for flexible, scalable data storage, making it perfect for dynamic team and project management!")

# -----------------------------
# MANAGER DASHBOARD
# -----------------------------
if role == "Manager":
    st.header("👨‍💼 Manager Dashboard: Project & Task Oversight")
    tab1, tab2, tab3, tab4 = st.tabs(["Assign Tasks / Projects", "Review Tasks", "Reassign Tasks", "360° Overview"])

    # --- Assign Task / Project ---
    with tab1:
        st.subheader("📝 Assign New Task or Project")
        with st.form("assign_task_form"):
            company = st.text_input("Company Name", key="manager_company_assign")
            task_title = st.text_input("Task/Project Title", key="manager_task_title")
            task_desc = st.text_area("Description", key="manager_task_desc")
            deadline = st.date_input("Deadline", date.today(), key="manager_deadline")
            is_project = st.checkbox("Is this a Project (multiple assignees)?", key="is_project_checkbox")

            if is_project:
                # Allow selecting multiple employees and a department
                st.markdown("##### Assign to Multiple Team Members")
                # Filter employees by selected company for more relevant choices
                available_employees = mock_employees_df[mock_employees_df['company'] == company]['name'].tolist() if company else mock_employees_df['name'].tolist()
                assignees = st.multiselect("Select Team Members", options=available_employees, key="project_assignees")
                departments = mock_employees_df['department'].unique().tolist()
                department = st.selectbox("Department (optional)", options=[''] + departments, key="project_department")
            else:
                # Single employee assignment
                st.markdown("##### Assign to Single Team Member")
                available_employees = mock_employees_df[mock_employees_df['company'] == company]['name'].tolist() if company else mock_employees_df['name'].tolist()
                employee_name = st.selectbox("Employee Name", options=[''] + available_employees, key="single_employee_assign")
                # Fetch email/phone for the selected single employee
                selected_employee_info = mock_employees_df[mock_employees_df['name'] == employee_name].iloc[0] if employee_name else {}
                employee_email = selected_employee_info.get('email', '')
                employee_phone = selected_employee_info.get('phone', '')
                # Department is taken from the employee's mock data
                department = selected_employee_info.get('department', '')

            submit_assign = st.form_submit_button("Assign Task/Project")

            if submit_assign:
                if not company or not task_title or not deadline:
                    st.error("Company, Task/Project Title, and Deadline are required.")
                elif is_project and not assignees:
                    st.error("Please select at least one assignee for a project.")
                elif not is_project and not employee_name:
                    st.error("Please select an employee for a single task.")
                else:
                    tasks_to_upsert = []
                    if is_project:
                        for assignee in assignees:
                            assignee_info = mock_employees_df[mock_employees_df['name'] == assignee].iloc[0].to_dict()
                            md = {
                                "_id": str(uuid.uuid4()),
                                "type": "task", # Each part of a project is still a 'task'
                                "is_project_task": True, # New flag
                                "project_id": task_title.replace(" ", "_").lower() + "_" + str(uuid.uuid4())[:8], # A unique project ID
                                "project_title": task_title, # Parent project title
                                "company": company,
                                "employee": assignee_info['name'],
                                "email": assignee_info.get('email', ''),
                                "phone": assignee_info.get('phone', ''),
                                "department": assignee_info.get('department', department), # Use employee's dept or project dept
                                "task": task_title + f" ({assignee_info['name']}'s part)", # Task name indicates part of project
                                "description": task_desc,
                                "completion": 0, "marks": 0.0, "month": current_month,
                                "deadline": str(deadline), "status": "Assigned", "assigned_on": now(),
                                "reviewed": False, "client_reviewed": False
                            }
                            tasks_to_upsert.append(md)
                            send_notification(md['email'], md['phone'], f"Project Task Assigned: {md['task']}", f"You have been assigned a task for project '{task_title}' at {company}. Deadline: {deadline}")
                        st.success(f"✅ Project '{task_title}' assigned to {', '.join(assignees)} for {company}.")
                    else:
                        md = {
                            "_id": str(uuid.uuid4()),
                            "type": "task",
                            "is_project_task": False,
                            "company": company,
                            "employee": employee_name,
                            "email": employee_email,
                            "phone": employee_phone,
                            "department": department, # Use department from selected employee
                            "task": task_title,
                            "description": task_desc,
                            "completion": 0, "marks": 0.0, "month": current_month,
                            "deadline": str(deadline), "status": "Assigned", "assigned_on": now(),
                            "reviewed": False, "client_reviewed": False
                        }
                        tasks_to_upsert.append(md)
                        send_notification(employee_email, employee_phone, f"Task Assigned: {task_title}", f"You have been assigned a task: {task_title} at {company}. Deadline: {deadline}")
                        st.success(f"✅ Task '{task_title}' assigned to {employee_name} for {company}.")

                    if tasks_to_upsert and safe_upsert(tasks_to_upsert):
                        safe_rerun()

    # --- Review Tasks ---
    with tab2:
        st.subheader("📋 Review Employee Tasks and Project Progress")
        review_company = st.text_input("Company for Review", key="review_company_input_mgr")
        # Allow filtering by employee OR department
        review_filter_option = st.radio("Filter By:", ["Employee", "Department", "All Unreviewed"], key="review_filter_option_mgr", horizontal=True)

        employee_filter_name = ""
        department_filter_name = ""
        if review_filter_option == "Employee":
            available_employees = mock_employees_df[mock_employees_df['company'] == review_company]['name'].tolist() if review_company else mock_employees_df['name'].tolist()
            employee_filter_name = st.selectbox("Select Employee", options=[''] + available_employees, key="review_employee_input_mgr")
        elif review_filter_option == "Department":
            available_departments = mock_employees_df[mock_employees_df['company'] == review_company]['department'].unique().tolist() if review_company else mock_employees_df['department'].unique().tolist()
            department_filter_name = st.selectbox("Select Department", options=[''] + available_departments, key="review_department_input_mgr")

        if st.button("Load Tasks for Review", key="load_review_tasks_btn_mgr"):
            if not review_company:
                st.error("Please enter Company Name.")
            elif review_filter_option == "Employee" and not employee_filter_name:
                st.error("Please select an Employee.")
            elif review_filter_option == "Department" and not department_filter_name:
                st.error("Please select a Department.")
            else:
                try:
                    query_filter = {"company": {"$eq": review_company}, "type": {"$eq": "task"}, "reviewed": {"$eq": False}}
                    if review_filter_option == "Employee" and employee_filter_name:
                        query_filter["employee"] = {"$eq": employee_filter_name}
                    elif review_filter_option == "Department" and department_filter_name:
                        query_filter["department"] = {"$eq": department_filter_name}
                    # If "All Unreviewed" is selected, just use company and reviewed=False

                    res = index.query(vector=rand_vec(), top_k=500, include_metadata=True, filter=query_filter)
                    st.session_state["review_tasks"] = [(m.id, m.metadata) for m in res.matches]
                    if not st.session_state["review_tasks"]:
                        st.info("No unreviewed tasks found based on your filters.")
                except Exception as e:
                    st.error(f"Error loading tasks for review: {e}")

        if "review_tasks" in st.session_state and st.session_state["review_tasks"]:
            st.write(f"**Tasks awaiting review:**")
            for tid, r in st.session_state["review_tasks"]:
                st.markdown(f"#### 🧾 {r.get('task')} (Employee: {r.get('employee')}, Dept: {r.get('department', 'N/A')})")
                if r.get('is_project_task'):
                    st.write(f"**Part of Project:** {r.get('project_title')}")
                st.write(f"Description: {r.get('description')}")
                st.write(f"Current Completion: {r.get('completion', 0)}%")
                st.write(f"Deadline: {r.get('deadline')}")

                with st.form(f"review_form_{tid}"):
                    adj_completion = st.slider(f"Adjust Completion for {r.get('task')}", 0, 100, int(r.get("completion", 0)), key=f"adj_comp_{tid}")
                    manager_comments = st.text_area("Manager Comments", key=f"mgr_comments_{tid}")
                    approve_task = st.radio("Approve Task?", ["Yes", "No"], index=1 if not r.get("approved_by_boss", False) else 0, key=f"approve_{tid}")
                    submit_review = st.form_submit_button(f"Finalize Review for '{r.get('task')}'")

                    if submit_review:
                        marks = float(lin_reg.predict([[adj_completion]])[0])
                        sentiment = "Positive" if manager_comments and svm_clf.predict(vectorizer.transform([manager_comments]))[0] == 1 else "Negative" if manager_comments else "Neutral"

                        updated_md = {**r,
                                      "completion": adj_completion,
                                      "marks": marks,
                                      "sentiment": sentiment,
                                      "manager_comments": manager_comments,
                                      "approved_by_boss": approve_task == "Yes",
                                      "reviewed": True,
                                      "reviewed_on": now()}

                        if safe_upsert([updated_md]):
                            send_notification(
                                r.get("email"), r.get("phone"),
                                f"Task Reviewed: {r.get('task')}",
                                f"Your task '{r.get('task')}' has been reviewed. Completion: {adj_completion}%, Marks: {marks:.2f}, Sentiment: {sentiment}. Manager Comments: {manager_comments}"
                            )
                            st.success(f"✅ Reviewed '{r.get('task')}' ({sentiment}).")
                            safe_rerun()
        elif "review_tasks" in st.session_state:
            st.info("No tasks to review at this moment or all tasks have been approved.")

    # --- Reassign Tasks ---
    with tab3:
        st.subheader("🔁 Reassign Tasks")
        reassign_company = st.text_input("Company Name for Reassignment", key="reassign_company_input")
        original_employee = st.selectbox("Original Employee (optional)", options=[''] + mock_employees_df['name'].tolist(), key="reassign_original_employee")

        # Load all active tasks that can be reassigned
        if st.button("Load Reassignable Tasks", key="load_reassign_tasks_btn"):
            if not reassign_company:
                st.error("Please enter the Company Name.")
            else:
                try:
                    reassign_filter = {"company": {"$eq": reassign_company}, "type": {"$eq": "task"}, "status": {"$ne": "Completed"}} # Only non-completed tasks
                    if original_employee:
                        reassign_filter["employee"] = {"$eq": original_employee}

                    res = index.query(vector=rand_vec(), top_k=500, include_metadata=True, filter=reassign_filter)
                    st.session_state["reassign_tasks"] = [(m.id, m.metadata) for m in res.matches]
                    if not st.session_state["reassign_tasks"]:
                        st.info("No active tasks found for reassignment based on your filters.")
                except Exception as e:
                    st.error(f"Error loading tasks for reassignment: {e}")

        if "reassign_tasks" in st.session_state and st.session_state["reassign_tasks"]:
            st.write("**Select a task to reassign:**")
            selected_task_for_reassign_id = st.selectbox(
                "Choose Task",
                options=[t[0] for t in st.session_state["reassign_tasks"]],
                format_func=lambda tid: f"{[t[1]['task'] for t in st.session_state['reassign_tasks'] if t[0]==tid][0]} (Current: {[t[1]['employee'] for t in st.session_state['reassign_tasks'] if t[0]==tid][0]})",
                key="select_task_to_reassign"
            )

            if selected_task_for_reassign_id:
                original_task_md = next(t[1] for t in st.session_state["reassign_tasks"] if t[0] == selected_task_for_reassign_id)
                st.markdown(f"**Reassigning:** {original_task_md['task']} (Currently assigned to {original_task_md['employee']})")

                with st.form(f"reassign_form_{selected_task_for_reassign_id}"):
                    available_employees_for_reassign = mock_employees_df[mock_employees_df['company'] == original_task_md['company']]['name'].tolist()
                    new_employee_name = st.selectbox(
                        "New Assignee",
                        options=[''] + available_employees_for_reassign,
                        key=f"new_assignee_{selected_task_for_reassign_id}"
                    )
                    reassignment_reason = st.text_area("Reason for Reassignment", key=f"reassign_reason_{selected_task_for_reassign_id}")
                    submit_reassign = st.form_submit_button("Confirm Reassignment")

                    if submit_reassign:
                        if not new_employee_name:
                            st.error("Please select a new assignee.")
                        else:
                            new_assignee_info = mock_employees_df[mock_employees_df['name'] == new_employee_name].iloc[0].to_dict()
                            old_employee_name = original_task_md['employee']

                            # Update the existing task with new assignee details
                            reassigned_md = {
                                **original_task_md,
                                "employee": new_assignee_info['name'],
                                "email": new_assignee_info.get('email', ''),
                                "phone": new_assignee_info.get('phone', ''),
                                "department": new_assignee_info.get('department', original_task_md.get('department', '')), # Update department
                                "reassigned_from": old_employee_name,
                                "reassigned_on": now(),
                                "reassignment_reason": reassignment_reason,
                                "status": "Reassigned" # Update status
                            }

                            if safe_upsert([reassigned_md]):
                                st.success(f"✅ Task '{reassigned_md['task']}' successfully reassigned from {old_employee_name} to {new_employee_name}.")
                                send_notification(original_task_md['email'], original_task_md['phone'],
                                                  f"Task Reassigned: {original_task_md['task']}",
                                                  f"Your task '{original_task_md['task']}' has been reassigned to {new_employee_name}. Reason: {reassignment_reason}")
                                send_notification(new_assignee_info.get('email'), new_assignee_info.get('phone'),
                                                  f"New Task Assignment: {reassigned_md['task']}",
                                                  f"You have been assigned task '{reassigned_md['task']}' (reassigned from {old_employee_name}). Reason: {reassignment_reason}. Deadline: {reassigned_md['deadline']}")
                                safe_rerun()
        elif "reassign_tasks" in st.session_state:
            st.info("No tasks to reassign at the moment or please load tasks.")

    # --- 360° Overview ---
    with tab4:
        st.subheader("📊 360° Performance Overview (All Tasks & Projects)")
        df_all_data = fetch_all()

        if df_all_data.empty or "type" not in df_all_data.columns:
            st.info("No data available for overview yet.")
        else:
            df_tasks = df_all_data[df_all_data["type"] == "task"].copy()

            if not df_tasks.empty:
                df_tasks["marks"] = pd.to_numeric(df_tasks.get("marks", pd.Series()), errors="coerce").fillna(0)
                df_tasks["completion"] = pd.to_numeric(df_tasks.get("completion", pd.Series()), errors="coerce").fillna(0)
                df_tasks["department"] = df_tasks["department"].fillna("N/A") # Fill missing departments

                st.subheader("Employee Performance Clusters")
                if len(df_tasks) >= 3:
                    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
                    df_tasks["cluster"] = kmeans.fit_predict(df_tasks[["completion", "marks"]])
                    fig = px.scatter(df_tasks, x="completion", y="marks", color=df_tasks["cluster"].astype(str),
                                     hover_data=["employee", "task", "company", "status", "marks", "completion", "department", "is_project_task", "project_title"],
                                     title="Employee Performance Clusters (Completion vs. Marks)")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough data points (at least 3) to form performance clusters.")

                st.markdown(f"**Overall Average Marks**: {df_tasks['marks'].mean():.2f}")
                st.markdown(f"**Overall Average Completion**: {df_tasks['completion'].mean():.2f}%")

                st.subheader("Department-wise Performance")
                dep_avg = df_tasks.groupby("department")["marks"].mean().reset_index()
                st.plotly_chart(px.bar(dep_avg, x="department", y="marks", title="Average Marks by Department"), use_container_width=True)

                st.subheader("Employee Performance Table")
                st.dataframe(df_tasks[['employee', 'company', 'department', 'task', 'project_title', 'completion', 'marks', 'sentiment', 'status', 'reviewed_on', 'approved_by_boss']].sort_values(by='marks', ascending=False), use_container_width=True)

                st.subheader("Project-wise Overview")
                projects_df = df_tasks[df_tasks['is_project_task']].copy()
                if not projects_df.empty:
                    project_summary = projects_df.groupby('project_title').agg(
                        total_tasks=('task', 'count'),
                        avg_completion=('completion', 'mean'),
                        avg_marks=('marks', 'mean'),
                        status_counts=('status', lambda x: x.value_counts().to_dict()),
                        assignees=('employee', lambda x: list(x.unique()))
                    ).reset_index()
                    st.dataframe(project_summary, use_container_width=True)
                else:
                    st.info("No projects found yet.")
            else:
                st.info("No task data available for 360° performance overview.")

# -----------------------------
# TEAM MEMBER PORTAL
# -----------------------------
elif role == "Team Member":
    st.header("👷 Team Member Portal: My Projects & Tasks")
    tab1, tab2, tab3 = st.tabs(["My Tasks & Projects", "AI Feedback", "Submit Leave"])

    # --- My Tasks ---
    with tab1:
        st.subheader("✅ My Assigned Tasks and Project Parts")
        member_company = st.text_input("Company Name", key="member_company_task_tm")
        member_name = st.text_input("Your Name", key="member_name_task_tm")

        if st.button("Load My Tasks", key="load_my_tasks_btn_tm"):
            if not member_company or not member_name:
                st.error("Please enter both Company and Your Name to load tasks.")
            else:
                try:
                    res = index.query(
                        vector=rand_vec(),
                        top_k=200,
                        include_metadata=True,
                        filter={
                            "company": {"$eq": member_company},
                            "employee": {"$eq": member_name},
                            "type": {"$eq": "task"}
                        }
                    )
                    st.session_state["my_tasks_tm"] = [(m.id, m.metadata) for m in res.matches]
                    if not st.session_state["my_tasks_tm"]:
                        st.info("No tasks or project parts found for you.")
                except Exception as e:
                    st.error(f"Error loading your tasks: {e}")

        if "my_tasks_tm" in st.session_state and st.session_state["my_tasks_tm"]:
            for tid, md in st.session_state["my_tasks_tm"]:
                st.markdown(f"#### 🏷️ {md.get('task')} (Status: {md.get('status', 'N/A')})")
                if md.get('is_project_task'):
                    st.write(f"**Project:** {md.get('project_title')}")
                st.write(f"Description: {md.get('description')}")
                st.write(f"Deadline: {md.get('deadline')}")
                st.write(f"Assigned On: {md.get('assigned_on')}")
                if md.get('manager_comments'):
                    st.info(f"Manager's Comment: {md.get('manager_comments')}")
                if md.get('approved_by_boss'):
                    st.success("Manager Approved: Yes")
                else:
                    st.warning("Manager Approved: No")
                if md.get('reassigned_from'):
                    st.info(f"Reassigned from: {md.get('reassigned_from')} on {md.get('reassigned_on')}. Reason: {md.get('reassignment_reason', 'N/A')}")

                with st.form(f"task_update_form_{tid}"):
                    current_prog = int(md.get("completion", 0))
                    prog = st.slider(f"Update Completion for '{md.get('task')}'", 0, 100, current_prog, key=f"prog_slider_{tid}")
                    submit_task_update = st.form_submit_button(f"Submit Update for '{md.get('task')}'")

                    if submit_task_update:
                        marks = float(lin_reg.predict([[prog]])[0])
                        track = "On Track" if log_reg.predict([[prog]])[0] == 1 else "Delayed"
                        if prog == 100:
                            track = "Completed" # Mark as completed if 100%
                        md2 = {**md, "completion": prog, "marks": marks, "status": track, "submitted_on": now()}
                        if safe_upsert([md2]):
                            st.success(f"✅ Updated '{md.get('task')}' to {prog}% ({track}).")
                            safe_rerun()
        elif "my_tasks_tm" in st.session_state:
            st.info("No tasks found for you or please load your tasks.")

    # --- AI Feedback ---
    with tab2:
        st.subheader("🧠 AI-Powered Self-Assessment & Feedback Summarization")
        feedback_company = st.text_input("Company Name", key="feedback_company_input_tm")
        feedback_employee = st.text_input("Your Name", key="feedback_employee_input_tm")
        feedback_text = st.text_area("Enter your feedback or comments here for AI analysis:", key="feedback_text_area_tm")

        if st.button("Analyze Feedback", key="analyze_feedback_btn_tm"):
            if not feedback_text:
                st.error("Please enter some feedback to analyze.")
            else:
                blob = TextBlob(feedback_text)
                pol = blob.sentiment.polarity
                sub = blob.sentiment.subjectivity

                st.markdown(f"**Sentiment Analysis Results:**")
                if pol > 0.1:
                    st.success(f"Overall Sentiment: Positive 😊 (Polarity: {pol:.2f})")
                elif pol < -0.1:
                    st.error(f"Overall Sentiment: Negative 😞 (Polarity: {pol:.2f})")
                else:
                    st.info(f"Overall Sentiment: Neutral 😐 (Polarity: {pol:.2f})")

                st.write(f"**Subjectivity**: {sub:.2f} (0.0 is objective, 1.0 is subjective - how much is it an opinion vs fact)")
                if blob.noun_phrases:
                    st.write(f"**Key Noun Phrases**: {', '.join(blob.noun_phrases)}")
                else:
                    st.write("No specific noun phrases identified.")

                feedback_md = {
                    "_id": str(uuid.uuid4()),
                    "type": "feedback",
                    "company": feedback_company,
                    "employee": feedback_employee,
                    "feedback_text": feedback_text,
                    "sentiment_polarity": pol,
                    "sentiment_subjectivity": sub,
                    "analyzed_on": now()
                }
                if safe_upsert([feedback_md]):
                    st.success("Feedback analyzed and recorded.")

    # --- Submit Leave ---
    with tab3:
        st.subheader("🏖️ Submit Leave Request")
        leave_company = st.text_input("Company Name", key="leave_company_input_tm")
        leave_employee = st.text_input("Your Name", key="leave_employee_input_tm")
        start_date = st.date_input("Start Date", key="leave_start_date_tm")
        end_date = st.date_input("End Date", key="leave_end_date_tm")
        leave_reason = st.text_area("Reason for Leave", key="leave_reason_text_tm")

        if st.button("Submit Leave Request", key="submit_leave_btn_tm"):
            if not leave_company or not leave_employee or not leave_reason:
                st.error("Company, Your Name, and Reason for Leave are required.")
            elif start_date > end_date:
                st.error("End date cannot be before start date.")
            else:
                md = {
                    "_id": str(uuid.uuid4()),
                    "type": "leave",
                    "company": leave_company,
                    "employee": leave_employee,
                    "start_date": str(start_date),
                    "end_date": str(end_date),
                    "reason": leave_reason,
                    "status": "Pending",
                    "requested_on": now()
                }
                if safe_upsert([md]):
                    st.success(f"✅ Leave submitted for {leave_employee} from {start_date} to {end_date}. Status: Pending.")
                    safe_rerun()

# -----------------------------
# CLIENT REVIEW
# -----------------------------
elif role == "Client":
    st.header("🧾 Client Review Portal")
    client_company = st.text_input("Company Name", key="client_company_input_cl")

    if st.button("Load Completed Tasks & Projects for Review", key="load_client_tasks_btn_cl"):
        if not client_company:
            st.error("Please enter the Company Name to load tasks.")
        else:
            try:
                # Filter for tasks that are reviewed by manager and not yet client reviewed
                res = index.query(
                    vector=rand_vec(),
                    top_k=500,
                    include_metadata=True,
                    filter={
                        "company": {"$eq": client_company},
                        "type": {"$eq": "task"},
                        "reviewed": {"$eq": True},
                        "client_reviewed": {"$eq": False}
                    }
                )
                st.session_state["client_review_tasks_cl"] = [(m.id, m.metadata) for m in res.matches]
                if not st.session_state["client_review_tasks_cl"]:
                    st.info("No completed tasks awaiting client review for this company.")
            except Exception as e:
                st.error(f"Error loading tasks for client review: {e}")

    if "client_review_tasks_cl" in st.session_state and st.session_state["client_review_tasks_cl"]:
        st.write(f"**Tasks & Project Parts awaiting your review for {client_company}:**")
        for tid, md in st.session_state["client_review_tasks_cl"]:
            st.markdown(f"#### 🏷️ {md.get('task')} (Employee: {md.get('employee')})")
            if md.get('is_project_task'):
                st.write(f"**Project:** {md.get('project_title')}")
            st.write(f"Description: {md.get('description')}")
            st.write(f"Completion: {md.get('completion', 0)}%")
            st.write(f"Manager Marks: {md.get('marks', 0.0):.2f}")
            st.write(f"Manager Comments: {md.get('manager_comments', 'N/A')}")
            st.write(f"Manager Approved: {'Yes' if md.get('approved_by_boss') else 'No'}")

            with st.form(f"client_review_form_{tid}"):
                client_comment = st.text_area(f"Your Feedback for '{md.get('task')}'", key=f"cf_{tid}")
                approve_client = st.radio(f"Approve '{md.get('task')}'?", ["Yes", "No"], key=f"app_client_{tid}")
                submit_client_review = st.form_submit_button(f"Submit Client Review for '{md.get('task')}'")

                if submit_client_review:
                    updated_md = {
                        **md,
                        "client_reviewed": True,
                        "client_approved": approve_client == "Yes",
                        "client_comments": client_comment,
                        "client_approved_on": now()
                    }
                    if safe_upsert([updated_md]):
                        st.success(f"✅ Client review submitted for '{md.get('task')}'.")
                        safe_rerun()
    elif "client_review_tasks_cl" in st.session_state:
        st.info("No tasks to review at this moment or please load tasks.")

# -----------------------------
# ADMIN DASHBOARD
# -----------------------------
elif role == "Admin":
    st.header("🏢 Admin Dashboard: Global System Overview")
    st.markdown("Monitor and manage all aspects of employees, tasks, projects, and system health.")

    admin_df = fetch_all()

    if admin_df.empty:
        st.info("No data found in the system yet. Please assign tasks and submit updates.")
    else:
        admin_df["marks"] = pd.to_numeric(admin_df.get("marks", pd.Series()), errors="coerce").fillna(0)
        admin_df["completion"] = pd.to_numeric(admin_df.get("completion", pd.Series()), errors="coerce").fillna(0)
        admin_df["department"] = admin_df["department"].fillna("N/A") # Fill missing departments for tasks

        st.subheader("📊 System-wide Performance Metrics")
        tasks_df = admin_df[admin_df["type"] == "task"].copy()
        if not tasks_df.empty:
            avg_marks = tasks_df['marks'].mean()
            avg_completion = tasks_df['completion'].mean()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Tasks/Project Parts", len(tasks_df))
            with col2:
                st.metric("Average Marks (Tasks)", f"{avg_marks:.2f}")
            with col3:
                st.metric("Average Completion (Tasks)", f"{avg_completion:.2f}%")

            st.markdown("---")
            st.subheader("Department-wise Performance & Projects")
            dep_avg = tasks_df.groupby("department").agg(
                avg_marks=('marks', 'mean'),
                avg_completion=('completion', 'mean'),
                total_tasks=('task', 'count'),
                active_employees=('employee', lambda x: x.nunique())
            ).reset_index().sort_values(by='avg_marks', ascending=False)
            st.dataframe(dep_avg, use_container_width=True)

            st.subheader("🏅 Top Employees & Projects Overview")
            # Group by employee for average performance
            employee_performance = tasks_df.groupby('employee').agg(
                avg_marks=('marks', 'mean'),
                avg_completion=('completion', 'mean'),
                tasks_assigned=('task', 'count'),
                completed_tasks=('completion', lambda x: (x == 100).sum()),
                department=('department', 'first') # Assuming employee belongs to one department
            ).reset_index().sort_values(by='avg_marks', ascending=False)
            st.dataframe(employee_performance.head(10), use_container_width=True)

            st.subheader("Project Progress Summary")
            projects_data = tasks_df[tasks_df['is_project_task']].copy()
            if not projects_data.empty:
                project_progress_summary = projects_data.groupby('project_title').agg(
                    total_parts=('task', 'count'),
                    avg_completion=('completion', 'mean'),
                    avg_marks=('marks', 'mean'),
                    pending_parts=('status', lambda x: (x != 'Completed').sum()),
                    assigned_to=('employee', lambda x: ', '.join(x.unique()))
                ).reset_index().sort_values(by='avg_completion', ascending=False)
                st.dataframe(project_progress_summary, use_container_width=True)
            else:
                st.info("No projects recorded yet for summary.")

            st.subheader("📉 Task Status Distribution")
            task_status_counts = tasks_df["status"].value_counts().reset_index()
            task_status_counts.columns = ['Status', 'Count']
            st.plotly_chart(px.pie(task_status_counts, values='Count', names='Status', title='Task Distribution by Status'), use_container_width=True)

        else:
            st.info("No task data found for detailed admin analysis.")

        st.markdown("---")
        st.subheader("🏖️ Leave Requests & Management")
        leaves_df = admin_df[admin_df.get("type") == "leave"].copy()
        if not leaves_df.empty:
            st.dataframe(leaves_df[["employee", "company", "start_date", "end_date", "reason", "status", "requested_on"]].sort_values(by='requested_on', ascending=False), use_container_width=True)

            st.markdown("#### Manage Leave Requests")
            # Filter for pending leaves
            pending_leaves = leaves_df[leaves_df["status"] == "Pending"]
            if not pending_leaves.empty:
                with st.form("manage_leave_form_admin"):
                    selected_leave_id = st.selectbox(
                        "Select Leave Request to Update",
                        options=pending_leaves["_id"].tolist(),
                        format_func=lambda x: f"[{x[:8]}] {pending_leaves[pending_leaves['_id']==x]['employee'].iloc[0]} ({pending_leaves[pending_leaves['_id']==x]['start_date'].iloc[0]} to {pending_leaves[pending_leaves['_id']==x]['end_date'].iloc[0]})",
                        key="admin_select_leave_to_update"
                    )
                    new_status = st.radio("Set New Status", ["Approved", "Rejected"], key="admin_new_leave_status", index=0) # Default to Approved
                    update_leave_btn = st.form_submit_button("Update Leave Status")

                    if update_leave_btn and selected_leave_id:
                        leave_to_update = leaves_df[leaves_df["_id"] == selected_leave_id].iloc[0].to_dict()
                        leave_to_update["status"] = new_status
                        if safe_upsert([leave_to_update]):
                            st.success(f"Leave request {selected_leave_id[:8]}... updated to '{new_status}'.")
                            send_notification(
                                email=mock_employees_df[mock_employees_df['name'] == leave_to_update['employee']].iloc[0].get('email'),
                                phone=mock_employees_df[mock_employees_df['name'] == leave_to_update['employee']].iloc[0].get('phone'),
                                subject=f"Leave Request {new_status}",
                                msg=f"Your leave request from {leave_to_update.get('start_date')} to {leave_to_update.get('end_date')} has been {new_status}."
                            )
                            safe_rerun()
            else:
                st.info("No pending leave requests to manage.")
        else:
            st.info("No leave requests found.")

        st.markdown("---")
        st.subheader("💬 All Feedback Entries (For Trend Analysis)")
        feedback_df = admin_df[admin_df.get("type") == "feedback"].copy()
        if not feedback_df.empty:
            # Basic sentiment trend over time
            feedback_df['analyzed_on_date'] = pd.to_datetime(feedback_df['analyzed_on']).dt.date
            sentiment_trend = feedback_df.groupby('analyzed_on_date')['sentiment_polarity'].mean().reset_index()
            fig_sentiment = px.line(sentiment_trend, x='analyzed_on_date', y='sentiment_polarity', title='Average Sentiment Polarity Over Time')
            st.plotly_chart(fig_sentiment, use_container_width=True)

            st.dataframe(feedback_df[["employee", "company", "feedback_text", "sentiment_polarity", "sentiment_subjectivity", "analyzed_on"]].sort_values(by='analyzed_on', ascending=False), use_container_width=True)
        else:
            st.info("No feedback entries found.")
