import streamlit as st
import numpy as np
import pandas as pd
import uuid, os
from datetime import date, datetime, timedelta
import plotly.express as px
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

# ----------------------------------------
# CONFIG
# ----------------------------------------
st.set_page_config(page_title="AI Enterprise Workforce System", layout="wide")
st.title("🏢 AI Enterprise Workforce & Task Management — Secure AI Edition")

# ----------------------------------------
# LOGIN SECURITY
# ----------------------------------------
users = {
    "Manager": {"username": "manager", "password": "admin123"},
    "Team Member": {"username": "team", "password": "emp123"},
    "Client": {"username": "client", "password": "client123"}
}

role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client"])
uname = st.sidebar.text_input("Username")
pwd = st.sidebar.text_input("Password", type="password")

if uname != users[role]["username"] or pwd != users[role]["password"]:
    st.warning("🔐 Please enter valid login credentials.")
    st.stop()

st.sidebar.success(f"✅ Logged in as {role}")

# ----------------------------------------
# LOCAL DB
# ----------------------------------------
if "data" not in st.session_state:
    st.session_state["data"] = []

def now_str(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def save_record(rec):
    st.session_state["data"].append(rec)

def get_records(filter_by=None):
    df = pd.DataFrame(st.session_state["data"])
    if filter_by:
        for k, v in filter_by.items():
            df = df[df[k] == v]
    return df if not df.empty else pd.DataFrame()

# ----------------------------------------
# SIMPLE MODELS
# ----------------------------------------
lin_reg = LinearRegression().fit([[0], [50], [100]], [0, 2.5, 5])
log_reg = LogisticRegression().fit([[0], [40], [80], [100]], [0, 0, 1, 1])
rf = RandomForestClassifier().fit(np.array([[10,2],[50,1],[90,0],[100,0]]), [1,0,0,0])
vec = CountVectorizer()
X = vec.fit_transform(["good job", "bad performance", "excellent work", "needs improvement", "great results"])
svm = SVC().fit(X, [1, 0, 1, 0, 1])

# ----------------------------------------
# MANAGER DASHBOARD
# ----------------------------------------
if role == "Manager":
    st.header("👨‍💼 Manager Dashboard — Enterprise Suite")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Assign / Reassign", "🧾 Review Tasks", "🏢 Inner Department", "🌐 360° Overview", "🏖 Leave Management"
    ])

    # ------------------ Assign / Reassign ------------------
    with tab1:
        st.subheader("Assign New Task")
        with st.form("assign_form"):
            company = st.text_input("🏢 Company Name")
            department = st.selectbox("🏬 Department", ["IT", "Finance", "HR", "Marketing", "Operations"])
            team = st.text_input("👥 Team Name")
            employee = st.text_input("👤 Employee Name")
            task = st.text_input("🧠 Task Title")
            desc = st.text_area("📝 Task Description")
            deadline = st.date_input("📅 Deadline", value=date.today() + timedelta(days=7))
            file = st.file_uploader("📎 Attach Proof (optional)")
            submit = st.form_submit_button("✅ Assign Task")

            if submit and company and employee and task:
                record = {
                    "id": str(uuid.uuid4()), "company": company, "department": department, "team": team,
                    "employee": employee, "task": task, "description": desc,
                    "completion": 0, "marks": 0, "status": "Assigned",
                    "deadline": deadline.isoformat(), "reviewed": False,
                    "assigned_on": now_str(), "sentiment": "N/A", "proof": file.name if file else None
                }
                save_record(record)
                st.success(f"✅ Task '{task}' assigned to {employee}")

        st.divider()
        st.subheader("♻️ Reassign Task")
        company_r = st.text_input("Company for Reassignment")
        emp_from = st.text_input("Current Employee")
        emp_to = st.text_input("New Employee")
        if st.button("🔁 Reassign"):
            df = get_records({"company": company_r, "employee": emp_from})
            if not df.empty:
                for i in df.index:
                    st.session_state["data"][i]["employee"] = emp_to
                    st.session_state["data"][i]["status"] = "Reassigned"
                st.success(f"Reassigned {len(df)} tasks from {emp_from} to {emp_to}")
            else:
                st.warning("No matching tasks found.")

    # ------------------ Review Tasks ------------------
    with tab2:
        st.subheader("🔍 Review Employee Tasks")
        company = st.text_input("Company to Review")
        if st.button("Load Tasks"):
            df = get_records({"company": company})
            if not df.empty:
                for i, r in df.iterrows():
                    st.write(f"### {r['employee']} — {r['task']}")
                    adj = st.slider("Completion %", 0, 100, int(r["completion"]), key=f"adj_{i}")
                    comments = st.text_area("Manager Comments", key=f"com_{i}")
                    if st.button(f"Finalize Review {i}"):
                        marks = float(lin_reg.predict([[adj]])[0])
                        status = "On Track" if log_reg.predict([[adj]])[0] == 1 else "Delayed"
                        sentiment = "Positive" if svm.predict(vec.transform([comments]))[0] == 1 else "Negative"
                        st.session_state["data"][i]["completion"] = adj
                        st.session_state["data"][i]["marks"] = marks
                        st.session_state["data"][i]["status"] = status
                        st.session_state["data"][i]["sentiment"] = sentiment
                        st.session_state["data"][i]["reviewed"] = True
                        st.success(f"✅ Reviewed {r['task']} ({sentiment})")
            else:
                st.warning("No tasks found.")

    # ------------------ Inner Department ------------------
    with tab3:
        st.subheader("🏢 Departmental Performance")
        df = get_records()
        if not df.empty:
            depts = df["department"].unique().tolist()
            dept = st.selectbox("Select Department", depts)
            ddf = df[df["department"] == dept]
            st.metric("👥 Employees", ddf["employee"].nunique())
            st.metric("🏆 Avg Marks", f"{ddf['marks'].mean():.2f}")
            st.metric("✅ Avg Completion", f"{ddf['completion'].mean():.1f}%")
            fig = px.bar(ddf, x="employee", y="marks", color="team", title=f"{dept} Department Performance")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available yet.")

    # ------------------ 360 Overview ------------------
    with tab4:
        st.subheader("🌐 360° Organizational Overview")
        df = get_records()
        if not df.empty:
            st.metric("Total Employees", df["employee"].nunique())
            st.metric("Average Marks", f"{df['marks'].mean():.2f}")
            st.metric("Average Completion", f"{df['completion'].mean():.1f}%")

            if "sentiment" in df.columns:
                sent = df["sentiment"].value_counts().reset_index()
                fig = px.pie(sent, names="index", values="sentiment", title="Sentiment Analysis Overview")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data to display.")

    # ------------------ Leave Management ------------------
    with tab5:
        st.subheader("🏖 Manage Leave Requests")
        df = get_records({"status": "Leave Applied"})
        if not df.empty:
            for i, l in df.iterrows():
                st.write(f"🧾 {l['employee']} | {l['leave_type']} ({l['from']} → {l['to']})")
                if st.button(f"Approve {l['employee']}", key=f"ap_{i}"):
                    st.session_state["data"][i]["status"] = "Leave Approved"
                    st.success(f"✅ Approved {l['employee']}'s leave")
        else:
            st.info("No pending leave requests.")

# ----------------------------------------
# TEAM MEMBER PORTAL
# ----------------------------------------
elif role == "Team Member":
    st.header("👩‍💻 Team Member Workspace")
    company = st.text_input("🏢 Company")
    employee = st.text_input("👤 Your Name")
    task = st.text_input("🧠 Task Title")
    completion = st.slider("✅ Completion %", 0, 100, 0)
    if st.button("📩 Submit Update"):
        marks = lin_reg.predict([[completion]])[0]
        status = "On Track" if log_reg.predict([[completion]])[0] == 1 else "Delayed"
        record = {
            "id": str(uuid.uuid4()), "company": company, "employee": employee, "task": task,
            "completion": completion, "marks": marks, "status": status,
            "reviewed": False, "submitted_on": now_str()
        }
        save_record(record)
        st.success(f"✅ Task update saved for {employee}")

    st.divider()
    st.subheader("🏖 Apply for Leave")
    leave_type = st.selectbox("Leave Type", ["Casual", "Sick", "Paid"])
    from_d = st.date_input("From")
    to_d = st.date_input("To", value=date.today() + timedelta(days=2))
    reason = st.text_area("Reason")
    if st.button("Apply Leave"):
        save_record({
            "id": str(uuid.uuid4()), "employee": employee, "leave_type": leave_type,
            "from": from_d.isoformat(), "to": to_d.isoformat(), "reason": reason, "status": "Leave Applied"
        })
        st.success("Leave application submitted!")

# ----------------------------------------
# CLIENT PORTAL
# ----------------------------------------
elif role == "Client":
    st.header("🧾 Client Project Portal")
    company = st.text_input("🏢 Company Name")
    if st.button("View Completed Projects"):
        df = get_records({"company": company, "reviewed": True})
        if not df.empty:
            for i, r in df.iterrows():
                st.markdown(
                    f"<div style='border:1px solid #ddd;border-radius:10px;padding:10px;margin:5px;background:#f9f9f9'>"
                    f"<b>{r['employee']}</b> — {r['task']}<br>"
                    f"✅ Completion: {r['completion']}% | Marks: {r['marks']:.2f}<br>"
                    f"💬 Sentiment: {r['sentiment']}</div>",
                    unsafe_allow_html=True
                )
            st.divider()
            st.subheader("⭐ Provide Feedback")
            rating = st.slider("Rate Overall Project", 1, 5, 4)
            comment = st.text_area("Add Comments")
            if st.button("Submit Feedback"):
                save_record({
                    "id": str(uuid.uuid4()), "company": company, "type": "Client Feedback",
                    "rating": rating, "comment": comment, "submitted_on": now_str()
                })
                st.success("✅ Feedback submitted successfully.")
        else:
            st.info("No completed tasks available.")

# ----------------------------------------
# FOOTER
# ----------------------------------------
st.markdown("---")
st.caption("🌟 Enterprise Edition — with login, 360° insights, and AI-driven workforce analytics.")
