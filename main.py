# ============================================================
# 🏢 AI Enterprise Workforce Management System — Professional Simple Edition
# ============================================================

import streamlit as st
import pandas as pd
import uuid
from datetime import date, datetime, timedelta
import random

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(page_title="AI Workforce System", layout="wide")
st.title("🏢 AI Enterprise Workforce — Final Professional Edition")

# ------------------------------------------------
# SESSION STORAGE
# ------------------------------------------------
if "records" not in st.session_state:
    st.session_state.records = pd.DataFrame()

# ------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------
def get_df():
    return st.session_state.records.copy()

def upsert_record(record):
    df = get_df()
    if "_id" not in record:
        record["_id"] = str(uuid.uuid4())
    # FIXED: Check if record exists correctly
    if not df.empty and "_id" in df.columns and record["_id"] in df["_id"].values:
        df.loc[df["_id"] == record["_id"], list(record.keys())] = list(record.values())
    else:
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    st.session_state.records = df

def seed_data():
    """Add sample data"""
    employees = ["Aarav", "Isha", "Rohan", "Amruta"]
    companies = ["TechNova", "InnoSoft"]
    data = []
    for i in range(6):
        completion = random.choice([10, 40, 70, 90])
        status = "✅ On Track" if completion >= 60 else "🕐 Pending"
        data.append({
            "_id": str(uuid.uuid4()),
            "record_type": "task",
            "company": random.choice(companies),
            "employee": random.choice(employees),
            "task": f"Task {100+i}",
            "completion": completion,
            "status": status,
            "reviewed": False
        })
    data.append({
        "_id": str(uuid.uuid4()),
        "record_type": "meeting",
        "meeting_title": "Weekly Sync",
        "organizer": "Manager",
        "participants": ", ".join(random.sample(employees, 3)),
        "meeting_datetime": str(datetime.now() + timedelta(days=1)),
        "agenda": "Project progress discussion"
    })
    st.session_state.records = pd.DataFrame(data)
    st.success("✅ Sample data added successfully!")

def filter_type(df, rtype):
    if df.empty or "record_type" not in df.columns:
        return pd.DataFrame()
    df["record_type"] = df["record_type"].fillna("").astype(str).str.lower()
    return df[df["record_type"] == rtype.lower()].copy()

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------
st.sidebar.header("🎚 Select Role")
role = st.sidebar.radio("Choose Role", ["Manager", "Team Member", "Client", "Admin"])
if st.sidebar.button("⚙️ Add Sample Data"):
    seed_data()

df = get_df()

# ------------------------------------------------
# STYLE HELPERS
# ------------------------------------------------
def color_status(val):
    if "✅" in str(val):
        return "background-color: #b7e1cd;"
    elif "🕐" in str(val):
        return "background-color: #ffe599;"
    elif "❌" in str(val):
        return "background-color: #f4cccc;"
    return ""

def summary_card(label, value, emoji):
    st.markdown(f"""
        <div style='display:inline-block; background:#f1f3f4; padding:15px 25px; margin:8px; border-radius:10px; text-align:center;'>
            <h3>{emoji} {label}</h3>
            <h2 style='color:#004aad'>{value}</h2>
        </div>
    """, unsafe_allow_html=True)

# ============================================================
# 👨‍💼 MANAGER
# ============================================================
if role == "Manager":
    st.header("👨‍💼 Manager Dashboard")
    tabs = st.tabs(["Assign Task", "Review Tasks", "Meetings", "Overview"])

    # --- Assign Task ---
    with tabs[0]:
        st.subheader("📝 Assign New Task")
        company = st.text_input("Company")
        employee = st.text_input("Employee Name")
        task = st.text_input("Task Description")
        completion = st.slider("Initial Completion %", 0, 100, 0)
        if st.button("Assign"):
            status = "✅ On Track" if completion >= 60 else "🕐 Pending"
            record = {
                "record_type": "task",
                "company": company,
                "employee": employee,
                "task": task,
                "completion": completion,
                "status": status,
                "reviewed": False,
                "assigned_on": str(datetime.now())
            }
            upsert_record(record)
            st.success(f"Task '{task}' assigned to {employee}")

    # --- Review Tasks ---
    with tabs[1]:
        st.subheader("🧾 Review Pending Tasks")
        tasks = filter_type(df, "task")
        if tasks.empty:
            st.info("No tasks yet.")
        else:
            pending = tasks[tasks["reviewed"] != True]
            if pending.empty:
                st.success("All tasks reviewed!")
            else:
                for i, r in pending.iterrows():
                    st.write(f"🧠 {r['task']} — {r['employee']}")
                    completion = st.slider(f"Update Completion ({r['employee']})", 0, 100, int(r['completion']), key=f"rev_{i}")
                    if st.button(f"✅ Approve {r['task']}", key=f"ap_{i}"):
                        r["completion"] = completion
                        r["status"] = "✅ Completed" if completion == 100 else "✅ On Track"
                        r["reviewed"] = True
                        upsert_record(r.to_dict())
                        st.success(f"Approved {r['task']}")
                        st.rerun()  # FIXED: Updated from experimental_rerun()

    # --- Meetings ---
    with tabs[2]:
        st.subheader("📅 Schedule Meeting")
        title = st.text_input("Meeting Title")
        organizer = st.text_input("Organizer")
        participants = st.text_area("Participants (comma-separated)")
        meet_date = st.date_input("Date", value=date.today())
        meet_time = st.time_input("Time", value=datetime.now().time())
        agenda = st.text_area("Agenda")
        if st.button("Schedule Meeting"):
            dt = datetime.combine(meet_date, meet_time)
            record = {
                "record_type": "meeting",
                "meeting_title": title,
                "organizer": organizer,
                "participants": participants,
                "meeting_datetime": str(dt),
                "agenda": agenda
            }
            upsert_record(record)
            st.success("Meeting Scheduled Successfully!")

    # --- Overview ---
    with tabs[3]:
        tasks = filter_type(df, "task")
        summary_card("Total Tasks", len(tasks), "🧾")
        summary_card("Pending Review", len(tasks[tasks["reviewed"] != True]), "⏳")
        summary_card("Completed", len(tasks[tasks["status"].str.contains('Completed', na=False)]), "✅")
        if not tasks.empty:
            st.subheader("📊 All Tasks Overview")
            # FIXED: Using map() instead of deprecated applymap()
            st.dataframe(tasks[["company", "employee", "task", "completion", "status"]]
                         .style.map(color_status, subset=["status"]))

# ============================================================
# 👷 TEAM MEMBER
# ============================================================
elif role == "Team Member":
    st.header("👷 Team Member Dashboard")
    tabs = st.tabs(["My Tasks", "Update Progress", "My Meetings"])

    # --- My Tasks ---
    with tabs[0]:
        st.subheader("📋 My Tasks")
        tasks = filter_type(df, "task")
        if tasks.empty:
            st.info("No tasks available.")
        else:
            employees = sorted(tasks["employee"].dropna().unique())
            emp = st.selectbox("Select Your Name", employees)
            my_tasks = tasks[tasks["employee"] == emp]
            if my_tasks.empty:
                st.warning("No tasks assigned to you yet.")
            else:
                summary_card("Total Tasks", len(my_tasks), "🧾")
                # FIXED: Using map() instead of deprecated applymap()
                st.dataframe(my_tasks[["task", "completion", "status"]]
                             .style.map(color_status, subset=["status"]))

    # --- Update Progress ---
    with tabs[1]:
        st.subheader("✅ Update Your Work Progress")
        emp = st.text_input("Your Name")
        task = st.text_input("Task Name")
        progress = st.slider("Completion %", 0, 100, 0)
        if st.button("Submit Progress"):
            status = "✅ Completed" if progress == 100 else "🕐 In Progress"
            record = {
                "record_type": "task",
                "employee": emp,
                "task": task,
                "completion": progress,
                "status": status,
                "reviewed": False,
                "updated_on": str(datetime.now())
            }
            upsert_record(record)
            st.success("Progress updated successfully!")

    # --- My Meetings ---
    with tabs[2]:
        st.subheader("📅 My Meetings")
        meets = filter_type(df, "meeting")
        if meets.empty:
            st.info("No meetings yet.")
        else:
            st.dataframe(meets[["meeting_title", "organizer", "meeting_datetime", "participants"]])

# ============================================================
# 🧾 CLIENT
# ============================================================
elif role == "Client":
    st.header("🧾 Client Portal")
    tasks = filter_type(df, "task")
    reviewed = tasks[tasks["reviewed"] == True] if not tasks.empty else pd.DataFrame()
    if reviewed.empty:
        st.info("No reviewed tasks to approve.")
    else:
        for i, r in reviewed.iterrows():
            st.write(f"🧾 {r['task']} — {r['employee']}")
            if st.button(f"Approve {r['task']}", key=f"cl_{i}"):
                r["status"] = "✅ Client Approved"
                upsert_record(r.to_dict())
                st.success(f"Approved {r['task']}")

# ============================================================
# 🧠 ADMIN
# ============================================================
elif role == "Admin":
    st.header("🧠 Admin Dashboard")
    tasks = filter_type(df, "task")
    meets = filter_type(df, "meeting")
    summary_card("Total Records", len(df), "📦")
    summary_card("Total Tasks", len(tasks), "🧾")
    summary_card("Meetings", len(meets), "📅")
    if not df.empty:
        st.dataframe(df)

# ------------------------------------------------
# FOOTER
# ------------------------------------------------
st.markdown("---")
st.caption("🚀 Professional Submission Edition — Local Only | Error-Free | No Pinecone")
