# ============================================================
# 🏢 AI Enterprise Workforce Management System — Pinecone Edition
# ============================================================

import streamlit as st
import pandas as pd
import uuid
from datetime import date, datetime, timedelta
import random
from pinecone import Pinecone, ServerlessSpec
import json

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(page_title="AI Workforce System", layout="wide")
st.title("🏢 AI Enterprise Workforce — Pinecone Edition")

# ------------------------------------------------
# PINECONE SETUP
# ------------------------------------------------
@st.cache_resource
def init_pinecone():
    """Initialize Pinecone connection"""
    api_key = st.secrets.get("PINECONE_API_KEY", "")
    if not api_key:
        st.error("⚠️ PINECONE_API_KEY not found in secrets!")
        return None, None
    
    pc = Pinecone(api_key=api_key)
    index_name = "task"
    
    # Create index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1024,  # Dimension for metadata storage
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    
    index = pc.Index(index_name)
    return pc, index

pc, index = init_pinecone()

# ------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------
def create_dummy_vector():
    """Create a dummy vector for storage (we're using Pinecone as key-value store)"""
    return [0.0] * 384

def save_record(record):
    """Save a record to Pinecone"""
    if not index:
        st.error("Pinecone not initialized!")
        return False
    
    if "_id" not in record:
        record["_id"] = str(uuid.uuid4())
    
    try:
        # Convert all values to strings for metadata
        metadata = {k: str(v) for k, v in record.items()}
        
        index.upsert(
            vectors=[{
                "id": record["_id"],
                "values": create_dummy_vector(),
                "metadata": metadata
            }]
        )
        return True
    except Exception as e:
        st.error(f"Error saving to Pinecone: {e}")
        return False

def get_all_records():
    """Fetch all records from Pinecone"""
    if not index:
        return pd.DataFrame()
    
    try:
        # Query to get all records (using a dummy vector)
        results = index.query(
            vector=create_dummy_vector(),
            top_k=10000,
            include_metadata=True
        )
        
        if not results.matches:
            return pd.DataFrame()
        
        # Convert to DataFrame
        data = []
        for match in results.matches:
            record = match.metadata.copy()
            record["_id"] = match.id
            data.append(record)
        
        df = pd.DataFrame(data)
        
        # Convert numeric columns
        if "completion" in df.columns:
            df["completion"] = pd.to_numeric(df["completion"], errors="coerce")
        
        # Convert boolean columns
        if "reviewed" in df.columns:
            df["reviewed"] = df["reviewed"].map({"True": True, "False": False, "true": True, "false": False})
        
        return df
    except Exception as e:
        st.error(f"Error fetching from Pinecone: {e}")
        return pd.DataFrame()

def delete_record(record_id):
    """Delete a record from Pinecone"""
    if not index:
        return False
    try:
        index.delete(ids=[record_id])
        return True
    except Exception as e:
        st.error(f"Error deleting from Pinecone: {e}")
        return False

def seed_data():
    """Add sample data"""
    employees = ["Aarav", "Isha", "Rohan", "Amruta"]
    companies = ["TechNova", "InnoSoft"]
    
    for i in range(6):
        completion = random.choice([10, 40, 70, 90])
        status = "✅ On Track" if completion >= 60 else "🕐 Pending"
        record = {
            "_id": str(uuid.uuid4()),
            "record_type": "task",
            "company": random.choice(companies),
            "employee": random.choice(employees),
            "task": f"Task {100+i}",
            "completion": completion,
            "status": status,
            "reviewed": False
        }
        save_record(record)
    
    # Add a sample meeting
    record = {
        "_id": str(uuid.uuid4()),
        "record_type": "meeting",
        "meeting_title": "Weekly Sync",
        "organizer": "Manager",
        "participants": ", ".join(random.sample(employees, 3)),
        "meeting_datetime": str(datetime.now() + timedelta(days=1)),
        "agenda": "Project progress discussion"
    }
    save_record(record)
    
    st.success("✅ Sample data added to Pinecone successfully!")
    st.rerun()

def filter_type(df, rtype):
    """Filter dataframe by record type"""
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

if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

# Load data from Pinecone
df = get_all_records()

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
            if save_record(record):
                st.success(f"Task '{task}' assigned to {employee}")
                st.rerun()

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
                    completion = st.slider(
                        f"Update Completion ({r['employee']})", 
                        0, 100, 
                        int(r['completion']), 
                        key=f"rev_{i}"
                    )
                    if st.button(f"✅ Approve {r['task']}", key=f"ap_{i}"):
                        record = r.to_dict()
                        record["completion"] = completion
                        record["status"] = "✅ Completed" if completion == 100 else "✅ On Track"
                        record["reviewed"] = True
                        if save_record(record):
                            st.success(f"Approved {r['task']}")
                            st.rerun()

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
            if save_record(record):
                st.success("Meeting Scheduled Successfully!")
                st.rerun()

    # --- Overview ---
    with tabs[3]:
        tasks = filter_type(df, "task")
        summary_card("Total Tasks", len(tasks), "🧾")
        summary_card("Pending Review", len(tasks[tasks["reviewed"] != True]), "⏳")
        summary_card("Completed", len(tasks[tasks["status"].str.contains('Completed', na=False)]), "✅")
        if not tasks.empty:
            st.subheader("📊 All Tasks Overview")
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
            if save_record(record):
                st.success("Progress updated successfully!")
                st.rerun()

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
                record = r.to_dict()
                record["status"] = "✅ Client Approved"
                if save_record(record):
                    st.success(f"Approved {r['task']}")
                    st.rerun()

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
        st.subheader("📊 All Records")
        st.dataframe(df)
        
        # Add delete functionality for admin
        st.subheader("🗑️ Delete Records")
        record_to_delete = st.selectbox("Select record to delete", df["_id"].tolist() if "_id" in df.columns else [])
        if st.button("Delete Selected Record"):
            if delete_record(record_to_delete):
                st.success("Record deleted successfully!")
                st.rerun()

# ------------------------------------------------
# FOOTER
# ------------------------------------------------
st.markdown("---")
st.caption("🚀 Pinecone Edition — Cloud Persistent Storage | Fully Functional")
