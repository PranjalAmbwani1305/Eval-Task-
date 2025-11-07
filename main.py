# ============================================================
# 🏢 AI Enterprise Workforce & Task Management — Role-Based Edition (No Login)
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
from pinecone import Pinecone
from sklearn.cluster import KMeans
import plotly.express as px
import uuid
from datetime import date, datetime, timedelta

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(page_title="AI Enterprise Workforce System", layout="wide")
st.title("🏢 AI Enterprise Workforce & Task Management — Role-Based Edition")

# ------------------------------------------------
# PINECONE CONNECTION
# ------------------------------------------------
INDEX_NAME = "task"
DIMENSION = 1024

try:
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index(INDEX_NAME)
    st.sidebar.success("✅ Connected to Pinecone Database")
except Exception as e:
    st.sidebar.error(f"⚠️ Pinecone Connection Failed: {e}")
    index = None

# ------------------------------------------------
# DATA HELPERS
# ------------------------------------------------
def fetch_data():
    """Fetch all records from Pinecone."""
    try:
        res = index.query(vector=np.random.rand(DIMENSION).tolist(), top_k=1000, include_metadata=True)
        return pd.DataFrame([m.metadata for m in res.matches if m.metadata])
    except Exception as e:
        st.warning(f"⚠️ Data fetch error: {e}")
        return pd.DataFrame()

def upsert_data(records):
    """Insert/Update data into Pinecone."""
    batch = [{
        "id": str(uuid.uuid4()),
        "values": np.random.rand(DIMENSION).tolist(),
        "metadata": rec
    } for rec in records]
    index.upsert(batch)

# ------------------------------------------------
# SIDEBAR: ROLE SELECTION
# ------------------------------------------------
st.sidebar.header("🎚 Select Role")
role = st.sidebar.radio("Choose Role", ["Manager", "Team Member", "Client", "Admin"])
st.sidebar.markdown("---")
st.sidebar.caption("Role-based enterprise dashboard — no login required")

# ------------------------------------------------
# COMMON DATA FETCH
# ------------------------------------------------
df = fetch_data()

# ============================================================
# 👨‍💼 MANAGER DASHBOARD
# ============================================================
if role == "Manager":
    st.header("👨‍💼 Manager Dashboard — Enterprise Control")

    tabs = st.tabs([
        "📋 All Tasks", "♻️ Reassign Log", "🧾 Review Tasks",
        "🏢 Department Insights", "📊 360° AI Overview", "🏖 Leave Requests"
    ])

    # --- All Tasks ---
    with tabs[0]:
        if not df.empty:
            st.subheader("📋 All Assigned Tasks")
            st.dataframe(df[["company", "department", "employee", "task", "completion", "marks", "status"]])
        else:
            st.info("No tasks found.")

    # --- Reassign Log ---
    with tabs[1]:
        if not df.empty and "reassigned_on" in df.columns:
            reassign_df = df.dropna(subset=["reassigned_on"])
            st.subheader("♻️ Task Reassignment Log")
            st.dataframe(reassign_df[["company", "employee", "task", "reassigned_on"]])
        else:
            st.info("No reassignments available.")

    # --- Review Tasks ---
    with tabs[2]:
        st.subheader("🧾 Review Pending Tasks")
        if not df.empty and "reviewed" in df.columns:
            pending = df[df["reviewed"] != True]
            if not pending.empty:
                for i, r in pending.iterrows():
                    st.markdown(f"### {r['task']} — {r['employee']}")
                    adj = st.slider(f"Completion % ({r['employee']})", 0, 100, int(r["completion"]))
                    marks = round(adj * 0.05, 2)
                    comment = st.text_area(f"Manager Comment for {r['task']}", key=f"c_{i}")
                    if st.button(f"✅ Approve {r['task']}", key=f"f_{i}"):
                        r["completion"] = adj
                        r["marks"] = marks
                        r["reviewed"] = True
                        r["comments"] = comment
                        upsert_data([r.to_dict()])
                        st.success(f"✅ Reviewed {r['task']}")
                        st.experimental_rerun()
            else:
                st.success("All tasks reviewed.")
        else:
            st.info("No review data available.")

    # --- Department Insights ---
    with tabs[3]:
        st.subheader("🏢 Department Performance Overview")
        if not df.empty and "department" in df.columns:
            avg_df = df.groupby("department")[["marks", "completion"]].mean().reset_index()
            st.dataframe(avg_df)
            fig = px.bar(avg_df, x="department", y="marks", color="department", title="Average Marks by Department")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No department data found.")

    # --- 360° Overview ---
    with tabs[4]:
        st.subheader("📊 AI 360° Performance Clustering")
        if not df.empty and {"marks", "completion"} <= set(df.columns):
            df["marks"] = pd.to_numeric(df["marks"], errors="coerce")
            df["completion"] = pd.to_numeric(df["completion"], errors="coerce")
            df = df.dropna(subset=["marks", "completion"])
            if len(df) >= 3:
                kmeans = KMeans(n_clusters=3, random_state=42)
                df["cluster"] = kmeans.fit_predict(df[["marks", "completion"]])
                fig = px.scatter(
                    df, x="completion", y="marks", color="cluster",
                    hover_data=["employee", "department"], title="Employee Performance Clusters"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Need at least 3 records for clustering.")
        else:
            st.info("No sufficient data for clustering.")

    # --- Leave Requests ---
    with tabs[5]:
        st.subheader("🏖 Leave Management")
        if not df.empty and "status" in df.columns:
            leaves = df[df["status"] == "Leave Applied"]
            if not leaves.empty:
                for i, l in leaves.iterrows():
                    st.write(f"🧾 {l['employee']} — {l['leave_type']} ({l['from']} → {l['to']})")
                    if st.button(f"Approve {l['employee']}", key=f"ap_{i}"):
                        l["status"] = "Leave Approved"
                        upsert_data([l.to_dict()])
                        st.success("Leave approved.")
            else:
                st.info("No leave requests.")
        else:
            st.info("No leave data available.")


# ============================================================
# 👷 TEAM MEMBER PORTAL
# ============================================================
elif role == "Team Member":
    st.header("👷 Team Member Portal")

    tabs = st.tabs(["📋 My Tasks", "✅ Update Progress", "🏖 Apply for Leave", "📊 Performance Snapshot"])

    # --- My Tasks ---
    with tabs[0]:
        if not df.empty and "employee" in df.columns:
            employees = sorted(df["employee"].dropna().unique())
            emp = st.selectbox("Select Your Name", employees)
            my_tasks = df[df["employee"] == emp]
            if not my_tasks.empty:
                st.dataframe(my_tasks[["company", "task", "completion", "marks", "status", "sentiment"]])
            else:
                st.info("No tasks assigned yet.")
        else:
            st.warning("No data found.")

    # --- Update Progress ---
    with tabs[1]:
        emp = st.text_input("👤 Your Name")
        task_name = st.text_input("🧠 Task Title")
        progress = st.slider("Completion %", 0, 100, 0)
        if st.button("Submit Update"):
            marks = round(progress * 0.05, 2)
            status = "On Track" if progress >= 60 else "Delayed"
            record = {
                "employee": emp, "task": task_name,
                "completion": progress, "marks": marks,
                "status": status, "reviewed": False,
                "submitted_on": str(datetime.now())
            }
            upsert_data([record])
            st.success(f"✅ Update submitted for {task_name}")

    # --- Apply for Leave ---
    with tabs[2]:
        emp = st.text_input("Employee Name")
        leave_type = st.selectbox("Leave Type", ["Casual", "Sick", "Paid"])
        from_date = st.date_input("From", value=date.today())
        to_date = st.date_input("To", value=date.today() + timedelta(days=1))
        reason = st.text_area("Reason")
        if st.button("Apply Leave"):
            record = {
                "employee": emp, "leave_type": leave_type,
                "from": str(from_date), "to": str(to_date),
                "reason": reason, "status": "Leave Applied"
            }
            upsert_data([record])
            st.success("✅ Leave application submitted.")

    # --- Performance Snapshot ---
    with tabs[3]:
        if not df.empty and {"employee", "marks", "completion"} <= set(df.columns):
            emp = st.text_input("Your Name for Report")
            perf_df = df[df["employee"] == emp]
            if not perf_df.empty:
                st.metric("📈 Avg Completion", f"{perf_df['completion'].mean():.1f}%")
                st.metric("🏅 Avg Marks", f"{perf_df['marks'].mean():.2f}")
                fig = px.bar(perf_df, x="task", y="completion", color="status", title=f"{emp}'s Task Overview")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No performance data available.")


# ============================================================
# 🧾 CLIENT PORTAL
# ============================================================
elif role == "Client":
    st.header("🧾 Client Project Review Portal")

    tabs = st.tabs(["📊 Project Overview", "✅ Approve Tasks", "💬 Feedback Summary"])

    # --- Project Overview ---
    with tabs[0]:
        if not df.empty:
            companies = sorted(df["company"].dropna().unique())
            company = st.selectbox("🏢 Select Company", companies)
            company_df = df[df["company"] == company]
            st.dataframe(company_df[["employee", "task", "completion", "marks", "status", "sentiment"]])
            fig = px.bar(company_df, x="employee", y="marks", color="sentiment", title=f"{company} — Employee Performance")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No company data available.")

    # --- Approve Tasks ---
    with tabs[1]:
        if not df.empty and "reviewed" in df.columns:
            reviewed_df = df[df["reviewed"] == True]
            for i, r in reviewed_df.iterrows():
                st.markdown(f"### {r['task']} — {r['employee']}")
                if st.button(f"✅ Approve {r['task']}", key=f"app_{i}"):
                    r["client_approved_on"] = str(datetime.now())
                    upsert_data([r.to_dict()])
                    st.success(f"Approved {r['task']}")
        else:
            st.info("No tasks ready for approval.")

    # --- Feedback Summary ---
    with tabs[2]:
        if not df.empty and "sentiment" in df.columns:
            sentiment_df = df["sentiment"].value_counts().reset_index()
            sentiment_df.columns = ["Sentiment", "Count"]
            fig = px.pie(sentiment_df, names="Sentiment", values="Count", title="Client Feedback Sentiment Summary")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No feedback data available.")


# ============================================================
# 🧠 ADMIN DASHBOARD (Optional)
# ============================================================
elif role == "Admin":
    st.header("🧠 Admin Dashboard — Global Overview")

    if not df.empty:
        st.metric("👥 Total Employees", df["employee"].nunique())
        st.metric("🏢 Companies", df["company"].nunique())
        st.metric("🧾 Total Tasks", len(df))
        fig = px.scatter(df, x="completion", y="marks", color="department", hover_data=["employee", "company"], title="Performance Distribution by Department")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for admin overview.")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("🚀 Enterprise Role-Based Edition — AI-Driven Workforce Intelligence (No Username or Password)")
