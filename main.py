import streamlit as st
from pinecone import Pinecone
import pandas as pd
import numpy as np
import uuid
from datetime import datetime, date
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import plotly.express as px

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Task Management System", layout="wide")
st.title("AI-Powered Task Management System")

INDEX_NAME = "task-index"
DIMENSION = 1024

# ---------------- PINECONE INIT ----------------
try:
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index(INDEX_NAME)
    st.sidebar.success(f"Connected to Pinecone ({INDEX_NAME})")
except Exception as e:
    st.sidebar.error(f"Pinecone connection failed: {e}")
    st.stop()

# ---------------- UTILITIES ----------------
def rand_vec():
    return np.random.rand(DIMENSION).tolist()

def safe_meta(md):
    clean = {}
    for k, v in md.items():
        if isinstance(v, (datetime, date)):
            v = v.isoformat()
        elif v is None:
            v = ""
        clean[k] = v
    return clean

def fetch_all():
    """Fetch all records from Pinecone and return as DataFrame"""
    try:
        res = index.query(vector=rand_vec(), top_k=1000, include_metadata=True)
        if not res.matches:
            return pd.DataFrame()
        rows = []
        for m in res.matches:
            md = m.metadata or {}
            md["_id"] = m.id
            rows.append(md)
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Simple marks model
lin_reg = LinearRegression()
lin_reg.fit([[0], [50], [100]], [0, 2.5, 5])

# ---------------- ROLE SELECTION ----------------
role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client", "Boss / Admin"])

# ---------------- MANAGER ----------------
if role == "Manager":
    st.header("Manager Dashboard")
    tab1, tab2 = st.tabs(["Assign Task", "Review Tasks"])

    with tab1:
        st.subheader("Assign New Task")
        with st.form("assign_form"):
            company = st.text_input("Company Name")
            employee = st.text_input("Employee Name")
            task = st.text_input("Task Title")
            desc = st.text_area("Task Description")
            deadline = st.date_input("Deadline", value=date.today())
            month = datetime.now().strftime("%B %Y")
            submit = st.form_submit_button("Assign Task")

            if submit and all([company, employee, task]):
                tid = str(uuid.uuid4())
                meta = safe_meta({
                    "company": company,
                    "employee": employee,
                    "task": task,
                    "description": desc,
                    "deadline": deadline,
                    "month": month,
                    "completion": 0,
                    "marks": 0,
                    "status": "Assigned",
                    "manager_reviewed": False,
                    "boss_reviewed": False,
                    "client_reviewed": False,
                    "assigned_on": datetime.now().isoformat()
                })
                index.upsert([{"id": tid, "values": rand_vec(), "metadata": meta}])
                st.success(f"Task '{task}' assigned to {employee}")

    with tab2:
        df = fetch_all()
        if df.empty:
            st.info("No tasks found.")
        else:
            pending = df[df["manager_reviewed"].astype(str).str.lower() != "true"]
            if pending.empty:
                st.info("No tasks pending review.")
            else:
                for _, row in pending.iterrows():
                    st.markdown(f"### {row['task']}")
                    st.write(f"Employee: {row['employee']}")
                    st.write(f"Reported Completion: {row.get('completion', 0)}%")

                    new_comp = st.slider(f"Adjust Completion for {row['task']}", 0, 100, int(float(row.get("completion", 0))), step=5)
                    new_marks = round(float(lin_reg.predict([[new_comp]])[0]), 2)
                    st.write(f"Marks: {new_marks}")
                    comments = st.text_area(f"Manager Comments for {row['task']}")
                    approve = st.radio("Approve?", ["Yes", "No"], key=f"m_{row['_id']}")

                    if st.button(f"Finalize Review for {row['task']}", key=row["_id"]):
                        updated = safe_meta({
                            **row.to_dict(),
                            "completion": new_comp,
                            "marks": new_marks,
                            "manager_comments": comments,
                            "manager_reviewed": True,
                            "manager_approved": approve == "Yes",
                            "status": "Manager Approved" if approve == "Yes" else "Needs Rework"
                        })
                        index.upsert([{"id": row["_id"], "values": rand_vec(), "metadata": updated}])
                        st.success(f"Review complete for {row['task']}")
                        st.rerun()

# ---------------- TEAM MEMBER ----------------
elif role == "Team Member":
    st.header("Team Member Dashboard")
    company = st.text_input("Company")
    employee = st.text_input("Your Name")

    if st.button("Load My Tasks"):
        df = fetch_all()
        df = df[(df["company"] == company) & (df["employee"] == employee)]
        if df.empty:
            st.info("No tasks found.")
        else:
            for _, row in df.iterrows():
                st.subheader(row["task"])
                st.write(f"Description: {row['description']}")
                curr = int(float(row.get("completion", 0)))
                new = st.slider(f"Completion % for {row['task']}", 0, 100, curr)
                if st.button(f"Submit {row['task']}", key=row["_id"]):
                    marks = float(lin_reg.predict([[new]])[0])
                    updated = safe_meta({
                        **row.to_dict(),
                        "completion": new,
                        "marks": marks,
                        "status": "Submitted"
                    })
                    index.upsert([{"id": row["_id"], "values": rand_vec(), "metadata": updated}])
                    st.success(f"Progress updated for {row['task']}")
                    st.rerun()

# ---------------- CLIENT ----------------
elif role == "Client":
    st.header("Client Dashboard")
    company = st.text_input("Company Name")

    if st.button("Load Finalized Tasks"):
        df = fetch_all()
        # ✅ Filter after fetch instead of Pinecone query
        df = df[
            (df["company"] == company)
            & (df["manager_reviewed"].astype(str).str.lower() == "true")
            & (df["boss_reviewed"].astype(str).str.lower() == "true")
            & (df["client_reviewed"].astype(str).str.lower() != "true")
        ]

        if df.empty:
            st.info("No tasks ready for client review.")
        else:
            for _, row in df.iterrows():
                st.subheader(row["task"])
                st.write(f"Employee: {row['employee']}")
                st.write(f"Completion: {row['completion']}% | Marks: {row['marks']}")
                st.write(f"Manager Comments: {row.get('manager_comments', '')}")
                st.write(f"Boss Comments: {row.get('boss_comments', '')}")

                comments = st.text_area(f"Client Feedback for {row['task']}")
                approve = st.radio("Approve Task?", ["Yes", "No"], key=f"c_{row['_id']}")

                if st.button(f"Finalize Review {row['task']}", key=f"client_btn_{row['_id']}"):
                    updated = safe_meta({
                        **row.to_dict(),
                        "client_comments": comments,
                        "client_reviewed": True,
                        "client_approved": approve == "Yes",
                        "status": "Client Approved" if approve == "Yes" else "Rework Requested"
                    })
                    index.upsert([{"id": row["_id"], "values": rand_vec(), "metadata": updated}])
                    st.success(f"Client review submitted for {row['task']}")
                    st.rerun()

# ---------------- ADMIN / BOSS ----------------
elif role == "Boss / Admin":
    st.header("Boss / Admin Dashboard")
    df = fetch_all()

    if df.empty:
        st.info("No data available.")
    else:
        st.subheader("All Tasks Overview")
        st.dataframe(df[["company", "employee", "task", "completion", "marks", "status"]])

        # ✅ Dynamic KMeans
        n_clusters = min(3, len(df))
        if n_clusters >= 1:
            df["completion"] = pd.to_numeric(df["completion"], errors="coerce").fillna(0)
            df["marks"] = pd.to_numeric(df["marks"], errors="coerce").fillna(0)
            km = KMeans(n_clusters=n_clusters, n_init=10).fit(df[["completion", "marks"]])
            df["cluster"] = km.labels_
            st.subheader("K-Means Task Clusters")
            st.plotly_chart(px.scatter(
                df, x="completion", y="marks", color=df["cluster"].astype(str),
                hover_data=["employee", "task"], title="Performance Clusters"
            ))

        # ✅ Boss override
        st.subheader("Boss Review & Override")
        pending = df[df["manager_reviewed"].astype(str).str.lower() == "true"]
        for _, row in pending.iterrows():
            st.markdown(f"### {row['task']} - {row['employee']}")
            st.write(f"Manager Completion: {row['completion']}% | Marks: {row['marks']}")
            new_c = st.slider(f"Adjust Completion for {row['task']}", 0, 100, int(float(row["completion"])))
            new_m = round(float(lin_reg.predict([[new_c]])[0]), 2)
            comment = st.text_area(f"Boss Comments for {row['task']}")
            approve = st.radio("Approve?", ["Yes", "No"], key=f"b_{row['_id']}")
            if st.button(f"Finalize Boss Review {row['task']}", key=row["_id"]):
                updated = safe_meta({
                    **row.to_dict(),
                    "completion": new_c,
                    "marks": new_m,
                    "boss_comments": comment,
                    "boss_reviewed": True,
                    "boss_approved": approve == "Yes",
                    "status": "Final Approved" if approve == "Yes" else "Rework Requested"
                })
                index.upsert([{"id": row["_id"], "values": rand_vec(), "metadata": updated}])
                st.success(f"Boss review finalized for {row['task']}")
                st.rerun()
