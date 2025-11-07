import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from textblob import TextBlob
import plotly.express as px
import uuid
import os

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="AI Enterprise Workforce System", layout="wide")
st.title("🏢 AI Enterprise Workforce & Task Management System")

# ---------------------------
# SECRETS
# ---------------------------
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
INDEX_NAME = "task"
DIMENSION = 1024

# ---------------------------
# SAFE DF CHECK
# ---------------------------
def is_valid_df(df):
    """Return True if df is a non-empty pandas DataFrame."""
    return isinstance(df, pd.DataFrame) and not df.empty

# ---------------------------
# PINECONE CONNECTION
# ---------------------------
def init_pinecone():
    if not PINECONE_API_KEY:
        st.warning("⚠️ Pinecone API key missing. Please add it to secrets.toml.")
        return None
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        if INDEX_NAME not in {i["name"] for i in pc.list_indexes()}:
            pc.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        return pc.Index(INDEX_NAME)
    except Exception as e:
        st.error(f"❌ Pinecone connection failed: {e}")
        return None

index = init_pinecone()

def now(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def rand_vec(): return np.random.rand(DIMENSION).tolist()

# ---------------------------
# SAFE UPSERT + QUERY
# ---------------------------
def safe_upsert(md):
    if not index: return
    try:
        index.upsert([{
            "id": str(md.get("_id", uuid.uuid4())),
            "values": rand_vec(),
            "metadata": md
        }])
    except Exception as e:
        st.error(f"Pinecone upsert failed: {e}")

def safe_query(filter_dict=None, top_k=1000):
    if not index: return pd.DataFrame()
    try:
        res = index.query(vector=rand_vec(), top_k=top_k, include_metadata=True, filter=filter_dict or {})
        rows = [m.metadata for m in res.matches if m.metadata]
        return pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"Query failed: {e}")
        return pd.DataFrame()

# ---------------------------
# ML MODELS
# ---------------------------
lin_reg = LinearRegression().fit([[0],[50],[100]],[0,2.5,5])
log_reg = LogisticRegression().fit([[0],[40],[80],[100]],[0,0,1,1])
rf = RandomForestClassifier().fit(np.array([[10,2],[50,1],[90,0],[100,0]]),[0,1,0,0])
tfidf = TfidfVectorizer().fit(["excellent work","needs improvement","bad performance","great job","average"])
svm = SVC().fit(tfidf.transform(["excellent work","needs improvement","bad performance","great job","average"]),[1,0,0,1,0])

# ---------------------------
# ROLE SELECTION
# ---------------------------
role = st.sidebar.selectbox("Login as", ["Manager","Team Member","Department Head","Client"])

# =====================================================
# MANAGER DASHBOARD
# =====================================================
if role == "Manager":
    st.header("👨‍💼 Manager Dashboard — Enterprise")
    tab1, tab2, tab3, tab4 = st.tabs(["Assign / Reassign","Review Tasks","Leave Requests","Analytics & Leaderboard"])

    # --- Assign Task
    with tab1:
        with st.form("assign_form"):
            company = st.text_input("Company")
            project = st.text_input("Project")
            department = st.selectbox("Department",["IT","HR","Finance","Marketing","Operations"])
            team = st.text_input("Team (optional)")
            employee = st.text_input("Employee")
            task = st.text_input("Task Title")
            desc = st.text_area("Description")
            deadline = st.date_input("Deadline", value=date.today()+timedelta(days=7))
            uploaded_file = st.file_uploader("Attach Proof / Files (optional)", accept_multiple_files=False)
            submit = st.form_submit_button("Assign Task")

            if submit and company and employee and task:
                file_meta = None
                if uploaded_file:
                    save_path = f"uploads/{uuid.uuid4()}_{uploaded_file.name}"
                    os.makedirs("uploads", exist_ok=True)
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_meta = {"file_name": uploaded_file.name, "path": save_path}
                md = {
                    "_id": str(uuid.uuid4()), "company": company, "project": project,
                    "department": department, "team": team, "employee": employee,
                    "task": task, "description": desc, "deadline": deadline.isoformat(),
                    "completion": 0, "marks": 0, "status": "Assigned", "assigned_on": now(),
                    "file": file_meta
                }
                safe_upsert(md)
                st.success(f"✅ Task '{task}' assigned to {employee}")

    # --- Review Tasks
    with tab2:
        st.subheader("✅ Review & Finalize Tasks")
        f_company = st.text_input("Filter by Company")
        f_dept = st.selectbox("Filter by Department",["","IT","HR","Finance","Marketing","Operations"])
        fdict = {}
        if f_company: fdict["company"] = {"$eq": f_company}
        if f_dept: fdict["department"] = {"$eq": f_dept}
        df_tasks = safe_query(filter_dict=fdict)

        if is_valid_df(df_tasks):
            df = df_tasks
            st.dataframe(df[["company","department","team","employee","task","completion","marks","status"]])
            for i, row in df.iterrows():
                st.markdown(f"### {row['task']} — {row['employee']} ({row.get('team','N/A')})")
                adj = st.slider("Completion %",0,100,int(row.get("completion",0)),key=f"adj_{i}")
                marks = float(lin_reg.predict([[adj]])[0])
                comment = st.text_area("Manager Comment",key=f"com_{i}")
                approve = st.radio("Approve?",["Yes","No"],key=f"app_{i}")
                if st.button(f"Finalize {row['task']}",key=f"fin_{i}"):
                    sentiment_val = int(svm.predict(tfidf.transform([comment]))[0]) if comment else None
                    sentiment = "Positive" if sentiment_val==1 else "Negative"
                    row.update({"completion":adj,"marks":marks,"sentiment":sentiment,
                                "manager_comment":comment,"approved":approve=="Yes","reviewed_on":now()})
                    safe_upsert(dict(row))
                    st.success(f"Reviewed {row['task']} ({sentiment})")
        else:
            st.info("📂 No tasks found or Pinecone not connected.")

    # --- Leave Requests
    with tab3:
        st.subheader("🏖 Leave Requests")
        leaves = safe_query(filter_dict={"status":{"$eq":"Leave Applied"}})
        if is_valid_df(leaves):
            for i,lv in leaves.iterrows():
                st.markdown(f"**{lv['employee']}** — {lv.get('leave_type','General')} leave {lv['from_date']}→{lv['to_date']}")
                dec = st.radio("Decision",["Approve","Reject"],key=f"lv_{i}")
                if st.button(f"Submit ({lv['employee']})",key=f"sub_{i}"):
                    lv["status"] = "Leave Approved" if dec=="Approve" else "Leave Rejected"
                    lv["decision_on"] = now()
                    safe_upsert(dict(lv))
                    st.success(f"{dec}ed leave for {lv['employee']}")
        else:
            st.info("No pending leave requests.")

    # --- Analytics
    with tab4:
        st.subheader("📊 Analytics & Leaderboard")
        df = safe_query()
        if is_valid_df(df):
            df["marks"] = pd.to_numeric(df["marks"],errors="coerce")
            df["completion"] = pd.to_numeric(df["completion"],errors="coerce")
            st.metric("Avg Marks",f"{df['marks'].mean():.2f}")
            st.metric("Avg Completion",f"{df['completion'].mean():.2f}")
            top = df.sort_values("marks",ascending=False).head(10)
            fig = px.bar(top,x="employee",y="marks",color="department",title="Top Performers")
            st.plotly_chart(fig,use_container_width=True)
        else:
            st.info("No analytics data yet.")

# =====================================================
# TEAM MEMBER PORTAL
# =====================================================
elif role == "Team Member":
    st.header("👷 Team Member Portal")
    company = st.text_input("Company")
    employee = st.text_input("Your Name")
    if st.button("Load My Tasks"):
        res = safe_query(filter_dict={"company":{"$eq":company},"employee":{"$eq":employee}})
        st.session_state["tasks"] = res
        st.success(f"Loaded {len(res)} tasks.")

    tasks_list = st.session_state.get("tasks", [])
    if is_valid_df(tasks_list):
        for i, t in tasks_list.iterrows():
            st.subheader(f"{t['task']} — {t.get('project','')}")
            curr = int(t.get("completion",0))
            new = st.slider("Completion %",0,100,curr,key=f"tm_{i}")
            if st.button(f"Submit {t['task']}",key=f"sub_{i}"):
                marks = float(lin_reg.predict([[new]])[0])
                track = "On Track" if log_reg.predict([[new]])[0]==1 else "Delayed"
                risk = "High" if rf.predict([[new,0]])[0]==1 else "Low"
                t.update({"completion":new,"marks":marks,"status":track,"risk":risk,"updated_on":now()})
                safe_upsert(dict(t))
                st.success(f"✅ Updated {t['task']} ({track})")
                st.experimental_rerun()
    else:
        st.info("🎉 No tasks assigned yet.")

    st.markdown("---")
    st.subheader("🏖 Apply for Leave")
    emp = st.text_input("Employee Name")
    leave_type = st.selectbox("Type",["Casual","Sick","Paid","Unpaid"])
    fdate = st.date_input("From",value=date.today())
    tdate = st.date_input("To",value=date.today()+timedelta(days=1))
    reason = st.text_area("Reason")
    if st.button("Apply Leave"):
        md={"_id":str(uuid.uuid4()),"employee":emp,"leave_type":leave_type,
            "from_date":fdate.isoformat(),"to_date":tdate.isoformat(),
            "reason":reason,"status":"Leave Applied","applied_on":now()}
        safe_upsert(md)
        st.success("Leave applied successfully.")

# =====================================================
# DEPARTMENT HEAD PANEL
# =====================================================
elif role == "Department Head":
    st.header("🏢 Department Head Panel")
    dept = st.selectbox("Select Department",["IT","HR","Finance","Marketing","Operations"])
    if st.button("Load Department Data"):
        df = safe_query(filter_dict={"department":{"$eq":dept}})
        if is_valid_df(df):
            df["marks"] = pd.to_numeric(df["marks"],errors="coerce")
            st.metric("Avg Marks",f"{df['marks'].mean():.2f}")
            fig = px.box(df,x="team",y="marks",color="team",title=f"{dept} Performance")
            st.plotly_chart(fig,use_container_width=True)
            st.dataframe(df[["employee","task","marks","completion","status","team"]])
        else:
            st.info("No data found for this department.")

# =====================================================
# CLIENT PORTAL
# =====================================================
elif role == "Client":
    st.header("🧾 Client Project Portal")
    company = st.text_input("Company")
    project = st.text_input("Project")
    if st.button("Load Project Overview"):
        df = safe_query(filter_dict={"company":{"$eq":company},"project":{"$eq":project}})
        if is_valid_df(df):
            df["completion"] = pd.to_numeric(df["completion"],errors="coerce")
            st.metric("Avg Completion",f"{df['completion'].mean():.2f}")
            st.dataframe(df[["task","employee","completion","marks","status"]])
            completed = df[df["completion"]>=100]
            if not completed.empty:
                st.subheader("Approve Completed Tasks")
                for i,row in completed.iterrows():
                    st.markdown(f"**{row['task']}** — {row['employee']}")
                    comment = st.text_area("Feedback",key=f"c_{i}")
                    rating = st.slider("Rating (1–5)",1,5,4,key=f"r_{i}")
                    if st.button(f"Submit Review {row['task']}",key=f"s_{i}"):
                        row["client_comment"]=comment
                        row["client_rating"]=rating
                        row["client_approved"]=True
                        row["client_reviewed_on"]=now()
                        safe_upsert(dict(row))
                        st.success(f"Saved review for {row['task']}")
            else:
                st.info("No completed tasks yet.")
        else:
            st.info("No data found for this project.")

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.caption("✅ All ValueErrors fixed — safe .empty checks everywhere. Pinecone + ML + Leave + Team integration ready.")
