# main.py — AI Workforce Intelligence Dashboard (Enterprise)
"""
Dependencies:
  pip install streamlit pinecone-client scikit-learn plotly huggingface-hub PyPDF2 openpyxl pandas
Place your API keys in .streamlit/secrets.toml:
[PINECONE_API_KEY]
HUGGINGFACEHUB_API_TOKEN
"""

import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import pandas as pd
import uuid, json, logging, time, os
from datetime import datetime, date
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import plotly.express as px

# Optional imports
try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except:
    HF_AVAILABLE = False

try:
    import PyPDF2
    PDF_AVAILABLE = True
except:
    PDF_AVAILABLE = False

# ---------------------------------------------------------
# Streamlit Config
# ---------------------------------------------------------
st.set_page_config(page_title="AI Workforce Dashboard", layout="wide")
st.title("AI Workforce Intelligence Platform")

# ---------------------------------------------------------
# Pinecone Configuration
# ---------------------------------------------------------
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
HF_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")

INDEX_NAME = "task"  # ✅ always lowercase, no underscores
DIMENSION = 1024

pc = None
index = None
if PINECONE_API_KEY:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        existing = [i["name"] for i in pc.list_indexes()]
        if INDEX_NAME not in existing:
            pc.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            with st.spinner("Creating Pinecone index..."):
                while True:
                    desc = pc.describe_index(INDEX_NAME)
                    if desc["status"].get("ready"):
                        break
                    time.sleep(2)
        index = pc.Index(INDEX_NAME)
        st.success(f"Connected to Pinecone index: {INDEX_NAME}")
    except Exception as e:
        st.warning(f"Pinecone init failed — running local only. ({e})")
else:
    st.warning("Pinecone API key missing in Streamlit secrets. Running local only.")

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def now(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def rand_vec(): return np.random.rand(DIMENSION).tolist()

def safe_meta(md):
    clean = {}
    for k, v in md.items():
        if isinstance(v, (datetime, date)):
            v = v.isoformat()
        elif isinstance(v, (list, dict)):
            v = json.dumps(v)
        elif pd.isna(v):
            v = ""
        clean[k] = v
    return clean

def safe_rerun():
    try: st.rerun()
    except AttributeError: st.experimental_rerun()

def upsert(id_, md):
    if not index:
        st.session_state.setdefault("LOCAL_DATA", {})[id_] = md
        return
    md = safe_meta(md)
    try:
        index.upsert([{"id": id_, "values": rand_vec(), "metadata": md}])
    except Exception as e:
        st.warning(f"Upsert failed: {e}")

def fetch_all():
    if not index:
        return pd.DataFrame(st.session_state.get("LOCAL_DATA", {}).values())
    try:
        stats = index.describe_index_stats()
        if stats["total_vector_count"] == 0:
            return pd.DataFrame()
        res = index.query(vector=rand_vec(), top_k=1000, include_metadata=True)
        rows = []
        for m in res.matches:
            md = m.metadata or {}
            md["_id"] = m.id
            rows.append(md)
        return pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"Fetch error: {e}")
        return pd.DataFrame()

# ---------------------------------------------------------
# Base Models
# ---------------------------------------------------------
lin_reg = LinearRegression().fit([[0], [50], [100]], [0, 2.5, 5])
vectorizer = CountVectorizer()
svm = SVC()
try:
    X = vectorizer.fit_transform(["good", "excellent", "poor", "bad", "average"])
    svm.fit(X, [1, 1, 0, 0, 0])
except:
    pass

# ---------------------------------------------------------
# Role Selection
# ---------------------------------------------------------
role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client", "Admin"])
current_month = datetime.now().strftime("%B %Y")

# ---------------------------------------------------------
# Manager Portal
# ---------------------------------------------------------
if role == "Manager":
    st.header("Manager Command Center")
    tabs = st.tabs(["Task Management", "AI Insights & Analytics", "Meetings & Feedback", "Leave Management", "360° Overview"])

    # ---- Task Management ----
    with tabs[0]:
        st.subheader("Assign or Reassign Tasks")
        with st.form("assign_task"):
            company = st.text_input("Company")
            department = st.text_input("Department")
            employee = st.text_input("Employee")
            task = st.text_input("Task Title")
            desc = st.text_area("Description")
            deadline = st.date_input("Deadline", value=date.today())
            submit = st.form_submit_button("Assign Task")

            if submit and all([company, employee, task]):
                tid = str(uuid.uuid4())
                md = {
                    "company": company, "department": department, "employee": employee,
                    "task": task, "description": desc, "deadline": deadline.isoformat(),
                    "completion": 0, "marks": 0, "status": "Assigned", "created": now()
                }
                upsert(tid, md)
                st.success("Task assigned successfully.")

        df = fetch_all()
        if not df.empty:
            df["completion"] = pd.to_numeric(df.get("completion", 0), errors="coerce").fillna(0)
            st.subheader("Task Overview")
            st.dataframe(df[["employee", "task", "completion", "status", "deadline"]])
            st.plotly_chart(px.bar(df, x="employee", y="completion", color="department", title="Task Completion by Employee"), use_container_width=True)

            # Reassign
            task_choice = st.selectbox("Select Task to Reassign", df["task"].unique())
            new_emp = st.text_input("New Employee Name")
            reason = st.text_area("Reason for Reassignment")
            if st.button("Reassign Task"):
                r = df[df["task"] == task_choice].iloc[0].to_dict()
                r["employee"] = new_emp
                r["status"] = "Reassigned"
                r["reassigned_reason"] = reason
                r["reassigned_on"] = now()
                upsert(r.get("_id") or str(uuid.uuid4()), r)
                st.success("Task reassigned.")

    # ---- AI Insights ----
    with tabs[1]:
        st.subheader("AI Insights & Analytics Center")
        df = fetch_all()
        if df.empty:
            st.info("No task data available.")
        else:
            df["completion"] = pd.to_numeric(df["completion"], errors="coerce").fillna(0)
            st.metric("Average Completion", f"{df['completion'].mean():.1f}%")
            st.metric("Total Tasks", len(df))
            st.metric("Active Employees", df["employee"].nunique())

            st.plotly_chart(px.histogram(df, x="completion", nbins=10, title="Completion Distribution"))

            query = st.text_input("Ask AI for insights (e.g. 'Who is underperforming this month?')")
            if st.button("Analyze"):
                if HF_AVAILABLE and HF_TOKEN:
                    try:
                        hf = InferenceClient(token=HF_TOKEN)
                        prompt = f"Given this task data summary:\n{df.describe().to_dict()}\nQuestion: {query}\nAnswer in one paragraph."
                        try:
                            res = hf.text_generation(model="mistralai/Mixtral-8x7B-Instruct", prompt=prompt, max_new_tokens=200)
                        except TypeError:
                            res = hf.text_generation(model="mistralai/Mixtral-8x7B-Instruct", inputs=prompt, max_new_tokens=200)
                        out = res["generated_text"] if isinstance(res, dict) else str(res)
                        st.write(out)
                    except Exception as e:
                        st.error(f"AI query failed: {e}")
                else:
                    st.warning("Hugging Face API key not configured.")

    # ---- Meetings & Feedback ----
    with tabs[2]:
        st.subheader("Meeting Scheduler & Feedback")
        with st.form("meet_form"):
            mtitle = st.text_input("Meeting Title")
            mdate = st.date_input("Meeting Date", value=date.today())
            attendees = st.text_area("Attendees (comma separated)")
            upload = st.file_uploader("Upload meeting notes (pdf, csv, xlsx, txt)", type=["pdf","csv","xlsx","txt"])
            submit = st.form_submit_button("Save Meeting")

            if submit and mtitle:
                mid = str(uuid.uuid4())
                md = {"meeting": mtitle, "date": str(mdate), "attendees": attendees, "created": now()}
                if upload:
                    fname = upload.name
                    extracted = ""
                    if fname.endswith(".pdf") and PDF_AVAILABLE:
                        reader = PyPDF2.PdfReader(upload)
                        extracted = "\n".join([p.extract_text() or "" for p in reader.pages])
                    elif fname.endswith(".csv"):
                        extracted = pd.read_csv(upload).to_csv(index=False)
                    elif fname.endswith(".xlsx"):
                        extracted = pd.read_excel(upload).to_csv(index=False)
                    elif fname.endswith(".txt"):
                        extracted = upload.getvalue().decode(errors="ignore")
                    md["file"] = fname
                    md["notes"] = extracted[:30000]
                upsert(mid, md)
                st.success("Meeting saved successfully.")

        df_meet = fetch_all()
        meets = df_meet[df_meet.get("meeting").notna()] if not df_meet.empty else pd.DataFrame()
        if not meets.empty:
            st.dataframe(meets[["meeting", "date", "attendees", "file"]].fillna(""))

    # ---- Leave Management ----
    with tabs[3]:
        st.subheader("Leave Management")
        leaves = st.session_state.get("LEAVES", pd.DataFrame(columns=["Employee","Type","From","To","Reason","Status"]))
        st.dataframe(leaves)
        with st.form("leave_form"):
            emp = st.text_input("Employee")
            typ = st.selectbox("Leave Type", ["Casual","Sick","Earned"])
            f = st.date_input("From")
            t = st.date_input("To")
            reason = st.text_area("Reason")
            sub = st.form_submit_button("Submit Leave")
            if sub:
                new = pd.DataFrame([{"Employee": emp, "Type": typ, "From": str(f), "To": str(t), "Reason": reason, "Status": "Pending"}])
                st.session_state["LEAVES"] = pd.concat([leaves, new], ignore_index=True)
                st.success("Leave submitted.")

    # ---- 360 Overview ----
    with tabs[4]:
        st.subheader("360° Overview")
        df = fetch_all()
        if df.empty:
            st.info("No data.")
        else:
            emp = st.selectbox("Select Employee", df["employee"].unique())
            emp_df = df[df["employee"] == emp]
            st.dataframe(emp_df[["task","completion","marks","status"]])
            avg = emp_df["completion"].mean()
            st.metric("Average Completion", f"{avg:.1f}%")

# ---------------------------------------------------------
# Team Member Portal
# ---------------------------------------------------------
elif role == "Team Member":
    st.header("Team Member Portal")
    name = st.text_input("Enter your name")
    if name:
        df = fetch_all()
        my = df[df["employee"] == name] if not df.empty else pd.DataFrame()
        if my.empty:
            st.info("No tasks assigned.")
        else:
            for _, r in my.iterrows():
                st.markdown(f"### {r.get('task')}")
                comp = st.slider(f"Completion", 0, 100, int(r.get("completion", 0)), key=r.get("_id"))
                if st.button("Update", key=f"upd_{r.get('_id')}"):
                    r["completion"] = comp
                    r["marks"] = float(lin_reg.predict([[comp]])[0])
                    r["status"] = "In Progress" if comp < 100 else "Completed"
                    upsert(r.get("_id") or str(uuid.uuid4()), r)
                    st.success("Progress updated.")
                    safe_rerun()

# ---------------------------------------------------------
# Client Portal
# ---------------------------------------------------------
elif role == "Client":
    st.header("Client Review Panel")
    company = st.text_input("Company Name")
    if company:
        df = fetch_all()
        dfc = df[df["company"].str.lower() == company.lower()] if not df.empty else pd.DataFrame()
        if dfc.empty:
            st.info("No tasks.")
        else:
            st.metric("Overall Completion", f"{dfc['completion'].mean():.1f}%")
            st.dataframe(dfc[["task","employee","completion","status"]])

# ---------------------------------------------------------
# Admin Portal with HR Analytics
# ---------------------------------------------------------
elif role == "Admin":
    st.header("Admin Dashboard & HR Intelligence Layer")
    df = fetch_all()
    if df.empty:
        st.info("No data found.")
    else:
        df["completion"] = pd.to_numeric(df.get("completion", 0), errors="coerce").fillna(0)
        df["marks"] = pd.to_numeric(df.get("marks", 0), errors="coerce").fillna(0)

        st.subheader("Performance Summary")
        st.dataframe(df[["employee","department","task","completion","marks"]])
        st.metric("Average Completion", f"{df['completion'].mean():.1f}%")

        if len(df) > 2:
            km = (df[["completion","marks"]] - df[["completion","marks"]].mean()).fillna(0)
            df["risk"] = (100 - df["completion"]) / 100 + (2 - df["marks"]) / 5
            st.plotly_chart(px.scatter(df, x="completion", y="marks", color="risk", title="HR Intelligence: Performance & Risk"), use_container_width=True)

        top_risk = df.sort_values("risk", ascending=False).head(5)
        st.subheader("Top At-Risk Employees (Explainable)")
        for _, r in top_risk.iterrows():
            reasons = []
            if r["completion"] < 60:
                reasons.append("Low completion")
            if r["marks"] < 2:
                reasons.append("Low marks")
            st.write(f"{r['employee']}: {'; '.join(reasons) or 'Monitor'}")
