# main.py — AI Workforce Intelligence Platform (Enterprise Edition)

import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import pandas as pd
import uuid, json, time
from datetime import datetime, date
from sklearn.linear_model import LinearRegression
from huggingface_hub import InferenceClient
import plotly.express as px

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="AI Workforce Intelligence Platform", layout="wide")
st.title("AI Workforce Intelligence Platform")

# ---------------------------------------------------------
# CONNECTIONS
# ---------------------------------------------------------
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
HF_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")

INDEX_NAME = "task"
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
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            with st.spinner("Creating Pinecone index..."):
                while True:
                    desc = pc.describe_index(INDEX_NAME)
                    if desc["status"].get("ready"):
                        break
                    time.sleep(2)
        index = pc.Index(INDEX_NAME)
        st.caption(f"Connected to Pinecone index: {INDEX_NAME}")
    except Exception as e:
        st.warning(f"Pinecone init failed; running local only. ({e})")
else:
    st.warning("Pinecone API key missing — running local only.")

# ---------------------------------------------------------
# HELPERS
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

def upsert_data(id_, md):
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

# ML model for performance mapping
lin_reg = LinearRegression().fit([[0], [50], [100]], [0, 2.5, 5])

# ---------------------------------------------------------
# ROLE SELECTION
# ---------------------------------------------------------
role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client", "Admin"])
current_month = datetime.now().strftime("%B %Y")

# ---------------------------------------------------------
# MANAGER DASHBOARD
# ---------------------------------------------------------
if role == "Manager":
    st.header("Manager Dashboard")
    tabs = st.tabs(["Task Management", "AI Insights", "Meetings & Feedback", "Leave Management", "360 Overview"])

    # --- Task Management ---
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

            if submit and all([employee, task]):
                tid = str(uuid.uuid4())
                md = {
                    "company": company,
                    "department": department or "General",
                    "employee": employee,
                    "task": task,
                    "description": desc,
                    "deadline": deadline.isoformat(),
                    "completion": 0,
                    "marks": 0,
                    "status": "Assigned",
                    "created": now(),
                }
                upsert_data(tid, md)
                st.success(f"Task '{task}' assigned to {employee}.")

        df = fetch_all()
        if not df.empty:
            df["completion"] = pd.to_numeric(df.get("completion", 0), errors="coerce").fillna(0)
            st.dataframe(df[["employee", "task", "completion", "status", "deadline"]])

            # Reassign Section
            st.markdown("---")
            st.subheader("Reassign Task")
            task_choice = st.selectbox("Select Task", df["task"].unique())
            new_emp = st.text_input("New Employee Name")
            reason = st.text_area("Reason for Reassignment")
            if st.button("Reassign Task"):
                r = df[df["task"] == task_choice].iloc[0].to_dict()
                r.update({
                    "employee": new_emp,
                    "status": "Reassigned",
                    "reassigned_reason": reason,
                    "reassigned_on": now()
                })
                upsert_data(r.get("_id") or str(uuid.uuid4()), r)
                st.success("Task reassigned successfully.")

    # --- AI Insights ---
    with tabs[1]:
        st.subheader("AI Insights & Analytics Center")
        df = fetch_all()
        if df.empty:
            st.info("No data available.")
        else:
            df["completion"] = pd.to_numeric(df["completion"], errors="coerce").fillna(0)
            st.metric("Average Completion", f"{df['completion'].mean():.1f}%")
            st.metric("Total Tasks", len(df))
            st.metric("Active Employees", df["employee"].nunique())

            query = st.text_input("Ask AI for insights (e.g. 'Who is underperforming this month?')")
            if st.button("Analyze"):
                if HF_TOKEN:
                    client = InferenceClient(token=HF_TOKEN)
                    prompt = f"Task data summary: {df.describe().to_dict()}\nQuestion: {query}\nAnswer:"
                    try:
                        res = client.text_generation(model="mistralai/Mixtral-8x7B-Instruct", prompt=prompt, max_new_tokens=150)
                        st.write(res)
                    except Exception as e:
                        st.error(f"AI query failed: {e}")
                else:
                    st.warning("Hugging Face API key missing.")

    # --- Meetings ---
    with tabs[2]:
        st.subheader("Meeting Notes & Feedback")
        title = st.text_input("Meeting Title")
        notes = st.text_area("Summary")
        date_val = st.date_input("Meeting Date", value=date.today())
        if st.button("Save Meeting"):
            mid = str(uuid.uuid4())
            upsert_data(mid, {"meeting": title, "notes": notes, "date": str(date_val), "created": now()})
            st.success("Meeting saved successfully.")

    # --- Leave Management ---
    with tabs[3]:
        st.subheader("Leave Requests")
        emp = st.text_input("Employee")
        typ = st.selectbox("Type", ["Casual", "Sick", "Earned"])
        f = st.date_input("From")
        t = st.date_input("To")
        reason = st.text_area("Reason")
        if st.button("Submit Leave"):
            lid = str(uuid.uuid4())
            md = {"employee": emp, "type": typ, "from": str(f), "to": str(t), "reason": reason, "status": "Pending", "submitted": now()}
            upsert_data(lid, md)
            st.success("Leave request submitted.")

    # --- 360 Overview ---
    with tabs[4]:
        st.subheader("Employee Performance Overview")
        df = fetch_all()
        if df.empty:
            st.info("No data found.")
        else:
            emp = st.selectbox("Select Employee", df["employee"].unique())
            emp_df = df[df["employee"] == emp]
            st.dataframe(emp_df[["task", "completion", "marks", "status"]])
            st.metric("Avg Completion", f"{emp_df['completion'].mean():.1f}%")

# ---------------------------------------------------------
# TEAM MEMBER DASHBOARD (WITH FEEDBACK INSIGHTS)
# ---------------------------------------------------------
elif role == "Team Member":
    st.header("Team Member Portal")

    name = st.text_input("Enter your name")
    if name:
        df = fetch_all()
        my = df[df["employee"].str.lower() == name.lower()] if not df.empty else pd.DataFrame()

        if my.empty:
            st.info("No tasks assigned yet.")
        else:
            st.subheader("My Tasks")
            for _, r in my.iterrows():
                st.markdown(f"**Task:** {r.get('task', '')} | **Status:** {r.get('status', '')}")
                comp = st.slider("Completion %", 0, 100, int(r.get("completion", 0)), key=r.get("_id"))
                if st.button(f"Update {r.get('task')}", key=f"u_{r.get('_id')}"):
                    r["completion"] = comp
                    r["marks"] = float(lin_reg.predict([[comp]])[0])
                    r["status"] = "In Progress" if comp < 100 else "Completed"
                    upsert_data(r.get("_id") or str(uuid.uuid4()), r)
                    st.success(f"Task '{r.get('task')}' updated successfully.")

            # -------------------------
            # FEEDBACK INSIGHT SECTION
            # -------------------------
            st.markdown("---")
            st.subheader("Feedback & AI Insights")

            fb_df = my[my[["client_feedback", "manager_comments"]].notna().any(axis=1)].copy()

            if fb_df.empty:
                st.info("No feedback received yet.")
            else:
                fb_df["feedback_text"] = fb_df.apply(
                    lambda x: x.get("client_feedback") or x.get("manager_comments") or "",
                    axis=1,
                )
                fb_df["source"] = fb_df.apply(
                    lambda x: "Client" if pd.notna(x.get("client_feedback")) else "Manager",
                    axis=1,
                )

                st.dataframe(fb_df[["task", "source", "feedback_text"]], use_container_width=True)

                if HF_TOKEN:
                    try:
                        client = InferenceClient(token=HF_TOKEN)
                        all_feedback = "\n".join(fb_df["feedback_text"].dropna().tolist())
                        if all_feedback.strip():
                            st.markdown("### AI Sentiment Overview")
                            prompt = f"Analyze these feedbacks for sentiment (Positive/Neutral/Negative) and summarize strengths & improvement points:\n\n{all_feedback}"
                            res = client.text_generation(model="mistralai/Mixtral-8x7B-Instruct", prompt=prompt, max_new_tokens=250)
                            out = res.get("generated_text") if isinstance(res, dict) else str(res)
                            st.success("AI Insight Summary:")
                            st.write(out)
                    except Exception as e:
                        st.warning(f"AI analysis unavailable ({e})")
                else:
                    st.warning("Hugging Face API key missing for AI insights.")

# ---------------------------------------------------------
# CLIENT DASHBOARD
# ---------------------------------------------------------
elif role == "Client":
    st.header("Client Review & Insight Center")
    company = st.text_input("Enter Company Name")
    if company:
        df = fetch_all()
        df_client = df[df["company"].str.lower() == company.lower()] if not df.empty else pd.DataFrame()
        if df_client.empty:
            st.info("No data found.")
        else:
            avg_completion = df_client["completion"].mean()
            avg_marks = df_client["marks"].mean()
            st.metric("Avg Completion", f"{avg_completion:.1f}%")
            st.metric("Avg Rating", f"{avg_marks:.2f}")
            st.dataframe(df_client[["task", "employee", "completion", "status"]])

            task_sel = st.selectbox("Task", df_client["task"].unique())
            fb = st.text_area("Feedback")
            rating = st.slider("Rating (1–5)", 1, 5, 3)
            if st.button("Submit Feedback"):
                row = df_client[df_client["task"] == task_sel].iloc[0].to_dict()
                row.update({"client_feedback": fb, "client_rating": rating, "client_reviewed": True})
                upsert_data(row.get("_id") or str(uuid.uuid4()), row)
                st.success("Feedback submitted.")

# ---------------------------------------------------------
# ADMIN DASHBOARD
# ---------------------------------------------------------
elif role == "Admin":
    st.header("Admin Dashboard & HR Intelligence")
    df = fetch_all()
    if df.empty:
        st.info("No data available.")
    else:
        df["completion"] = pd.to_numeric(df["completion"], errors="coerce").fillna(0)
        df["marks"] = pd.to_numeric(df["marks"], errors="coerce").fillna(0)

        st.subheader("Performance Summary")
        st.dataframe(df[["employee", "department", "task", "completion", "marks"]])
        avg_c = df["completion"].mean()
        st.metric("Average Completion", f"{avg_c:.1f}%")

        if len(df) > 2:
            df["risk"] = (100 - df["completion"]) / 100 + (2 - df["marks"]) / 5
            fig = px.scatter(df, x="completion", y="marks", color="risk",
                             hover_data=["employee", "task"], title="Performance & Risk Overview")
            st.plotly_chart(fig, use_container_width=True)
