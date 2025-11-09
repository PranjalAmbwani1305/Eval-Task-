import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import pandas as pd
import uuid
import json
import logging
import time
from datetime import datetime, date
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# CONFIG & INIT
# -----------------------------
st.set_page_config(page_title="AI Workforce System", layout="wide")
st.title("💼 AI-Powered Workforce Performance & Task Management System")

# Pinecone setup
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
# SIMPLE AI MODELS
# -----------------------------
lin_reg = LinearRegression().fit([[0], [50], [100]], [0, 2.5, 5])
log_reg = LogisticRegression().fit([[0], [40], [80], [100]], [0, 0, 1, 1])

comments = ["excellent work", "needs improvement", "bad performance", "great job", "average"]
sentiments = [1, 0, 0, 1, 0]
vectorizer = CountVectorizer()
svm_clf = SVC().fit(vectorizer.fit_transform(comments), sentiments)
rf = RandomForestClassifier().fit(np.array([[10, 2], [50, 1], [90, 0], [100, 0]]), [0, 1, 0, 0])

# -----------------------------
# LOGGER
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("safe_meta")

# -----------------------------
# HELPERS
# -----------------------------
def safe_meta(md):
    clean = {}
    for k, v in md.items():
        if v is None or (isinstance(v, float) and (pd.isna(v) or np.isnan(v))):
            v = ""
        elif isinstance(v, (datetime, date)):
            v = v.isoformat()
        elif isinstance(v, (np.generic, np.number)):
            v = float(v)
        elif isinstance(v, (list, dict)):
            v = json.dumps(v)
        elif not isinstance(v, (str, int, float, bool)):
            v = str(v)
        clean[k] = v
    return clean

def safe_upsert(id_, vec, md, retries=2, delay=1.5):
    for attempt in range(retries):
        try:
            index.upsert([{"id": id_, "values": vec, "metadata": md}])
            return True
        except Exception as e:
            logger.warning(f"Upsert attempt {attempt+1} failed: {e}")
            time.sleep(delay)
    st.error("❌ Pinecone upsert failed after multiple retries.")
    return False

def fetch_all():
    try:
        res = index.query(vector=rand_vec(), top_k=1000, include_metadata=True)
        rows = []
        for m in res.matches:
            md = m.metadata or {}
            md["_id"] = m.id
            rows.append(md)
        return pd.DataFrame(rows)
    except:
        return pd.DataFrame()

# -----------------------------
# ROLE SELECTION
# -----------------------------
role = st.sidebar.selectbox("🔐 Login as", ["Manager", "Team Member", "Client", "Admin"])
current_month = datetime.now().strftime("%B %Y")

# -----------------------------
# MANAGER COMMAND CENTER
# -----------------------------
if role == "Manager":
    st.header("👑 Manager Command Center")

    df = fetch_all()
    if df.empty:
        st.warning("No tasks found.")
    else:
        for col in ["completion", "marks"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            else:
                df[col] = 0

        # 1️⃣ Task Summary
        st.subheader("📊 Task Summary Overview")
        total = len(df)
        done = len(df[df["completion"] == 100])
        progress = len(df[(df["completion"] > 0) & (df["completion"] < 100)])
        pending = len(df[df["completion"] == 0])
        overdue = len(df[df["status"].astype(str).str.contains("Delayed", case=False, na=False)])

        cols = st.columns(5)
        metrics = [("📋 Total", total), ("🕐 Pending", pending),
                   ("🚧 In Progress", progress), ("✅ Completed", done), ("⚠️ Overdue", overdue)]
        for c, (label, value) in zip(cols, metrics):
            c.metric(label, value)

        # 2️⃣ Team Heatmap
        st.subheader("🌡️ Team Performance Heatmap")
        if "employee" in df.columns:
            pivot = df.pivot_table(index="employee", values="completion", aggfunc="mean")
            fig = px.imshow(pivot, color_continuous_scale="greens", title="Team Performance %")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No employee data available for heatmap.")

        # 3️⃣ AI Alerts
        st.subheader("🤖 AI Alerts & Insights")
        alerts = []
        for _, row in df.iterrows():
            if row["completion"] < 30:
                alerts.append(f"⚠️ Low progress on '{row['task']}' ({row.get('employee','N/A')})")
            if "Delayed" in str(row.get("status", "")):
                alerts.append(f"⏰ '{row['task']}' may miss the deadline.")
            if row["marks"] < 2:
                alerts.append(f"❗ '{row['task']}' underperforming (Marks {row['marks']})")
        if alerts:
            for a in alerts:
                st.warning(a)
        else:
            st.success("✅ No AI alerts detected. Team performance normal.")

        # 4️⃣ Goal Tracker
        st.subheader("🎯 Goal Tracker")
        avg_completion = df["completion"].mean()
        goal = 85
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_completion,
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "green"},
                   "threshold": {"value": goal, "line": {"color": "red", "width": 4}}},
            title={"text": f"Overall Team Progress (Target {goal}%)"}
        ))
        st.plotly_chart(fig, use_container_width=True)

        # 5️⃣ Managerial Adjustments
        st.subheader("🧩 Managerial Adjustments")
        for _, r in df.iterrows():
            with st.expander(f"⚙️ Adjust '{r['task']}' ({r.get('employee')})"):
                new_completion = st.slider(f"Adjust Completion %", 0, 100, int(r["completion"]))
                new_marks = float(lin_reg.predict([[new_completion]])[0])
                note = st.text_input(f"Manager Comment for {r['task']}")
                if st.button(f"Save Adjustment for {r['task']}", key=f"adj_{r['_id']}"):
                    md = safe_meta({
                        **r,
                        "completion": new_completion,
                        "marks": new_marks,
                        "manager_note": note,
                        "reviewed": True,
                        "updated_on": now()
                    })
                    safe_upsert(r["_id"], rand_vec(), md)
                    st.success(f"Updated task '{r['task']}' successfully!")
                    st.rerun()

    # -----------------------------
    # 360° EMPLOYEE FEEDBACK & INSIGHTS
    # -----------------------------
    st.header("💬 360° Employee Performance & Feedback Insights")

    df_feedback = fetch_all()
    if df_feedback.empty:
        st.info("No feedback data available yet.")
    else:
        st.subheader("🧠 AI Sentiment Analysis on Feedback")
        with st.form("feedback_form"):
            emp = st.selectbox("Select Employee", sorted(df_feedback["employee"].dropna().unique()))
            feedback_text = st.text_area("Enter Feedback")
            reviewer = st.text_input("Reviewer Name (optional)")
            source = st.selectbox("Feedback Type", ["Self", "Peer", "Manager"])
            submit_feedback = st.form_submit_button("Submit Feedback")

            if submit_feedback and feedback_text:
                sentiment_val = int(svm_clf.predict(vectorizer.transform([feedback_text]))[0])
                sentiment_label = "Positive" if sentiment_val == 1 else "Negative"
                md = safe_meta({
                    "employee": emp,
                    "feedback_text": feedback_text,
                    "reviewer": reviewer or "Anonymous",
                    "feedback_source": source,
                    "sentiment": sentiment_label,
                    "timestamp": now()
                })
                fid = str(uuid.uuid4())
                safe_upsert(fid, rand_vec(), md)
                st.success(f"✅ Feedback recorded for {emp} ({sentiment_label})")

        feedback_df = df_feedback[df_feedback["sentiment"].isin(["Positive", "Negative"])]
        if not feedback_df.empty:
            score_df = (
                feedback_df.groupby("employee")["sentiment"]
                .apply(lambda x: (x == "Positive").sum() / len(x) * 100)
                .reset_index(name="Positivity (%)")
            )
            st.dataframe(score_df)
            fig = px.bar(score_df, x="employee", y="Positivity (%)", color="Positivity (%)",
                         title="Employee Positivity Scores", range_y=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
            avg_team_sentiment = score_df["Positivity (%)"].mean()
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_team_sentiment,
                gauge={"axis": {"range": [0, 100]},
                       "bar": {"color": "green" if avg_team_sentiment > 60 else "red"}},
                title={"text": "Overall Team Sentiment Health"}
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.subheader("🗂️ Feedback History")
            st.dataframe(
                feedback_df[["employee", "feedback_text", "reviewer", "feedback_source", "sentiment"]],
                use_container_width=True
            )

    # -----------------------------
    # MANAGERIAL ACTIONS & APPROVAL HUB
    # -----------------------------
    st.header("🧾 Managerial Actions & Approvals Hub")

    df_actions = fetch_all()
    if df_actions.empty:
        st.info("No data for actions yet.")
    else:
        employees = df_actions["employee"].dropna().unique()
        selected_emp = st.selectbox("Select Employee for Appraisal", employees)
        emp_data = df_actions[df_actions["employee"] == selected_emp]
        if not emp_data.empty:
            avg_mark = emp_data["marks"].mean()
            avg_comp = emp_data["completion"].mean()
            st.metric("Average Marks", f"{avg_mark:.2f}")
            st.metric("Average Completion", f"{avg_comp:.1f}%")

            # Generate AI appraisal summary
            if st.button(f"🧠 Generate AI Appraisal for {selected_emp}"):
                feedbacks = emp_data.get("manager_note", "")
                summary = (
                    f"**Employee:** {selected_emp}\n"
                    f"**Performance Score:** {avg_mark:.2f}\n"
                    f"**Task Completion:** {avg_comp:.1f}%\n"
                    f"**AI Summary:** {selected_emp} shows consistent performance with a "
                    f"{'strong' if avg_mark > 3 else 'moderate'} track record. "
                    f"Recommended focus areas include timely delivery and continuous improvement.\n"
                )
                st.markdown(summary)
                st.download_button(
                    "📄 Download Appraisal Report",
                    summary,
                    f"{selected_emp}_appraisal.txt"
                )

            # Recognition & Notes
            recog = st.text_area(f"💌 Recognition or Performance Note for {selected_emp}")
            if st.button(f"Send Recognition to {selected_emp}"):
                rid = str(uuid.uuid4())
                md = safe_meta({
                    "employee": selected_emp,
                    "recognition": recog,
                    "sentiment": "Positive",
                    "timestamp": now(),
                    "type": "Recognition"
                })
                safe_upsert(rid, rand_vec(), md)
                st.success(f"Recognition saved for {selected_emp}")
