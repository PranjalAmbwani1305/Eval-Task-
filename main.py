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
            logging.warning(f"Upsert attempt {attempt+1} failed: {e}")
            time.sleep(delay)
    st.error("❌ Pinecone upsert failed after retries.")
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
# MANAGER DASHBOARD (TABS)
# -----------------------------
if role == "Manager":
    st.header("👑 Manager Dashboard")

    df = fetch_all()
    if df.empty:
        st.warning("No tasks found.")
    else:
        for col in ["completion", "marks"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            else:
                df[col] = 0

        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Task Summary", "🌡️ Heatmap", "🤖 AI Alerts", "🎯 Goal Tracker",
            "🧩 Adjustments", "💬 Feedback", "🧾 Appraisals"
        ])

        # TAB 1: Task Summary
        with tab1:
            st.subheader("📊 Task Summary Overview")
            total = len(df)
            done = len(df[df["completion"] == 100])
            progress = len(df[(df["completion"] > 0) & (df["completion"] < 100)])
            pending = len(df[df["completion"] == 0])
            overdue = len(df[df["status"].astype(str).str.contains("Delayed", case=False, na=False)])
            cols = st.columns(5)
            for c, (label, value) in zip(cols, [
                ("📋 Total", total), ("🕐 Pending", pending),
                ("🚧 In Progress", progress), ("✅ Completed", done),
                ("⚠️ Overdue", overdue)
            ]):
                c.metric(label, value)
            fig_pie = px.pie(
                pd.DataFrame({"Status": ["Completed", "In Progress", "Pending", "Overdue"],
                              "Count": [done, progress, pending, overdue]}),
                values="Count", names="Status", title="Task Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)

        # TAB 2: Heatmap
        with tab2:
            st.subheader("🌡️ Team Performance Heatmap")
            if "employee" in df.columns:
                pivot = df.pivot_table(index="employee", values="completion", aggfunc="mean")
                fig = px.imshow(pivot, color_continuous_scale="greens", title="Team Performance %")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No employee data available.")

        # TAB 3: AI Alerts
        with tab3:
            st.subheader("🤖 AI Alerts & Insights")
            alerts = []
            for _, r in df.iterrows():
                if r["completion"] < 30:
                    alerts.append(f"⚠️ Low progress on '{r['task']}' ({r.get('employee')})")
                if "Delayed" in str(r.get("status", "")):
                    alerts.append(f"⏰ '{r['task']}' may miss the deadline.")
                if r["marks"] < 2:
                    alerts.append(f"❗ Underperforming task '{r['task']}' (Marks {r['marks']})")
            if alerts:
                for a in alerts:
                    st.warning(a)
            else:
                st.success("✅ No performance issues detected.")

        # TAB 4: Goal Tracker
        with tab4:
            st.subheader("🎯 Goal Tracker")
            avg = df["completion"].mean()
            goal = 85
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg,
                gauge={"axis": {"range": [0, 100]},
                       "bar": {"color": "green"},
                       "threshold": {"value": goal, "line": {"color": "red", "width": 4}}},
                title={"text": f"Team Progress (Goal: {goal}%)"}))
            st.plotly_chart(fig, use_container_width=True)

        # TAB 5: Managerial Adjustments
        with tab5:
            st.subheader("🧩 Managerial Adjustments")
            for _, r in df.iterrows():
                with st.expander(f"⚙️ Adjust '{r['task']}' ({r.get('employee')})"):
                    new_completion = st.slider("Completion %", 0, 100, int(r["completion"]))
                    new_marks = float(lin_reg.predict([[new_completion]])[0])
                    note = st.text_input("Manager Note", key=f"note_{r['_id']}")
                    if st.button("Save", key=f"save_{r['_id']}"):
                        md = safe_meta({**r, "completion": new_completion,
                                        "marks": new_marks, "manager_note": note,
                                        "updated_on": now()})
                        safe_upsert(r["_id"], rand_vec(), md)
                        st.success(f"Updated '{r['task']}'")

        # TAB 6: Feedback
        with tab6:
            st.subheader("💬 360° Feedback & Sentiment")
            df_feedback = fetch_all()
            if df_feedback.empty:
                st.info("No feedback found.")
            else:
                with st.form("feedback_form"):
                    emp = st.selectbox("Select Employee", sorted(df_feedback["employee"].dropna().unique()))
                    text = st.text_area("Feedback")
                    src = st.selectbox("Source", ["Self", "Peer", "Manager"])
                    submit = st.form_submit_button("Submit Feedback")
                    if submit and text:
                        sent = int(svm_clf.predict(vectorizer.transform([text]))[0])
                        sentiment = "Positive" if sent == 1 else "Negative"
                        fid = str(uuid.uuid4())
                        md = safe_meta({"employee": emp, "feedback_text": text,
                                        "feedback_source": src, "sentiment": sentiment,
                                        "timestamp": now()})
                        safe_upsert(fid, rand_vec(), md)
                        st.success(f"Feedback saved ({sentiment})")

                feedback_df = df_feedback[df_feedback["sentiment"].isin(["Positive", "Negative"])]
                if not feedback_df.empty:
                    score_df = feedback_df.groupby("employee")["sentiment"].apply(
                        lambda x: (x == "Positive").sum() / len(x) * 100
                    ).reset_index(name="Positivity (%)")
                    st.plotly_chart(px.bar(score_df, x="employee", y="Positivity (%)",
                                           color="Positivity (%)", title="Employee Sentiment"),
                                    use_container_width=True)
                    avg_sent = score_df["Positivity (%)"].mean()
                    st.metric("Average Team Sentiment", f"{avg_sent:.1f}%")
                    st.subheader("🗂️ Feedback History")
                    expected = ["employee", "feedback_text", "feedback_source", "sentiment"]
                    available = [c for c in expected if c in feedback_df.columns]
                    st.dataframe(feedback_df[available], use_container_width=True)

        # TAB 7: Appraisals
        with tab7:
            st.subheader("🧾 Appraisals")
            employees = df["employee"].dropna().unique()
            emp = st.selectbox("Select Employee", employees)
            emp_df = df[df["employee"] == emp]
            if not emp_df.empty:
                avg_m, avg_c = emp_df["marks"].mean(), emp_df["completion"].mean()
                st.metric("Avg Marks", f"{avg_m:.2f}")
                st.metric("Avg Completion", f"{avg_c:.1f}%")
                if st.button(f"🧠 Generate Appraisal for {emp}"):
                    report = (f"Employee: {emp}\nPerformance: {avg_m:.2f}\nCompletion: {avg_c:.1f}%\n"
                              f"Summary: {'Strong' if avg_m > 3 else 'Moderate'} performer "
                              f"with {'excellent' if avg_c > 80 else 'improving'} consistency.")
                    st.text_area("Appraisal Report", report)
                    st.download_button("📄 Download Appraisal", report, f"{emp}_appraisal.txt")

# -----------------------------
# TEAM MEMBER (AUTO FEEDBACK)
# -----------------------------
elif role == "Team Member":
    st.header("👷 Team Member Panel — Auto Feedback Enabled")
    company = st.text_input("Company Name")
    employee = st.text_input("Your Name")

    if st.button("🔄 Load My Tasks"):
        res = index.query(
            vector=rand_vec(),
            top_k=500,
            include_metadata=True,
            filter={"employee": {"$eq": employee}}
        )
        st.session_state["tasks"] = [(m.id, m.metadata) for m in res.matches or []]
        st.success(f"Loaded {len(st.session_state['tasks'])} tasks.")

    for tid, md in st.session_state.get("tasks", []):
        st.subheader(md.get("task"))
        st.write(md.get("description"))
        curr = float(md.get("completion", 0))
        new = st.slider(f"Completion for {md.get('task')}", 0, 100, int(curr))
        if st.button(f"Submit {md.get('task')}", key=tid):
            marks = float(lin_reg.predict([[new]])[0])
            track = "On Track" if log_reg.predict([[new]])[0] == 1 else "Delayed"
            miss = rf.predict([[new, 0]])[0]

            if new >= 90:
                feedback = "🌟 Excellent work! You’ve nearly completed the task."
                sentiment = "Positive"
            elif 60 <= new < 90:
                feedback = "👍 Good progress! Stay consistent."
                sentiment = "Positive"
            elif 30 <= new < 60:
                feedback = "🕐 Halfway there. Keep pushing."
                sentiment = "Neutral"
            else:
                feedback = "⚠️ Low progress. Please focus on deadlines."
                sentiment = "Negative"

            md2 = safe_meta({
                **md,
                "completion": new,
                "marks": marks,
                "status": track,
                "deadline_risk": "High" if miss else "Low",
                "submitted_on": now(),
                "auto_feedback": feedback,
                "auto_sentiment": sentiment
            })
            safe_upsert(tid, rand_vec(), md2)
            st.success(f"Updated {md.get('task')} ({track})")
            st.info(f"🤖 **AI Feedback:** {feedback}")

# -----------------------------
# CLIENT
# -----------------------------
elif role == "Client":
    st.header("🤝 Client Review Panel")
    company = st.text_input("Company Name")
    if st.button("🔄 Load Completed Tasks"):
        res = index.query(vector=rand_vec(), top_k=500, include_metadata=True,
                          filter={"reviewed": {"$eq": True}})
        st.session_state["ctasks"] = [(m.id, m.metadata) for m in res.matches or []]
        st.success(f"Loaded {len(st.session_state['ctasks'])} tasks.")
    for tid, md in st.session_state.get("ctasks", []):
        st.subheader(md.get("task"))
        st.write(f"Employee: {md.get('employee')}")
        st.write(f"Completion: {md.get('completion')}%")
        comment = st.text_area(f"Feedback for {md.get('task')}", key=f"c_{tid}")
        if st.button(f"Approve {md.get('task')}", key=f"approve_{tid}"):
            md2 = safe_meta({**md, "client_reviewed": True,
                             "client_comments": comment, "client_approved_on": now()})
            safe_upsert(tid, rand_vec(), md2)
            st.success(f"Approved {md.get('task')}")

# -----------------------------
# ADMIN
# -----------------------------
elif role == "Admin":
    st.header("🧮 Admin Dashboard")
    df = fetch_all()
    if df.empty:
        st.warning("No data found.")
    else:
        for c in ["completion", "marks"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        st.subheader("Top Employees by Marks")
        top = df.groupby("employee")["marks"].mean().reset_index().sort_values("marks", ascending=False).head(5)
        st.dataframe(top)
        st.subheader("K-Means Performance Clustering")
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=min(3, len(df)), n_init=10).fit(df[["completion", "marks"]].fillna(0))
        df["cluster"] = km.labels_
        st.plotly_chart(px.scatter(df, x="completion", y="marks", color=df["cluster"].astype(str),
                                   hover_data=["employee", "task"], title="Performance Clusters"))
        st.download_button("📥 Download All Data", df.to_csv(index=False), "tasks.csv")
