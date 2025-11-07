# main.py
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import pandas as pd
from datetime import datetime, date
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go
import uuid
from sklearn.linear_model import LinearRegression as SklearnLinear
import math

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Employee Management", layout="wide")
st.title("🤖 AI-Driven Employee Management System")

# -----------------------------
# PINECONE INITIALIZATION (from Streamlit Secrets)
# -----------------------------
# Accept either grouped secret st.secrets["pinecone"]["api_key"] or top-level key PINECONE_API_KEY.
pinecone_key = None
pinecone_env = None
index_name_secret = None
dimension_secret = None

if "pinecone" in st.secrets:
    pinecone_key = st.secrets["pinecone"].get("api_key")
    pinecone_env = st.secrets["pinecone"].get("environment", "us-east-1")
    index_name_secret = st.secrets["pinecone"].get("index_name", "task")
    dimension_secret = int(st.secrets["pinecone"].get("dimension", 1024))
else:
    # fallback to older top-level keys
    pinecone_key = st.secrets.get("PINECONE_API_KEY") or st.secrets.get("PINECONE_KEY")
    pinecone_env = st.secrets.get("PINECONE_ENV", "us-east-1")
    index_name_secret = st.secrets.get("INDEX_NAME", "task")
    dimension_secret = int(st.secrets.get("DIMENSION", 1024))

if not pinecone_key:
    st.error("Pinecone API key not found in Streamlit secrets. Add it under [pinecone] or PINECONE_API_KEY.")
    st.stop()

try:
    pc = Pinecone(api_key=pinecone_key)
    INDEX_NAME = index_name_secret or "task"
    DIMENSION = dimension_secret or 1024

    if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    index = pc.Index(INDEX_NAME)
except Exception as e:
    st.error(f"Pinecone init failed: {e}")
    st.stop()

# -----------------------------
# HELPERS
# -----------------------------
def now(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def rand_vec(): return np.random.rand(DIMENSION).tolist()

def safe_upsert(md):
    try:
        index.upsert([{
            "id": str(md.get("_id", uuid.uuid4())),
            "values": rand_vec(),
            "metadata": md
        }])
    except Exception as e:
        st.warning(f"Pinecone upsert failed: {e}")

def fetch_all():
    try:
        res = index.query(vector=rand_vec(), top_k=1000, include_metadata=True)
        rows = []
        for m in res.matches:
            md = m.metadata or {}
            md["_id"] = m.id
            rows.append(md)
        return pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"Unable to fetch data: {e}")
        return pd.DataFrame()

def safe_rerun():
    try: st.experimental_rerun()
    except Exception: pass

# -----------------------------
# NOTIFICATIONS (placeholder)
# -----------------------------
def send_notification(email=None, phone=None, subject="Update", msg=""):
    # Placeholder: integrate SMTP or Twilio via st.secrets if desired
    # Example: send_email(email, subject, msg)
    st.info(f"Notification -> {email or phone}: {subject}")

# -----------------------------
# SIMPLE ML MODELS
# -----------------------------
lin_reg = LinearRegression().fit([[0], [50], [100]], [0, 2.5, 5])
log_reg = LogisticRegression().fit([[0], [40], [80], [100]], [0, 0, 1, 1])
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(["excellent work", "needs improvement", "bad performance", "great job", "average"])
svm_clf = SVC().fit(X_train, [1, 0, 0, 1, 0])
rf = RandomForestClassifier().fit(np.array([[10, 2], [50, 1], [90, 0], [100, 0]]), [1, 0, 0, 0])

# -----------------------------
# ROLE SELECTION
# -----------------------------
role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client", "Admin"])
current_month = datetime.now().strftime("%B %Y")

# -----------------------------
# MANAGER DASHBOARD
# -----------------------------
if role == "Manager":
    st.header("👨‍💼 Manager Dashboard")
    tab1, tab2, tab3, tab4 = st.tabs(["Assign Task", "Review Tasks", "360° Overview", "Managerial Actions"])

    # --- Assign Task ---
    with tab1:
        st.subheader("Assign Task")
        with st.form("assign_form"):
            company = st.text_input("Company Name", key="mgr_assign_company")
            employee = st.text_input("Employee Name", key="mgr_assign_employee")
            employee_email = st.text_input("Employee Email (optional)", key="mgr_assign_email")
            employee_phone = st.text_input("Employee Phone (optional)", key="mgr_assign_phone")
            task = st.text_input("Task Title", key="mgr_assign_task")
            desc = st.text_area("Description", key="mgr_assign_desc")
            deadline = st.date_input("Deadline", value=date.today(), key="mgr_assign_deadline")
            submitted = st.form_submit_button("Assign Task")
            if submitted:
                if not (company and employee and task):
                    st.warning("Please fill Company, Employee, Task.")
                else:
                    tid = str(uuid.uuid4())
                    md = {
                        "_id": tid, "type": "task", "company": company, "employee": employee,
                        "email": employee_email, "phone": employee_phone,
                        "task": task, "description": desc,
                        "deadline": deadline.isoformat(), "month": current_month,
                        "completion": 0, "marks": 0, "status": "Assigned",
                        "reviewed": False, "assigned_on": now()
                    }
                    safe_upsert(md)
                    send_notification(employee_email, employee_phone, f"New Task: {task}", f"Task assigned: {task}")
                    st.success(f"Assigned '{task}' to {employee}")

    # --- Review Tasks ---
    with tab2:
        st.subheader("Review Tasks (by Company & Employee)")
        c_name = st.text_input("Company to review", key="mgr_r_company")
        e_name = st.text_input("Employee to review", key="mgr_r_employee")
        if st.button("Load Tasks", key="mgr_load_tasks_btn"):
            if c_name and e_name:
                try:
                    res = index.query(vector=rand_vec(), top_k=500, include_metadata=True,
                                      filter={"company": {"$eq": c_name}, "employee": {"$eq": e_name}})
                    st.session_state["review_tasks"] = [(m.id, m.metadata) for m in res.matches or []]
                    st.success(f"Loaded {len(st.session_state['review_tasks'])} tasks.")
                except Exception as e:
                    st.error(f"Error loading tasks: {e}")
            else:
                st.warning("Enter both company and employee.")
        for tid, r in st.session_state.get("review_tasks", []):
            st.markdown(f"### {r.get('task', 'Unnamed Task')}")
            adj = st.slider(f"Adjust Completion ({r.get('employee', '')})", 0, 100, int(r.get("completion", 0)), key=f"mgr_slider_{tid}")
            comments = st.text_area(f"Manager Comments ({r.get('task', '')})", key=f"mgr_comments_{tid}")
            approve = st.radio(f"Approve {r.get('task', '')}?", ["Yes", "No"], key=f"mgr_approve_{tid}")
            if st.button(f"Finalize Review {r.get('task', '')}", key=f"mgr_finalize_{tid}"):
                sentiment_val = int(svm_clf.predict(vectorizer.transform([comments]))[0])
                sentiment = "Positive" if sentiment_val == 1 else "Negative"
                md = {**r, "completion": adj, "marks": float(lin_reg.predict([[adj]])[0]),
                      "reviewed": True, "comments": comments, "sentiment": sentiment,
                      "approved_by_boss": approve == "Yes", "reviewed_on": now()}
                safe_upsert(md)
                send_notification(r.get("email"), r.get("phone"),
                                  subject=f"Task Review: {r.get('task')}",
                                  msg=f"Your task was reviewed: Completion {adj}%, Marks {md['marks']:.2f}, Sentiment: {sentiment}")
                st.success("Review saved.")
                safe_rerun()

    # --- 360° Overview ---
    with tab3:
        st.subheader("360° Performance Overview")
        df = fetch_all()
        if df.empty:
            st.info("No tasks found.")
        else:
            for col in ["marks", "completion"]:
                if col not in df.columns:
                    df[col] = np.nan
            df["marks"] = pd.to_numeric(df["marks"], errors="coerce")
            df["completion"] = pd.to_numeric(df["completion"], errors="coerce")
            df = df.dropna(subset=["marks", "completion"])
            if len(df) >= 3:
                kmeans = KMeans(n_clusters=3, n_init=10, random_state=42).fit(df[["completion", "marks"]])
                df["cluster"] = kmeans.labels_
                fig = px.scatter(df, x="completion", y="marks", color=df["cluster"].astype(str),
                                 hover_data=["employee", "task"], title="Employee Clusters")
                st.plotly_chart(fig, use_container_width=True)
            st.info(f"Avg marks: {df['marks'].mean():.2f}" if not df.empty else "No marks yet.")

    # --- Managerial Actions placeholder ---
    with tab4:
        st.subheader("Managerial Actions")
        st.write("- Reassign tasks (not implemented in this minimal UI)")
        st.write("- Approve leave (use Admin or leave listing)")

# -----------------------------
# TEAM MEMBER PORTAL
# -----------------------------
elif role == "Team Member":
    st.header("👷 Team Member Portal")
    tab1, tab2, tab3 = st.tabs(["My Tasks", "AI Feedback", "Submit Leave"])

    # --- My Tasks
    with tab1:
        st.subheader("My Tasks")
        company_tm = st.text_input("Company Name", key="tm_tasks_company")
        employee_tm = st.text_input("Your Name", key="tm_tasks_name")
        if st.button("Load Tasks", key="tm_load_btn"):
            try:
                res = index.query(vector=rand_vec(), top_k=500, include_metadata=True,
                                  filter={"company": {"$eq": company_tm}, "employee": {"$eq": employee_tm}})
                st.session_state["tm_tasks"] = [(m.id, m.metadata) for m in res.matches or []]
                st.success(f"Loaded {len(st.session_state.get('tm_tasks',[]))} tasks.")
            except Exception as e:
                st.warning(f"Error: {e}")

        for tid, md in st.session_state.get("tm_tasks", []):
            st.subheader(md.get("task"))
            curr = float(md.get("completion", 0))
            new = st.slider(f"Completion {md.get('task')}", 0, 100, int(curr), key=f"tm_slider_{tid}")
            if st.button(f"Submit {md.get('task')}", key=f"tm_submit_{tid}"):
                marks = float(lin_reg.predict([[new]])[0])
                status = "On Track" if log_reg.predict([[new]])[0] == 1 else "Delayed"
                miss = rf.predict([[new, 0]])[0]
                md2 = {**md, "completion": new, "marks": marks, "status": status,
                       "deadline_risk": "High" if miss else "Low", "submitted_on": now()}
                safe_upsert(md2)
                send_notification(md.get("email"), md.get("phone"), f"Task Update: {md.get('task')}", f"Updated to {new}%")
                st.success("Update saved.")
                safe_rerun()

    # --- AI Feedback (automatic aggregator as alternative) ---
    with tab2:
        st.subheader("AI Feedback Summarization (Automatic)")
        fb_company = st.text_input("Company Name (for summary)", key="tm_fb_company")
        fb_employee = st.text_input("Your Name (for summary)", key="tm_fb_employee")
        if st.button("Load & Analyze Feedback", key="tm_fb_load"):
            if not fb_company or not fb_employee:
                st.warning("Enter company and your name.")
            else:
                try:
                    res = index.query(vector=rand_vec(), top_k=500, include_metadata=True,
                                      filter={"company": {"$eq": fb_company}, "employee": {"$eq": fb_employee}, "reviewed": {"$eq": True}})
                    records = []
                    for m in res.matches or []:
                        md = m.metadata or {}
                        records.append({
                            "task": md.get("task", "Unnamed"),
                            "manager_comments": md.get("comments", ""),
                            "client_comments": md.get("client_comments", "")
                        })
                    df_fb = pd.DataFrame(records)
                    if df_fb.empty:
                        st.info("No feedback found.")
                    else:
                        combined = " ".join(df_fb["manager_comments"].fillna("") + " " + df_fb["client_comments"].fillna(""))
                        blob = TextBlob(combined)
                        polarity = blob.sentiment.polarity
                        subjectivity = blob.sentiment.subjectivity
                        sentiment = "Positive" if polarity > 0.2 else ("Negative" if polarity < -0.2 else "Neutral")
                        st.markdown(f"### Sentiment: {sentiment}")
                        st.markdown(f"Polarity: `{polarity:.3f}` | Subjectivity: `{subjectivity:.3f}`")
                        st.progress((polarity + 1) / 2)
                        st.write("Key themes:", ", ".join(blob.noun_phrases) or "None")
                except Exception as e:
                    st.warning(f"Error loading feedback: {e}")

    # --- Submit Leave ---
    with tab3:
        st.subheader("Submit Leave Request")
        leave_company = st.text_input("Company Name", key="tm_leave_company")
        leave_name = st.text_input("Your Name", key="tm_leave_name")
        start = st.date_input("Start Date", key="tm_leave_start")
        end = st.date_input("End Date", key="tm_leave_end")
        reason = st.text_area("Reason", key="tm_leave_reason")
        if st.button("Submit Leave", key="tm_leave_submit"):
            if not (leave_company and leave_name and reason):
                st.warning("Please fill required fields.")
            else:
                md = {"_id": str(uuid.uuid4()), "type": "leave", "company": leave_company,
                      "employee": leave_name, "start_date": start.isoformat(), "end_date": end.isoformat(),
                      "reason": reason, "status": "Pending", "requested_on": now()}
                safe_upsert(md)
                st.success("Leave submitted.")
                safe_rerun()

# -----------------------------
# CLIENT REVIEW
# -----------------------------
elif role == "Client":
    st.header("Client Review")
    comp = st.text_input("Company Name", key="client_company")
    if st.button("Load Tasks", key="client_load"):
        try:
            res = index.query(vector=rand_vec(), top_k=500, include_metadata=True,
                              filter={"company": {"$eq": comp}})
            st.session_state["client_tasks"] = [(m.id, m.metadata) for m in res.matches or []]
            st.success(f"Loaded {len(st.session_state.get('client_tasks', []))} tasks.")
        except Exception as e:
            st.warning(f"Error: {e}")

    for tid, md in st.session_state.get("client_tasks", []):
        st.markdown(f"### {md.get('task', 'Unnamed Task')} — Employee: {md.get('employee', 'Unknown')}")
        st.write(f"Completion: {md.get('completion', 0)}%")
        st.write(f"Status: {md.get('status', 'In Process')}")
        st.progress(int(md.get('completion', 0)))
        comment = st.text_area(f"Client Feedback ({md.get('task')})", key=f"client_fb_{tid}")
        if st.button(f"Submit Feedback {tid}", key=f"client_submit_{tid}"):
            blob = TextBlob(comment)
            polarity = blob.sentiment.polarity
            sentiment = "Positive" if polarity > 0.1 else ("Negative" if polarity < -0.1 else "Neutral")
            md2 = {**md, "client_reviewed": True, "client_comments": comment,
                   "client_sentiment": sentiment, "client_polarity": polarity, "client_approved_on": now()}
            safe_upsert(md2)
            st.success(f"Saved feedback ({sentiment}).")
            safe_rerun()

# -----------------------------
# ADMIN DASHBOARD
# -----------------------------
elif role == "Admin":
    st.header("Admin Dashboard — HR & Analytics")
    st.markdown("Organization-wide performance, top performers, leave overview, forecasting & alerts.")

    df = fetch_all()
    if df.empty:
        st.info("No data available yet. Create/submit tasks to populate dashboard.")
        # Demo data toggle
        if st.button("Insert demo data for testing"):
            demo = [
                {"_id": str(uuid.uuid4()), "type": "task", "company": "ERP", "employee": "Alice", "department": "Tech", "completion": 90, "marks": 4.6, "month": current_month, "assigned_on": now()},
                {"_id": str(uuid.uuid4()), "type": "task", "company": "ERP", "employee": "Bob", "department": "Design", "completion": 70, "marks": 3.8, "month": current_month, "assigned_on": now()},
                {"_id": str(uuid.uuid4()), "type": "leave", "company": "ERP", "employee": "Clara", "start_date": now(), "end_date": now(), "reason": "Sick", "status": "Pending", "requested_on": now()}
            ]
            for r in demo: safe_upsert(r)
            st.success("Demo data inserted. Refresh to view.")
        st.stop()

    # Ensure numeric fields exist
    for c in ["marks", "completion"]:
        if c not in df.columns:
            df[c] = np.nan
    df["marks"] = pd.to_numeric(df["marks"], errors="coerce")
    df["completion"] = pd.to_numeric(df["completion"], errors="coerce")

    # Department assignment for demo if missing
    if "department" not in df.columns:
        # assign random departments for visualization
        df["department"] = np.random.choice(["Tech", "Design", "HR", "Sales", "Ops"], size=len(df))

    # --- Department performance chart ---
    st.subheader("Department-wise Performance")
    dep_perf = df.groupby("department").agg({"marks": "mean", "completion": "mean"}).reset_index()
    fig_dep = px.bar(dep_perf, x="department", y="marks", color="completion",
                     title="Department Average Marks (color = completion)", text_auto=".2f")
    st.plotly_chart(fig_dep, use_container_width=True)

    # --- Top performers (weighted metric) ---
    st.subheader("Top Performers")
    if "employee" in df.columns:
        df["score"] = 0.7 * df["marks"].fillna(0) + 0.3 * df["completion"].fillna(0)
        top = df.groupby("employee")["score"].mean().reset_index().sort_values("score", ascending=False).head(10)
        st.dataframe(top)
        st.plotly_chart(px.bar(top, x="employee", y="score", title="Top employees (weighted)"), use_container_width=True)

    # --- Leave overview ---
    st.subheader("Leave Overview")
    leaves = df[df["type"] == "leave"] if "type" in df.columns else pd.DataFrame()
    if not leaves.empty:
        st.dataframe(leaves[["employee", "company", "start_date", "end_date", "reason", "status", "requested_on"]])
        leave_chart = leaves.groupby("company").size().reset_index(name="leave_requests")
        st.plotly_chart(px.pie(leave_chart, names="company", values="leave_requests", title="Leave distribution by company"), use_container_width=True)
    else:
        st.info("No leave requests found.")

    # --- Department heatmap by month (pivot) ---
    st.subheader("Department Performance Heatmap")
    if "month" not in df.columns:
        # try to build month from dates if present
        for dcol in ["assigned_on", "reviewed_on", "submitted_on", "requested_on"]:
            if dcol in df.columns:
                try:
                    df["month"] = pd.to_datetime(df[dcol], errors="coerce").dt.to_period("M").astype(str)
                    break
                except Exception:
                    pass
        if "month" not in df.columns:
            df["month"] = current_month

    pivot = df.pivot_table(values="marks", index="department", columns="month", aggfunc="mean").fillna(0)
    st.dataframe(pivot.style.background_gradient(cmap="Greens"))

    # -----------------------------
    # Forecasting & Alerts (linear trend + simple anomaly detection)
    # -----------------------------
    st.subheader("Forecasting & Alerts")
    # Build timeseries with month -> avg marks per department
    df_ts = df.copy()
    # ensure month -> datetime period
    try:
        df_ts["_period"] = pd.to_datetime(df_ts["month"].astype(str) + "-01", errors="coerce")
    except Exception:
        df_ts["_period"] = pd.to_datetime(df_ts.get("month"), errors="coerce")
    if df_ts["_period"].isna().all():
        # fallback: use assigned_on/reviewed_on/submitted_on
        for dcol in ["assigned_on", "reviewed_on", "submitted_on", "requested_on"]:
            if dcol in df_ts.columns:
                df_ts["_period"] = pd.to_datetime(df_ts[dcol], errors="coerce")
                break
    df_ts = df_ts.dropna(subset=["_period"])
    if df_ts.empty:
        st.info("Not enough dated records to forecast.")
    else:
        # choose aggregate level
        agg_by_dept = st.checkbox("Aggregate by department (uncheck for overall)", value=True)
        groups = df_ts.groupby("department") if (agg_by_dept and "department" in df_ts.columns) else [("ALL", df_ts)]

        # plotting
        fig = go.Figure()
        colors = px.colors.qualitative.Dark24
        alerts = []
        FORECAST_MONTHS = st.sidebar.number_input("Forecast months", min_value=1, max_value=12, value=6)
        ALERT_DROP_PCT = st.sidebar.slider("Alert drop % threshold", 5, 80, 20) / 100.0
        Z_THRESH = st.sidebar.slider("Z-score threshold", 1, 4, 2)

        for i, (gname, gdf) in enumerate(groups):
            # prepare monthly aggregated series
            monthly = gdf.groupby(gdf["_period"].dt.to_period("M")).agg({"marks": "mean"}).reset_index()
            if monthly.empty: continue
            monthly["_period"] = monthly["_period"].dt.to_timestamp()
            monthly = monthly.sort_values("_period").reset_index(drop=True)
            monthly["t"] = np.arange(len(monthly))
            X = monthly[["t"]].values
            y = monthly["marks"].values

            # fit linear model if enough points
            if len(X) >= 2:
                lr = SklearnLinear()
                lr.fit(X, y)
                future_t = np.arange(len(monthly), len(monthly) + FORECAST_MONTHS).reshape(-1,1)
                forecast = lr.predict(future_t)
                last_period = monthly["_period"].max()
                future_periods = pd.date_range(start=(last_period + pd.offsets.MonthBegin(1)), periods=FORECAST_MONTHS, freq="MS")
                # trend over existing points
                trend = lr.predict(X)
            else:
                # flat forecast
                forecast = np.array([y[-1]] * FORECAST_MONTHS) if len(y) > 0 else np.array([])
                last_period = monthly["_period"].max() if len(monthly)>0 else pd.Timestamp.now()
                future_periods = pd.date_range(start=(last_period + pd.offsets.MonthBegin(1)), periods=FORECAST_MONTHS, freq="MS")
                trend = y

            # add traces
            fig.add_trace(go.Scatter(x=monthly["_period"], y=monthly["marks"], mode="lines+markers",
                                     name=f"{gname} actual", line=dict(color=colors[i % len(colors)])))
            if len(X) >= 2:
                fig.add_trace(go.Scatter(x=monthly["_period"], y=trend, mode="lines", name=f"{gname} trend",
                                         line=dict(color=colors[i % len(colors)], dash="dash")))
            if forecast.size:
                fig.add_trace(go.Scatter(x=future_periods, y=forecast, mode="lines+markers", name=f"{gname} forecast",
                                         line=dict(color=colors[i % len(colors)], dash="dot")))

            # percent-drop alerts
            for idx in range(1, len(monthly)):
                prev = monthly.loc[idx-1, "marks"]
                curr = monthly.loc[idx, "marks"]
                if pd.notna(prev) and pd.notna(curr) and prev > 0:
                    drop = (prev - curr) / prev
                    if drop >= ALERT_DROP_PCT:
                        alerts.append({"group": gname, "type": "Percent Drop", "month": monthly.loc[idx, "_period"].strftime("%Y-%m"),
                                       "drop_pct": round(drop*100,1), "prev": round(prev,2), "curr": round(curr,2)})

            # z-score anomalies
            if len(y) >= 3:
                mean_y = np.mean(y)
                std_y = np.std(y, ddof=0)
                if std_y > 0:
                    zscores = (y - mean_y) / std_y
                    for j, z in enumerate(zscores):
                        if z <= -Z_THRESH:
                            alerts.append({"group": gname, "type": "Z-Score Low", "month": monthly.loc[j, "_period"].strftime("%Y-%m"),
                                           "zscore": round(z,2), "value": round(monthly.loc[j, "marks"],2)})

        fig.update_layout(title="Actual Marks + Trend + Forecast", xaxis_title="Month", yaxis_title="Average Marks")
        st.plotly_chart(fig, use_container_width=True)

        if alerts:
            st.subheader("Alerts detected")
            st.dataframe(pd.DataFrame(alerts))
            st.warning(f"{len(alerts)} alert(s) detected. Please review.")
        else:
            st.success("No anomalies detected.")

# End of main.py
