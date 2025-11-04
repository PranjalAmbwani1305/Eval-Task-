import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import uuid
from datetime import datetime, date
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import pandas as pd
import io
import plotly.express as px

# ---------------------------
# Config / Init
# ---------------------------
st.set_page_config(page_title="Task Completion & Review", layout="wide")
st.title("AI-Powered Task Completion & Review System — Extended")

# Pinecone init
PC_API_KEY = st.secrets["PINECONE_API_KEY"]
pc = Pinecone(api_key=PC_API_KEY)
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

# ---------------------------
# Simple AI components (demo)
# ---------------------------
lin_reg = LinearRegression(); lin_reg.fit([[0], [100]], [0, 5])
log_reg = LogisticRegression(); log_reg.fit([[0], [50], [100]], [0, 0, 1])
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(["good work", "excellent", "needs improvement", "bad performance"])
y_train = [1, 1, 0, 0]
svm_clf = SVC(); svm_clf.fit(X_train, y_train)

# ---------------------------
# Helpers
# ---------------------------
def now_ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def random_vector(dim=DIMENSION):
    return np.random.rand(dim).tolist()

def safe_metadata(md: dict):
    """Sanitize metadata to be Pinecone-safe (no None, no numpy types, dates -> iso)."""
    clean = {}
    for k, v in md.items():
        if v is None:
            v = ""
        elif isinstance(v, (np.generic,)):
            v = v.item()
        elif isinstance(v, (datetime, date)):
            v = v.isoformat()
        elif isinstance(v, (float, int, str, bool)):
            pass
        else:
            v = str(v)
        clean[k] = v
    return clean

def parse_date_safe(s):
    """Try to parse an ISO date/time or 'YYYY-MM-DD' string; return datetime or None."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        try:
            return datetime.strptime(s, "%Y-%m-%d")
        except Exception:
            return None

def task_to_row(md: dict):
    """Convert metadata dict to flat row for DataFrame/export."""
    row = md.copy()
    # ensure certain keys exist
    for k in ["company","employee","task","month","deadline","completion","marks","status","reviewed","client_reviewed","assigned_on","submitted_on","client_approved_on","manager_comments","client_comments","sentiment"]:
        row.setdefault(k, "")
    return row

def compute_perf_category(avg):
    if avg >= 4:
        return "High"
    if avg >= 2.5:
        return "Medium"
    return "Low"

# ---------------------------
# UI: Role selection
# ---------------------------
role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client", "Admin"])
current_month = datetime.now().strftime("%B %Y")

# ---------------------------
# Shared: quick function to fetch all metadata (used by dashboard/export)
# ---------------------------
def fetch_all_tasks(top_k=1000):
    res = index.query(vector=random_vector(), top_k=top_k, include_metadata=True)
    matches = res.matches or []
    rows = []
    for m in matches:
        md = m.metadata or {}
        md_row = task_to_row(md)
        md_row["_id"] = m.id
        rows.append(md_row)
    if rows:
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(columns=["_id"])
    return df

# ---------------------------
# Manager: assign and final review (after client approval)
# ---------------------------
if role == "Manager":
    st.header("Manager — Assign Tasks & Final Review")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Assign New Task")
        with st.form("assign_form"):
            company = st.text_input("Company Name")
            employee = st.text_input("Employee Name")
            task_title = st.text_input("Task Title")
            description = st.text_area("Task Description")
            deadline = st.date_input("Deadline", value=date.today())
            month = st.selectbox("Month", [current_month])
            submit_assign = st.form_submit_button("Assign Task")
            if submit_assign:
                if not (company and employee and task_title):
                    st.error("Please fill Company, Employee, and Task Title.")
                else:
                    tid = str(uuid.uuid4())
                    md = safe_metadata({
                        "company": company,
                        "employee": employee,
                        "task": task_title,
                        "description": description,
                        "deadline": deadline.isoformat(),
                        "month": month,
                        "completion": 0.0,
                        "marks": 0.0,
                        "status": "Assigned",
                        "reviewed": False,
                        "client_reviewed": False,
                        "assigned_on": now_ts(),
                        "notifications": ""  # placeholder for notifications log
                    })
                    values = random_vector()
                    try:
                        index.upsert([{"id": tid, "values": values, "metadata": md}])
                        st.success(f"Assigned {task_title} to {employee}.")
                        # simulate notification by adding a notifications field with timestamp
                        notif = f"assigned:{now_ts()}"
                        md2 = safe_metadata({**md, "notifications": notif})
                        index.upsert([{"id": tid, "values": values, "metadata": md2}])
                    except Exception as e:
                        st.error("Failed to assign task — check logs.")
                        st.exception(e)

    with col2:
        st.subheader("Load Client-Approved Tasks for Final Review")
        company_review = st.text_input("Company (for review)", key="mgr_company_review")
        if st.button("Load client-approved tasks"):
            if not company_review:
                st.error("Provide company name.")
            else:
                res = index.query(
                    vector=random_vector(),
                    top_k=500,
                    include_metadata=True,
                    include_values=True,
                    filter={"company": {"$eq": company_review}, "client_reviewed": {"$eq": True}, "reviewed": {"$eq": False}}
                )
                matches = res.matches or []
                st.session_state["mgr_review_matches"] = matches
                st.info(f"Loaded {len(matches)} tasks awaiting final review.")

        matches = st.session_state.get("mgr_review_matches", [])
        if matches:
            for m in matches:
                md = m.metadata or {}
                st.markdown(f"### {md.get('task','?')}")
                st.write(f"Employee: {md.get('employee','?')}")
                st.write(f"Client comments: {md.get('client_comments','')}")
                st.write(f"Submitted on: {md.get('submitted_on','')}")
                completion_val = float(md.get("completion", 0) or 0)
                final_marks = st.number_input(f"Final marks (0-5) for {md.get('task')}", min_value=0.0, max_value=5.0, step=0.1, key=f"fm_{m.id}")
                final_comments = st.text_area(f"Manager comments for {md.get('task')}", key=f"mc_{m.id}")
                if st.button(f"Finalize Review: {md.get('task')}", key=f"final_{m.id}"):
                    updated_md = safe_metadata({
                        **md,
                        "marks": float(final_marks),
                        "status": md.get("status", "Completed"),
                        "reviewed": True,
                        "manager_comments": final_comments,
                        "reviewed_on": now_ts()
                    })
                    values_to_use = m.values if hasattr(m, "values") and m.values and len(m.values) == DIMENSION else random_vector()
                    try:
                        index.upsert([{"id": m.id, "values": values_to_use, "metadata": updated_md}])
                        st.success(f"Finalized review for {md.get('task')}.")
                        # optionally remove from session list
                        st.session_state["mgr_review_matches"] = [mm for mm in st.session_state["mgr_review_matches"] if mm.id != m.id]
                    except Exception as e:
                        st.error("Failed to finalize review.")
                        st.exception(e)

    st.markdown("---")
    st.subheader("Manager Filters & Search")
    df_all = fetch_all_tasks()
    if not df_all.empty:
        q_company = st.text_input("Filter company (optional)", key="mgr_search_company")
        q_employee = st.text_input("Filter employee (optional)", key="mgr_search_employee")
        q_status = st.selectbox("Filter status", ["All", "Assigned", "On Track", "Delayed", "Completed"], index=0)
        filtered = df_all.copy()
        if q_company:
            filtered = filtered[filtered["company"].str.contains(q_company, case=False, na=False)]
        if q_employee:
            filtered = filtered[filtered["employee"].str.contains(q_employee, case=False, na=False)]
        if q_status != "All":
            filtered = filtered[filtered[filtered["status"].str.contains(q_status, na=False)]]
        st.dataframe(filtered)

        # Export
        buf = io.BytesIO()
        csv = filtered.to_csv(index=False)
        st.download_button("Download filtered CSV", csv, file_name="tasks_filtered.csv", mime="text/csv")

# ---------------------------
# Team Member: update progress (persistent, auto reload)
# ---------------------------
elif role == "Team Member":
    st.header("Team Member — Load & Submit Progress")
    company = st.text_input("Company Name")
    employee = st.text_input("Your Name (exact)")
    month_input = st.text_input("Month (e.g., 'November 2025')", value=current_month)

    if st.button("Load My Tasks"):
        if not (company and employee):
            st.error("Enter company and employee name.")
        else:
            res = index.query(
                vector=random_vector(),
                top_k=500,
                include_metadata=True,
                include_values=True,
                filter={"company": {"$eq": company}, "employee": {"$eq": employee}, "month": {"$eq": month_input}}
            )
            st.session_state["tm_tasks"] = [(m.id, m.metadata, m.values) for m in (res.matches or [])]
            st.success(f"Loaded {len(st.session_state.get('tm_tasks', []))} tasks.")

    tm_tasks = st.session_state.get("tm_tasks", [])
    if tm_tasks:
        for tid, md, vals in tm_tasks:
            st.markdown(f"### {md.get('task','?')}")
            st.write(md.get("description", ""))
            dead = md.get("deadline", "")
            deadline_dt = parse_date_safe(dead)
            overdue = False
            if deadline_dt:
                if datetime.now() > deadline_dt:
                    overdue = True
                    st.warning(f"Deadline passed: {deadline_dt.date()}")
                else:
                    st.write(f"Deadline: {deadline_dt.date()}")

            completion_current = float(md.get("completion", 0) or 0)
            st.write(f"Current completion: {completion_current}%")
            new_completion = st.slider(f"Update completion for {md.get('task')}", 0, 100, int(completion_current), key=f"tm_{tid}")

            if st.button(f"Submit Progress for {md.get('task')}", key=f"tm_submit_{tid}"):
                marks = float(lin_reg.predict([[new_completion]])[0])
                status_pred = log_reg.predict([[new_completion]])[0]
                status_txt = "On Track" if status_pred == 1 else "Delayed"
                updated_md = safe_metadata({
                    **md,
                    "completion": float(new_completion),
                    "marks": float(marks),
                    "status": status_txt,
                    "submitted_on": now_ts(),
                    # reset client_reviewed since team changed progress
                    "client_reviewed": False,
                    "client_comments": ""
                })
                values_to_use = vals if vals and len(vals) == DIMENSION else random_vector()
                try:
                    index.upsert([{"id": tid, "values": values_to_use, "metadata": updated_md}])
                    st.success(f"Submitted progress for {md.get('task')} at {updated_md['submitted_on']}")
                    # update session_state
                    for i, (xid, xmd, xvals) in enumerate(st.session_state["tm_tasks"]):
                        if xid == tid:
                            st.session_state["tm_tasks"][i] = (xid, updated_md, xvals)
                            break
                except Exception as e:
                    st.error("Failed to submit progress.")
                    st.exception(e)

# ---------------------------
# Client: view completed tasks & approve (client_approved -> manager sees)
# ---------------------------
elif role == "Client":
    st.header("Client — Review & Approve Completed Tasks")
    company = st.text_input("Company Name (for client review)")
    if st.button("Load Completed Tasks"):
        if not company:
            st.error("Provide company name.")
        else:
            res = index.query(
                vector=random_vector(),
                top_k=500,
                include_metadata=True,
                include_values=True,
                filter={"company": {"$eq": company}, "completion": {"$gte": 99}, "client_reviewed": {"$eq": False}}
            )
            st.session_state["client_tasks"] = [(m.id, m.metadata, m.values) for m in (res.matches or [])]
            st.success(f"Loaded {len(st.session_state.get('client_tasks', []))} completed tasks.")

    client_tasks = st.session_state.get("client_tasks", [])
    if client_tasks:
        for tid, md, vals in client_tasks:
            st.markdown(f"### {md.get('task','?')}")
            st.write(f"Employee: {md.get('employee','?')}")
            st.write(f"Completion: {md.get('completion',0)}%")
            st.write(f"Submitted on: {md.get('submitted_on','N/A')}")
            client_comments = st.text_area(f"Your feedback for {md.get('task')}", key=f"cc_{tid}")
            if st.button(f"Client Approve {md.get('task')}", key=f"capprove_{tid}"):
                updated_md = safe_metadata({**md, "client_reviewed": True, "client_comments": client_comments, "client_approved_on": now_ts()})
                values_to_use = vals if vals and len(vals) == DIMENSION else random_vector()
                try:
                    index.upsert([{"id": tid, "values": values_to_use, "metadata": updated_md}])
                    st.success(f"Task {md.get('task')} approved and flagged for manager review.")
                    # update session_state copy
                    for i, (xid, xmd, xvals) in enumerate(st.session_state["client_tasks"]):
                        if xid == tid:
                            st.session_state["client_tasks"][i] = (xid, updated_md, xvals)
                            break
                except Exception as e:
                    st.error("Failed to record client approval.")
                    st.exception(e)

# ---------------------------
# Admin: overview, export, summary generator
# ---------------------------
elif role == "Admin":
    st.header("Admin Overview & Exports")

    df = fetch_all_tasks(top_k=1000)
    if df.empty:
        st.warning("No tasks in index yet.")
    else:
        st.subheader("Global Filters")
        colA, colB, colC = st.columns(3)
        with colA:
            filt_company = st.selectbox("Company (All)", ["All"] + sorted(df["company"].dropna().unique().tolist()))
        with colB:
            filt_employee = st.text_input("Employee filter (substring)")
        with colC:
            filt_month = st.selectbox("Month (All)", ["All"] + sorted(df["month"].dropna().unique().tolist()))

        filtered = df.copy()
        if filt_company and filt_company != "All":
            filtered = filtered[filtered["company"] == filt_company]
        if filt_employee:
            filtered = filtered[filtered[filtered["employee"].str.contains(filt_employee, case=False, na=False)]]
        if filt_month and filt_month != "All":
            filtered = filtered[filtered["month"] == filt_month]

        st.subheader("Summary KPIs")
        total_tasks = len(filtered)
        avg_completion = filtered["completion"].astype(float).mean() if not filtered.empty else 0
        avg_marks = filtered["marks"].astype(float).mean() if not filtered.empty else 0
        col1, col2, col3 = st.columns(3)
        col1.metric("Total tasks", total_tasks)
        col2.metric("Average completion", f"{avg_completion:.1f}%")
        col3.metric("Average marks", f"{avg_marks:.2f}")

        st.subheader("Employee Performance")
        perf = filtered.groupby("employee").agg({"marks":"mean","completion":"mean","task":"count"}).reset_index()
        perf["performance_category"] = perf["marks"].apply(lambda x: compute_perf_category(float(x) if not pd.isna(x) else 0))
        st.dataframe(perf.sort_values("marks", ascending=False))

        st.subheader("Export Data")
        csv_bytes = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv_bytes, file_name="tasks_export.csv", mime="text/csv")
        # excel
        towrite = io.BytesIO()
        with pd.ExcelWriter(towrite, engine="xlsxwriter") as writer:
            filtered.to_excel(writer, sheet_name="tasks", index=False)
        towrite.seek(0)
        st.download_button("Download Excel", towrite.read(), file_name="tasks_export.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        st.subheader("Cluster Dashboard (quick)")
        if not filtered.empty:
            filtered["completion"] = pd.to_numeric(filtered["completion"], errors="coerce")
            filtered["marks"] = pd.to_numeric(filtered["marks"], errors="coerce")
            fig = px.bar(filtered.groupby("employee").completion.mean().reset_index(), x="employee", y="completion", title="Avg Completion by Employee")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("AI-style Summary Generator")
        if st.button("Generate Summary for Current Filter"):
            # simple aggregated summary
            total = len(filtered)
            avg_comp = filtered["completion"].astype(float).mean() if not filtered.empty else 0
            top_emps = filtered.groupby("employee").completion.mean().sort_values(ascending=False).head(3)
            top_list = ", ".join([f"{idx} ({val:.0f}%)" for idx, val in top_emps.items()]) if not top_emps.empty else "N/A"
            summary = (
                f"As of {now_ts()}, there are {total} tasks in the selected filter. "
                f"The average completion is {avg_comp:.1f}%. Top performers by completion: {top_list}. "
                f"Average marks: {avg_marks:.2f}."
            )
            st.info(summary)
