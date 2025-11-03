import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import uuid
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_extraction.text import CountVectorizerimport streamlit as st
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import uuid
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

# ------------------------------------------------------
# 🔹 Step 1: Initialize Pinecone (Vector Database)
# ------------------------------------------------------
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])  # API key stored in Streamlit secrets

index_name = "task"
dimension = 1024  # size of vector embeddings (here using random demo vectors)

# Create Pinecone index if not already exists
if index_name not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",  # similarity metric
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to index
index = pc.Index(index_name)

# ------------------------------------------------------
# 🔹 Step 2: Machine Learning Models
# ------------------------------------------------------

# Linear Regression → predicts marks based on completion %
lin_reg = LinearRegression()
lin_reg.fit([[0], [100]], [0, 5])  # 0% → 0 marks, 100% → 5 marks

# Logistic Regression → predicts task status (On Track / Delayed)
log_reg = LogisticRegression()
log_reg.fit([[0], [50], [100]], [0, 0, 1])  # <50% delayed, >=100% on track

# SVM for sentiment analysis (Positive / Negative comments)
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(["good work", "excellent", "needs improvement", "bad performance"])
y_train = [1, 1, 0, 0]  # 1 = Positive, 0 = Negative
svm_clf = SVC()
svm_clf.fit(X_train, y_train)

# ------------------------------------------------------
# 🔹 Step 3: Helper Functions
# ------------------------------------------------------

def random_vector(dim=dimension):
    """Generate random vector (since we are not using embeddings here)."""
    return np.random.rand(dim).tolist()

def safe_metadata(md: dict):
    """Convert metadata values into safe JSON format (avoid numpy types)."""
    clean = {}
    for k, v in md.items():
        if isinstance(v, (np.generic,)):
            v = v.item()
        clean[k] = v
    return clean

# ------------------------------------------------------
# 🔹 Step 4: Streamlit App
# ------------------------------------------------------
st.title("📊 AI-Powered Task Completion & Review")

# User role selection (Team Member, Manager, Client)
role = st.sidebar.selectbox("Login as", ["Team Member", "Manager", "Client"])

# ------------------------------------------------------
# 🔹 TEAM MEMBER ROLE
# ------------------------------------------------------
if role == "Team Member":
    st.header("👩‍💻 Team Member Section")

    # Input fields
    company = st.text_input("🏢 Company Name")
    employee = st.text_input("👤 Your Name")
    task = st.text_input("📝 Task Title")
    completion = st.slider("✅ Completion %", 0, 100, 0)

    # Submit task
    if st.button("📩 Submit Task"):
        if company and employee and task:
            marks = lin_reg.predict([[completion]])[0]  # AI predicted marks
            status = log_reg.predict([[completion]])[0]  # AI status
            status_text = "On Track" if status == 1 else "Delayed"

            task_id = str(uuid.uuid4())  # unique ID for each task

            # Store task in Pinecone
            index.upsert(
                vectors=[{
                    "id": task_id,
                    "values": random_vector(),
                    "metadata": safe_metadata({
                        "company": company,
                        "employee": employee,
                        "task": task,
                        "completion": float(completion),
                        "marks": float(marks),
                        "status": status_text,
                        "reviewed": False  # Not yet reviewed by Manager
                    })
                }]
            )
            st.success(f"✅ Task '{task}' submitted by {employee}")
        else:
            st.error("❌ Fill all fields before submitting")

# ------------------------------------------------------
# 🔹 CLIENT ROLE
# ------------------------------------------------------
elif role == "Client":
    st.header("👨‍💼 Client Section")
    company = st.text_input("🏢 Company Name")

    if st.button("🔍 View Approved Tasks"):
        if company:
            # Fetch only reviewed (approved) tasks
            res = index.query(
                vector=random_vector(),
                top_k=100,
                include_metadata=True,
                filter={"company": {"$eq": company}, "reviewed": {"$eq": True}}
            )

            if res.matches:
                st.subheader(f"📌 Approved Tasks for {company}")
                for match in res.matches:
                    md = match.metadata or {}
                    st.write(
                        f"👤 {md.get('employee','?')} | **{md.get('task','?')}** → {md.get('completion',0)}% "
                        f"(Marks: {md.get('marks',0):.2f}) | Status: {md.get('status','?')}"
                    )
                    st.write(f"📝 Manager Sentiment: {md.get('sentiment','N/A')}")
            else:
                st.warning("⚠️ No approved tasks found.")
        else:
            st.error("❌ Enter company name")

# ------------------------------------------------------
# 🔹 MANAGER ROLE
# ------------------------------------------------------
elif role == "Manager":
    st.header("🧑‍💼 Manager Review Section")

    # First: Fetch all companies in Pinecone
    all_res = index.query(vector=random_vector(), top_k=100, include_metadata=True)
    companies = list(set([m.metadata["company"] for m in all_res.matches])) if all_res.matches else []

    # Dropdown for companies
    if companies:
        company = st.selectbox("🏢 Select Company", companies)
    else:
        st.warning("⚠️ No companies found.")
        company = None

    # Load only pending tasks (not yet reviewed)
    if company and st.button("📂 Load Pending Tasks"):
        res = index.query(
            vector=random_vector(),
            top_k=100,
            include_metadata=True,
            include_values=True,
            filter={"company": {"$eq": company}, "reviewed": {"$eq": False}}
        )

        # Case 1: All tasks are already reviewed
        if not res.matches:
            st.success(f"✅ All tasks for {company} have already been reviewed!")
        else:
            st.session_state["pending"] = res.matches

    # Show pending tasks if available
    if "pending" in st.session_state and st.session_state["pending"]:
        st.subheader(f"📌 Pending Tasks for {company}")

        for match in st.session_state["pending"]:
            md = match.metadata or {}
            emp = md.get("employee", "?")
            task = md.get("task", "?")
            emp_completion = float(md.get("completion", 0))

            st.write(f"👤 {emp} | Task: **{task}**")
            st.write(f"Employee Reported: {emp_completion}%")

            # Manager adjusts completion percentage
            manager_completion = st.slider(
                f"✅ Adjust Completion ({emp} - {task})",
                0, 100, int(emp_completion),
                key=f"adj_{match.id}"
            )

            # AI predicts marks and status again
            predicted_marks = float(lin_reg.predict([[manager_completion]])[0])
            status = log_reg.predict([[manager_completion]])[0]
            status_text = "On Track" if status == 1 else "Delayed"

            # Manager comments with sentiment analysis
            comments = st.text_area(
                f"📝 Manager Comments for {emp} - {task}",
                key=f"c_{match.id}"
            )
            sentiment_text = "N/A"
            if comments:
                try:
                    X_new = vectorizer.transform([comments])
                    sentiment = svm_clf.predict(X_new)[0]
                    sentiment_text = "Positive" if sentiment == 1 else "Negative"
                except Exception:
                    sentiment_text = "N/A"

            # Show AI output
            st.write(f"🤖 AI Marks: {predicted_marks:.2f}/5 | Status: {status_text}")
            st.write(f"🤖 Sentiment: {sentiment_text}")

            # Save review to Pinecone
            if st.button(f"💾 Save Review for {emp} - {task}", key=f"s_{match.id}"):
                index.upsert(vectors=[{
                    "id": match.id,
                    "values": match.values if hasattr(match, "values") else random_vector(),
                    "metadata": safe_metadata({
                        **md,
                        "completion": float(manager_completion),
                        "marks": predicted_marks,
                        "status": status_text,
                        "reviewed": True,  # Mark as reviewed
                        "comments": comments,
                        "sentiment": sentiment_text
                    })
                }])
                st.success(f"✅ Review saved for {emp} - {task}")
from sklearn.svm import SVC

# ------------------------------------------------------
# 🔹 Step 1: Initialize Pinecone (Vector Database)
# ------------------------------------------------------
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])  # API key stored in Streamlit secrets

index_name = "task"
dimension = 1024  # size of vector embeddings (here using random demo vectors)

# Create Pinecone index if not already exists
if index_name not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",  # similarity metric
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to index
index = pc.Index(index_name)

# ------------------------------------------------------
# 🔹 Step 2: Machine Learning Models
# ------------------------------------------------------

# Linear Regression → predicts marks based on completion %
lin_reg = LinearRegression()
lin_reg.fit([[0], [100]], [0, 5])  # 0% → 0 marks, 100% → 5 marks

# Logistic Regression → predicts task status (On Track / Delayed)
log_reg = LogisticRegression()
log_reg.fit([[0], [50], [100]], [0, 0, 1])  # <50% delayed, >=100% on track

# SVM for sentiment analysis (Positive / Negative comments)
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(["good work", "excellent", "needs improvement", "bad performance"])
y_train = [1, 1, 0, 0]  # 1 = Positive, 0 = Negative
svm_clf = SVC()
svm_clf.fit(X_train, y_train)

# ------------------------------------------------------
# 🔹 Step 3: Helper Functions
# ------------------------------------------------------

def random_vector(dim=dimension):
    """Generate random vector (since we are not using embeddings here)."""
    return np.random.rand(dim).tolist()

def safe_metadata(md: dict):
    """Convert metadata values into safe JSON format (avoid numpy types)."""
    clean = {}
    for k, v in md.items():
        if isinstance(v, (np.generic,)):
            v = v.item()
        clean[k] = v
    return clean

# ------------------------------------------------------
# 🔹 Step 4: Streamlit App
# ------------------------------------------------------
st.title("📊 AI-Powered Task Completion & Review")

# User role selection (Team Member, Manager, Client)
role = st.sidebar.selectbox("Login as", ["Team Member", "Manager", "Client"])

# ------------------------------------------------------
# 🔹 TEAM MEMBER ROLE
# ------------------------------------------------------
if role == "Team Member":
    st.header("👩‍💻 Team Member Section")

    # Input fields
    company = st.text_input("🏢 Company Name")
    employee = st.text_input("👤 Your Name")
    task = st.text_input("📝 Task Title")
    completion = st.slider("✅ Completion %", 0, 100, 0)

    # Submit task
    if st.button("📩 Submit Task"):
        if company and employee and task:
            marks = lin_reg.predict([[completion]])[0]  # AI predicted marks
            status = log_reg.predict([[completion]])[0]  # AI status
            status_text = "On Track" if status == 1 else "Delayed"

            task_id = str(uuid.uuid4())  # unique ID for each task

            # Store task in Pinecone
            index.upsert(
                vectors=[{
                    "id": task_id,
                    "values": random_vector(),
                    "metadata": safe_metadata({
                        "company": company,
                        "employee": employee,
                        "task": task,
                        "completion": float(completion),
                        "marks": float(marks),
                        "status": status_text,
                        "reviewed": False  # Not yet reviewed by Manager
                    })
                }]
            )
            st.success(f"✅ Task '{task}' submitted by {employee}")
        else:
            st.error("❌ Fill all fields before submitting")

# ------------------------------------------------------
# 🔹 CLIENT ROLE
# ------------------------------------------------------
elif role == "Client":
    st.header("👨‍💼 Client Section")
    company = st.text_input("🏢 Company Name")

    if st.button("🔍 View Approved Tasks"):
        if company:
            # Fetch only reviewed (approved) tasks
            res = index.query(
                vector=random_vector(),
                top_k=100,
                include_metadata=True,
                filter={"company": {"$eq": company}, "reviewed": {"$eq": True}}
            )

            if res.matches:
                st.subheader(f"📌 Approved Tasks for {company}")
                for match in res.matches:
                    md = match.metadata or {}
                    st.write(
                        f"👤 {md.get('employee','?')} | **{md.get('task','?')}** → {md.get('completion',0)}% "
                        f"(Marks: {md.get('marks',0):.2f}) | Status: {md.get('status','?')}"
                    )
                    st.write(f"📝 Manager Sentiment: {md.get('sentiment','N/A')}")
            else:
                st.warning("⚠️ No approved tasks found.")
        else:
            st.error("❌ Enter company name")

# ------------------------------------------------------
# 🔹 MANAGER ROLE
# ------------------------------------------------------
elif role == "Manager":
    st.header("🧑‍💼 Manager Review Section")

    # First: Fetch all companies in Pinecone
    all_res = index.query(vector=random_vector(), top_k=100, include_metadata=True)
    companies = list(set([m.metadata["company"] for m in all_res.matches])) if all_res.matches else []

    # Dropdown for companies
    if companies:
        company = st.selectbox("🏢 Select Company", companies)
    else:
        st.warning("⚠️ No companies found.")
        company = None

    # Load only pending tasks (not yet reviewed)
    if company and st.button("📂 Load Pending Tasks"):
        res = index.query(
            vector=random_vector(),
            top_k=100,
            include_metadata=True,
            include_values=True,
            filter={"company": {"$eq": company}, "reviewed": {"$eq": False}}
        )

        # Case 1: All tasks are already reviewed
        if not res.matches:
            st.success(f"✅ All tasks for {company} have already been reviewed!")
        else:
            st.session_state["pending"] = res.matches

    # Show pending tasks if available
    if "pending" in st.session_state and st.session_state["pending"]:
        st.subheader(f"📌 Pending Tasks for {company}")

        for match in st.session_state["pending"]:
            md = match.metadata or {}
            emp = md.get("employee", "?")
            task = md.get("task", "?")
            emp_completion = float(md.get("completion", 0))

            st.write(f"👤 {emp} | Task: **{task}**")
            st.write(f"Employee Reported: {emp_completion}%")

            # Manager adjusts completion percentage
            manager_completion = st.slider(
                f"✅ Adjust Completion ({emp} - {task})",
                0, 100, int(emp_completion),
                key=f"adj_{match.id}"
            )

            # AI predicts marks and status again
            predicted_marks = float(lin_reg.predict([[manager_completion]])[0])
            status = log_reg.predict([[manager_completion]])[0]
            status_text = "On Track" if status == 1 else "Delayed"

            # Manager comments with sentiment analysis
            comments = st.text_area(
                f"📝 Manager Comments for {emp} - {task}",
                key=f"c_{match.id}"
            )
            sentiment_text = "N/A"
            if comments:
                try:
                    X_new = vectorizer.transform([comments])
                    sentiment = svm_clf.predict(X_new)[0]
                    sentiment_text = "Positive" if sentiment == 1 else "Negative"
                except Exception:
                    sentiment_text = "N/A"

            # Show AI output
            st.write(f"🤖 AI Marks: {predicted_marks:.2f}/5 | Status: {status_text}")
            st.write(f"🤖 Sentiment: {sentiment_text}")

            # Save review to Pinecone
            if st.button(f"💾 Save Review for {emp} - {task}", key=f"s_{match.id}"):
                index.upsert(vectors=[{
                    "id": match.id,
                    "values": match.values if hasattr(match, "values") else random_vector(),
                    "metadata": safe_metadata({
                        **md,
                        "completion": float(manager_completion),
                        "marks": predicted_marks,
                        "status": status_text,
                        "reviewed": True,  # Mark as reviewed
                        "comments": comments,
                        "sentiment": sentiment_text
                    })
                }])
                st.success(f"✅ Review saved for {emp} - {task}")
