import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from textblob import TextBlob
import numpy as np
from datetime import datetime
import pandas as pd

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Team Member Portal", layout="wide")
st.title("👷 Team Member Portal")

# -----------------------------
# INITIALIZATION
# -----------------------------
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "task"
DIMENSION = 1024

# Create index if not exists
if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(INDEX_NAME)

def rand_vec():
    return np.random.rand(DIMENSION).tolist()

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# -----------------------------
# FETCH FEEDBACK FROM PINECONE
# -----------------------------
def fetch_feedback(company, employee):
    """
    Fetch reviewed tasks for a given employee and company from Pinecone.
    """
    try:
        res = index.query(
            vector=rand_vec(),
            top_k=500,
            include_metadata=True,
            filter={
                "company": {"$eq": company},
                "employee": {"$eq": employee},
                "reviewed": {"$eq": True}
            }
        )

        records = []
        for m in res.matches or []:
            md = m.metadata
            records.append({
                "task": md.get("task", "Unnamed Task"),
                "manager_comments": md.get("comments", ""),
                "client_comments": md.get("client_comments", "")
            })
        return pd.DataFrame(records)
    except Exception as e:
        st.error(f"⚠️ Error fetching feedback: {e}")
        return pd.DataFrame()

# -----------------------------
# PORTAL SECTIONS
# -----------------------------
tabs = st.tabs(["My Tasks", "AI Feedback Summarization"])

# ====================================================
# TAB 1: My Tasks (placeholder)
# ====================================================
with tabs[0]:
    st.subheader("📋 My Tasks")
    st.info("This section shows your assigned tasks and progress (connects to manager assignments).")

# ====================================================
# TAB 2: AI Feedback Summarization (Auto Mode)
# ====================================================
with tabs[1]:
    st.subheader("🧠 AI Feedback Summarization (Automatic)")

    company = st.text_input("🏢 Company Name")
    employee = st.text_input("👤 Your Name")

    if st.button("🔍 Load & Analyze Feedback"):
        if not company or not employee:
            st.warning("⚠️ Please enter both company and employee name.")
        else:
            df = fetch_feedback(company, employee)
            if df.empty:
                st.warning("No reviewed feedback found for this employee yet.")
            else:
                st.success(f"Loaded {len(df)} reviewed tasks with feedback.")

                feedback_combined = ""
                for _, row in df.iterrows():
                    task = row["task"]
                    manager = row["manager_comments"]
                    client = row["client_comments"]
                    st.markdown(f"### 🧾 {task}")
                    if manager:
                        st.write(f"**Manager Feedback:** {manager}")
                    if client:
                        st.write(f"**Client Feedback:** {client}")
                    feedback_combined += f"\nTask: {task}\nManager: {manager}\nClient: {client}\n"

                # --- Analyze combined feedback ---
                blob = TextBlob(feedback_combined)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity

                if polarity > 0.2:
                    sentiment = "🌟 Overall Positive Feedback"
                    suggestion = "Keep up the excellent work and consistency."
                elif polarity < -0.2:
                    sentiment = "⚠️ Overall Negative Feedback"
                    suggestion = "Focus on improving weak areas and clarify expectations."
                else:
                    sentiment = "💬 Neutral Feedback"
                    suggestion = "Mixed reviews — continue improving communication and consistency."

                st.markdown("---")
                st.markdown(f"### 📊 Sentiment Summary")
                st.markdown(f"**Sentiment:** {sentiment}")
                st.markdown(f"**Polarity:** `{polarity:.2f}` | **Subjectivity:** `{subjectivity:.2f}`")

                st.progress((polarity + 1) / 2)

                st.markdown(f"### 💡 AI Suggestion")
                st.info(suggestion)

                st.markdown("### 🧠 Key Feedback Themes")
                st.write(", ".join(blob.noun_phrases))
