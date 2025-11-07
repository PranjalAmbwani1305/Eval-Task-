import streamlit as st
from textblob import TextBlob
import uuid
from datetime import datetime
from utils.pinecone_utils import safe_upsert, rand_vec
from utils.ml_models import lin_reg, log_reg, rf
from utils.notifications import send_notification

def team_member_portal(index):
    st.header("👷 Team Member Portal")
    tab1, tab2, tab3 = st.tabs(["My Tasks", "AI Feedback", "Submit Leave"])

    # --- My Tasks ---
    with tab1:
        st.subheader("📋 My Tasks")
        company = st.text_input("Company Name", key="tasks_company")
        employee = st.text_input("Your Name", key="tasks_name")

        if st.button("Load Tasks", key="load_tasks_btn"):
            res = index.query(vector=rand_vec(), top_k=500, include_metadata=True)
            st.session_state["tasks"] = [(m.id, m.metadata) for m in res.matches or []]
            st.success(f"Loaded {len(st.session_state['tasks'])} tasks.")

        for tid, md in st.session_state.get("tasks", []):
            st.subheader(md.get("task"))
            curr = float(md.get("completion", 0))
            new = st.slider(f"Completion {md.get('task')}", 0, 100, int(curr), key=f"slider_{tid}")
            if st.button(f"Submit {md.get('task')}", key=f"submit_{tid}"):
                marks = float(lin_reg.predict([[new]])[0])
                track = "On Track" if log_reg.predict([[new]])[0] == 1 else "Delayed"
                miss = rf.predict([[new, 0]])[0]
                md2 = {
                    **md,
                    "completion": new,
                    "marks": marks,
                    "status": track,
                    "deadline_risk": "High" if miss else "Low",
                    "submitted_on": datetime.now().isoformat()
                }
                safe_upsert(index, md2)
                send_notification(md.get("email"), md.get("phone"),
                    subject=f"Task Update: {md.get('task')}",
                    msg=f"Task '{md.get('task')}' updated to {new}% ({track})")
                st.success(f"✅ Updated {md.get('task')} ({track})")

    # --- AI Feedback ---
    with tab2:
        st.subheader("🧠 AI Feedback Summarization")
        feedback_text = st.text_area("💬 Feedback text", key="feedback_text")

        if st.button("🔍 Analyze Feedback", key="analyze_feedback_btn"):
            if feedback_text.strip():
                blob = TextBlob(feedback_text)
                polarity = blob.sentiment.polarity
                if polarity > 0.1:
                    sentiment = "✅ Positive"
                elif polarity < -0.1:
                    sentiment = "⚠️ Negative"
                else:
                    sentiment = "💬 Neutral"
                st.write(f"**Sentiment:** {sentiment}")
                st.progress((polarity + 1) / 2)
            else:
                st.warning("Please enter feedback text.")

    # --- Submit Leave ---
    with tab3:
        st.subheader("🗓️ Submit Leave Request")
        company = st.text_input("Company Name", key="leave_company")
        employee = st.text_input("Your Name", key="leave_name")
        reason = st.text_area("Reason for Leave", key="leave_reason")
        start_date = st.date_input("Start Date", key="leave_start")
        end_date = st.date_input("End Date", key="leave_end")

        if st.button("Submit Leave", key="submit_leave_btn"):
            leave_data = {
                "_id": str(uuid.uuid4()),
                "company": company,
                "employee": employee,
                "reason": reason,
                "from": start_date.isoformat(),
                "to": end_date.isoformat(),
                "status": "Pending",
                "type": "leave",
                "applied_on": datetime.now().isoformat()
            }
            safe_upsert(index, leave_data)
            st.success(f"✅ Leave request submitted from {start_date} to {end_date}")
