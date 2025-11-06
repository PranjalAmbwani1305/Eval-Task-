import streamlit as st
from textblob import TextBlob
import pandas as pd
from datetime import datetime

# ---------- App Configuration ----------
st.set_page_config(page_title="AI-Powered Task Management System", layout="wide")
st.title(" AI-Powered Task Management System")
st.subheader("Team Member Portal")

# ---------- Tabs ----------
tab1, tab2 = st.tabs([" My Tasks", " AI Feedback Summarization"])

# ---------- My Tasks ----------
with tab1:
    st.subheader("My Tasks")
    st.info("This section shows your assigned tasks and their progress.")

    # Sample task data (you can link to your database here)
    tasks_data = pd.DataFrame({
        "Task": ["UI Design", "API Integration", "Testing ERP Connector", "Report Generation"],
        "Status": ["In Progress", "Pending", "Completed", "In Review"],
        "Deadline": ["2025-11-10", "2025-11-15", "2025-11-02", "2025-11-08"]
    })

    st.dataframe(tasks_data, use_container_width=True)

# ---------- AI Feedback Summarization ----------
with tab2:
    st.subheader("AI Feedback Summarization")

    company = st.text_input("Company Name (for summary)")
    employee = st.text_input("Your Name (for summary)")
    feedbacks = st.text_area("Paste feedback text below:")

    if st.button("Analyze Feedback"):
        if feedbacks.strip():
            blob = TextBlob(feedbacks)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            # Determine sentiment type
            if polarity > 0.1:
                sentiment_label = "Overall Positive Feedback"
                sentiment_text = (
                    "The feedback reflects a **positive sentiment**, indicating strong performance, "
                    "satisfaction, and overall good engagement."
                )
                sentiment_color = "success"
            elif polarity < -0.1:
                sentiment_label = "Overall Negative Feedback"
                sentiment_text = (
                    "The feedback indicates **negative sentiment**, suggesting areas for improvement — "
                    "for instance, ensuring the application integrates smoothly with the ERP system."
                )
                sentiment_color = "error"
            else:
                sentiment_label = "Neutral Feedback"
                sentiment_text = (
                    "The feedback appears **neutral**, showing a balanced response with both strengths and weaknesses."
                )
                sentiment_color = "info"

            # Summarize key ideas
            keywords = list(blob.noun_phrases)
            key_phrase = ", ".join(keywords[:3]) if keywords else "general feedback"
            full_summary = (
                f"The analysis of feedback from {employee or 'the employee'} at "
                f"{company or 'the organization'} suggests {sentiment_text.lower()} "
                f"Key areas mentioned include {key_phrase}. "
                f"This insight can help managers address specific concerns and improve team alignment."
            )

            # Display results
            st.markdown(f"### {sentiment_label}")
            st.progress((polarity + 1) / 2)
            st.write(full_summary)

            # Smart recommendations
            if polarity < -0.1:
                st.info(" **Recommendation:** Consider reviewing workflow or ERP integration issues.")
            elif polarity > 0.1:
                st.success(" **Recommendation:** Maintain current performance and share best practices.")
            else:
                st.warning("**Recommendation:** Collect additional feedback for clearer sentiment insights.")

            # Save summary (optional local record)
            record = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Employee": employee,
                "Company": company,
                "Feedback": feedbacks,
                "Sentiment": sentiment_label,
                "Summary": full_summary
            }

            if "summaries" not in st.session_state:
                st.session_state["summaries"] = []
            st.session_state["summaries"].append(record)

            st.divider()
            st.markdown("### Saved Summaries")
            st.dataframe(pd.DataFrame(st.session_state["summaries"]))

        else:
            st.warning("Please enter some feedback text before analysis.")
