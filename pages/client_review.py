import streamlit as st
from utils.pinecone_utils import init_pinecone, upsert_data
from utils.ml_models import predict_sentiment
from utils.ui_components import header

def app():
    header("Client Review", "Submit feedback on company projects")
    index = init_pinecone()

    with st.form("client_review"):
        company = st.text_input("Company Name")
        feedback = st.text_area("Your Feedback")
        submit = st.form_submit_button("Submit Review")

        if submit:
            sentiment = predict_sentiment(feedback)
            data = {"type": "review", "company": company, "feedback": feedback, "sentiment": sentiment}
            upsert_data(index, data)
            st.success(f"Feedback saved. Sentiment detected: {sentiment}")
