import streamlit as st
from utils.pinecone_utils import init_pinecone, upsert_data, query_data
from utils.ml_models import predict_risk, predict_status
from utils.ui_components import header, divider

def app():
    header("Manager Dashboard", "Assign tasks and monitor progress")
    index = init_pinecone()

    with st.form("assign_task"):
        name = st.text_input("Employee Name")
        company = st.text_input("Company")
        progress = st.slider("Completion %", 0, 100)
        priority = st.selectbox("Priority", [0, 1, 2])
        submit = st.form_submit_button("Assign Task")

        if submit:
            status = predict_status(progress)
            risk = predict_risk(progress, priority)
            task = {"type": "task", "employee": name, "company": company, "completion": progress, "status": status, "risk": risk}
            upsert_data(index, task)
            st.success(f"Task assigned ({status}, Risk: {risk})")

    divider()
    st.subheader("Recent Tasks")
    res = query_data(index, {"type": {"$eq": "task"}})
    for r in res.matches:
        st.write(r.metadata)
