import streamlit as st

def header(title, subtitle=None):
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)

def divider():
    st.markdown("---")

def card(title, value, color="blue"):
    st.markdown(f"<div style='padding:10px;background-color:{color};border-radius:10px;color:white;'>{title}: {value}</div>", unsafe_allow_html=True)
