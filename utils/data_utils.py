import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    return pd.DataFrame(columns=["employee", "company", "type", "status", "completion", "marks"])

def save_to_csv(df, filename="data.csv"):
    df.to_csv(filename, index=False)
