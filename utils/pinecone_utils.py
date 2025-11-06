import pinecone
import numpy as np
import uuid
import streamlit as st

def init_pinecone():
    # Load from Streamlit Secrets
    api_key = st.secrets["pinecone"]["api_key"]
    environment = st.secrets["pinecone"]["environment"]
    index_name = st.secrets["pinecone"]["index_name"]
    dimension = int(st.secrets["pinecone"]["dimension"])

    # Initialize Pinecone
    pinecone.init(api_key=api_key, environment=environment)

    # Create index if it doesn't exist
    if index_name not in [i["name"] for i in pinecone.list_indexes()]:
        pinecone.create_index(name=index_name, dimension=dimension)

    index = pinecone.Index(index_name)
    return index


def rand_vec():
    """Generate random vector of the correct dimension"""
    dim = int(st.secrets["pinecone"]["dimension"])
    return np.random.rand(dim).tolist()


def safe_upsert(index, metadata):
    """Upsert with metadata"""
    if "_id" not in metadata:
        metadata["_id"] = str(uuid.uuid4())
    vector = rand_vec()
    index.upsert([(metadata["_id"], vector, metadata)])
    return metadata
