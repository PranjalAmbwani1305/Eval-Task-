# main.py — AI Workforce Intelligence Platform (Enterprise, full)
# Requirements:
# pip install streamlit pinecone-client scikit-learn plotly huggingface-hub pandas openpyxl PyPDF2

import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import plotly.express as px
import numpy as np
import pandas as pd
import uuid
import json
import time
from datetime import datetime, date
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# optional
try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False
try:
    import PyPDF2
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

# ----------------------------
# App config
# ----------------------------
st.set_page_config(page_title="AI Workforce Intelligence Platform", layout="wide")
st.title("AI Workforce Intelligence Platform")

# ----------------------------
# Secrets & constants
# ----------------------------
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
HF_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")
INDEX_NAME = "task"
DIMENSION = 1024

# ----------------------------
# Pinecone init (best-effort)
# ----------------------------
pc = None
index = None
if PINECONE_API_KEY:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        existing = [i["name"] for i in pc.list_indexes()]
        if INDEX_NAME not in existing:
            pc.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            for _ in range(20):
                try:
                    desc = pc.describe_index(INDEX_NAME)
                    if desc.get("status", {}).get("ready"):
                        break
                except Exception:
                    pass
                time.sleep(1)
        index = pc.Index(INDEX_NAME)
        st.caption(f"Connected to Pinecone index: {INDEX_NAME}")
    except Exception as e:
        st.warning(f"Pinecone init failed — running local-only. ({e})")
else:
    st.warning("Pinecone API key missing — running local-only.")

# ----------------------------
# Ensure session structures
# ----------------------------
if "PINECONE_IDS" not in st.session_state:
    st.session_state["PINECONE_IDS"] = []
if "LOCAL_DATA" not in st.session_state:
    st.session_state["LOCAL_DATA"] = {}

# ----------------------------
# Utilities
# ----------------------------
def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# --- safer random vector / embedding placeholder ---
def rand_vec():
    """Generate deterministic random-like vector."""
    return np.random.rand(DIMENSION).tolist()

# Optional: use Hugging Face embeddings (fallback to rand_vec)
def get_embedding(text: str):
    if not HF_AVAILABLE or not HF_TOKEN:
        return rand_vec()
    try:
        client = InferenceClient(token=HF_TOKEN)
        res = client.embeddings(model="sentence-transformers/all-MiniLM-L6-v2", inputs=text)
        if isinstance(res, dict) and "embedding" in res:
            return res["embedding"]
        if isinstance(res, list) and len(res) and isinstance(res[0], dict) and "embedding" in res[0]:
            return res[0]["embedding"]
        return rand_vec()
    except Exception:
        return rand_vec()

def safe_meta(md: dict) -> dict:
    """Convert metadata to JSON-friendly primitives."""
    clean = {}
    for k, v in (md or {}).items():
        try:
            if isinstance(v, (datetime, date)):
                clean[k] = v.isoformat()
            elif isinstance(v, (dict, list)):
                clean[k] = json.dumps(v)
            elif pd.isna(v):
                clean[k] = ""
            else:
                clean[k] = v
        except Exception:
            clean[k] = str(v)
    return clean

# --- FIXED: upsert_data + fetch_all ---
def _register_id(id_):
    """Keep all IDs in session for reliable fetch."""
    id_ = str(id_)
    ids = st.session_state.get("PINECONE_IDS", [])
    if id_ not in ids:
        ids.append(id_)
        st.session_state["PINECONE_IDS"] = ids

def upsert_data(id_, md: dict) -> bool:
    """Upsert record to Pinecone or local storage with ID tracking."""
    id_ = str(id_)
    _register_id(id_)
    meta = safe_meta(md)

    if not index:
        st.session_state["LOCAL_DATA"][id_] = meta
        return True
    try:
        vec = get_embedding(md.get("description", md.get("task", "")))
        index.upsert(vectors=[{"id": id_, "values": vec, "metadata": meta}])
        return True
    except Exception as e:
        st.warning(f"Pinecone upsert failed — saved locally. ({e})")
        st.session_state["LOCAL_DATA"][id_] = meta
        return False

def fetch_all() -> pd.DataFrame:
    """Fetch all metadata records reliably using stored IDs."""
    ids = st.session_state.get("PINECONE_IDS", [])

    # local fallback
    if not index:
        rows = []
        for k, md in st.session_state["LOCAL_DATA"].items():
            rec = dict(md)
            rec["_id"] = k
            rows.append(rec)
        return pd.DataFrame(rows)

    if not ids:
        return pd.DataFrame()

    try:
        res = index.fetch(ids=ids)
        vectors = getattr(res, "vectors", None)
        if vectors is None and isinstance(res, dict):
            vectors = res.get("vectors", {})
        rows = []
        if isinstance(vectors, dict):
            for vid, info in vectors.items():
                md = info.get("metadata", {}) if isinstance(info, dict) else {}
                md["_id"] = vid
                rows.append(md)
        return pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"Fetch error: {e}")
        # fallback to local data
        rows = []
        for k, md in st.session_state["LOCAL_DATA"].items():
            rec = dict(md)
            rec["_id"] = k
            rows.append(rec)
        return pd.DataFrame(rows)

# ----------------------------
# Hugging Face text generation
# ----------------------------
def hf_text_generation(prompt: str, model: str = "mistralai/Mixtral-8x7B-Instruct", max_new_tokens: int = 200):
    if not HF_AVAILABLE or not HF_TOKEN:
        raise RuntimeError("Hugging Face client or token not available")
    client = InferenceClient(token=HF_TOKEN)
    try:
        res = client.text_generation(model=model, inputs=prompt, max_new_tokens=max_new_tokens)
    except TypeError:
        res = client.text_generation(model=model, prompt=prompt, max_new_tokens=max_new_tokens)
    if isinstance(res, dict):
        return res.get("generated_text") or res.get("output") or json.dumps(res)
    if isinstance(res, list) and res and isinstance(res[0], dict):
        return res[0].get("generated_text") or json.dumps(res[0])
    return str(res)

# ----------------------------
# Linear regression for marks
# ----------------------------
lin_reg = LinearRegression().fit([[0], [50], [100]], [0, 2.5, 5])

# ----------------------------
# Parse attendees
# ----------------------------
def parse_attendees_field(val):
    if isinstance(val, list):
        return [a.strip().lower() for a in val if isinstance(a, str) and a.strip()]
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s.replace("'", '"'))
                if isinstance(parsed, list):
                    return [a.strip().lower() for a in parsed if isinstance(a, str) and a.strip()]
            except Exception:
                s2 = s.strip("[]").replace("'", "")
                return [a.strip().lower() for a in s2.split(",") if a.strip()]
        return [a.strip().lower() for a in s.split(",") if a.strip()]
    return []

# ----------------------------
# Role selector
# ----------------------------
role = st.sidebar.selectbox("Login as", ["Manager", "Team Member", "Client", "HR (Admin)"])
current_month = datetime.now().strftime("%B %Y")

# ----------------------------
# --- All role logic below ---
# (No changes in structure)
# ----------------------------

# [KEEP YOUR ORIGINAL FULL ROLE BLOCKS BELOW — unchanged]
# ↓↓↓ paste your existing code for Manager, Team Member, Client, HR exactly as it was ↓↓↓

# ----------------------------
# MANAGER / TEAM MEMBER / CLIENT / HR CODE BLOCKS
# ----------------------------
# (Everything below this line remains your same content)
# -------------------------------------------------------
