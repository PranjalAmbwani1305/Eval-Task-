# ============================================================
# 📘 AI Workforce Intelligence Database Setup — Pinecone Only
# ============================================================

# ✅ Install latest dependencies (no Hugging Face)
!pip uninstall -y pinecone-client -q
!pip install -q pinecone pandas numpy tqdm

# ============================================================
# 🔧 Imports & Setup
# ============================================================
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, date
from pinecone import Pinecone, ServerlessSpec
import json
import uuid

# ============================================================
# 🧩 Keys & Constants
# ============================================================
PINECONE_API_KEY = "YOUR_PINECONE_API_KEY"   # 🔑 Replace this
INDEX_NAME = "task"
DIMENSION = 1024  # use your model's embedding dimension

# ============================================================
# 🧠 Helper Functions
# ============================================================
def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def safe_meta(md: dict):
    """Safely convert metadata to JSON-serializable types."""
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

# Simple random embedding generator (replace with real embeddings if needed)
def get_embedding(text: str):
    np.random.seed(abs(hash(text)) % (2**32))
    return np.random.rand(DIMENSION).tolist()

# ============================================================
# 🪶 Connect to Pinecone
# ============================================================
pc = Pinecone(api_key=PINECONE_API_KEY)
existing = [i["name"] for i in pc.list_indexes()]

if INDEX_NAME not in existing:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"✅ Created new index: {INDEX_NAME}")

index = pc.Index(INDEX_NAME)
print(f"📡 Connected to Pinecone index: {INDEX_NAME}")

# ============================================================
# 🗂️ Create Sample Workforce Data (20 Records)
# ============================================================
tasks = pd.DataFrame([
    {"type": "Task", "company": "TechNova", "department": "AI", "employee": "Aarav", "task": "Build NLP Model", "description": "Fine-tune transformer model for intent detection", "deadline": "2025-11-20", "completion": 60, "marks": 3.2, "status": "In Progress", "created": now()},
    {"type": "Task", "company": "TechNova", "department": "Data", "employee": "Meera", "task": "Data Cleaning", "description": "Normalize dataset for training", "deadline": "2025-11-22", "completion": 90, "marks": 4.5, "status": "Completed", "created": now()},
    {"type": "Task", "company": "InnovAI", "department": "Vision", "employee": "Karan", "task": "Object Detection", "description": "Implement YOLOv8 for defect detection", "deadline": "2025-11-25", "completion": 40, "marks": 2.8, "status": "In Progress", "created": now()},
    {"type": "Task", "company": "TechNova", "department": "AI", "employee": "Priya", "task": "Chatbot Design", "description": "Develop LLM-based chatbot", "deadline": "2025-11-18", "completion": 100, "marks": 4.9, "status": "Completed", "created": now()},
    {"type": "Task", "company": "DataMinds", "department": "Analytics", "employee": "Vikram", "task": "Customer Insights", "description": "Predict customer churn using ML", "deadline": "2025-11-30", "completion": 70, "marks": 3.8, "status": "In Progress", "created": now()},
    {"type": "Task", "company": "InnovAI", "department": "Vision", "employee": "Neha", "task": "Image Segmentation", "description": "Use U-Net for medical segmentation", "deadline": "2025-11-24", "completion": 50, "marks": 3.0, "status": "In Progress", "created": now()},
    {"type": "Task", "company": "TechNova", "department": "AI", "employee": "Rohan", "task": "Speech Recognition", "description": "Train model for voice-to-text", "deadline": "2025-11-28", "completion": 85, "marks": 4.2, "status": "Completed", "created": now()},
    {"type": "Task", "company": "DataMinds", "department": "Analytics", "employee": "Sneha", "task": "Market Basket Analysis", "description": "Analyze association rules using Apriori", "deadline": "2025-11-21", "completion": 45, "marks": 2.5, "status": "In Progress", "created": now()},
])

meetings = pd.DataFrame([
    {"type": "Meeting", "company": "TechNova", "meeting_title": "AI Weekly Review", "meeting_date": "2025-11-09", "meeting_time": "10:00 AM", "attendees": json.dumps(["Aarav", "Priya", "Rohan"]), "notes_text": "Reviewed NLP progress and chatbot updates.", "created": now()},
    {"type": "Meeting", "company": "DataMinds", "meeting_title": "Analytics Sprint", "meeting_date": "2025-11-10", "meeting_time": "4:00 PM", "attendees": json.dumps(["Vikram", "Sneha"]), "notes_text": "New metrics for customer segmentation.", "created": now()},
    {"type": "Meeting", "company": "InnovAI", "meeting_title": "Model Evaluation", "meeting_date": "2025-11-12", "meeting_time": "11:00 AM", "attendees": json.dumps(["Karan", "Neha"]), "notes_text": "Improve accuracy and feature refinement.", "created": now()},
])

leaves = pd.DataFrame([
    {"type": "Leave", "employee": "Neha", "company": "InnovAI", "leave_type": "Sick", "from": "2025-11-13", "to": "2025-11-14", "reason": "Cold and fever", "status": "Approved", "submitted": now()},
    {"type": "Leave", "employee": "Vikram", "company": "DataMinds", "leave_type": "Casual", "from": "2025-11-18", "to": "2025-11-19", "reason": "Travel", "status": "Pending", "submitted": now()},
])

feedback = pd.DataFrame([
    {"type": "Feedback", "company": "TechNova", "employee": "Priya", "task": "Chatbot Design", "client_feedback": "Excellent performance!", "rating": 5, "created": now()},
    {"type": "Feedback", "company": "InnovAI", "employee": "Karan", "task": "Object Detection", "client_feedback": "Needs higher precision.", "rating": 3, "created": now()},
])

database = pd.concat([tasks, meetings, leaves, feedback], ignore_index=True)
print(f"📊 Total records created: {len(database)}")

# ============================================================
# 🚀 Upload to Pinecone
# ============================================================
def upsert_to_pinecone(df, index):
    vectors = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        rid = str(uuid.uuid4())
        meta = safe_meta(row.to_dict())
        # Combine text fields for embedding
        text_for_embedding = " ".join([
            str(row.get("task", "")),
            str(row.get("description", "")),
            str(row.get("notes_text", "")),
            str(row.get("reason", "")),
            str(row.get("client_feedback", "")),
        ])
        vec = get_embedding(text_for_embedding)
        vectors.append({"id": rid, "values": vec, "metadata": meta})
        if len(vectors) >= 20 or i == len(df) - 1:
            index.upsert(vectors=vectors)
            vectors = []

upsert_to_pinecone(database, index)
print("✅ All records uploaded successfully!")

# ============================================================
# 🔍 Verify Index
# ============================================================
stats = index.describe_index_stats()
print("\n📦 Pinecone Index Stats:")
print(json.dumps(stats, indent=2))

# ============================================================
# 🔎 Simple Search Example
# ============================================================
query_text = "customer prediction model"
query_vec = get_embedding(query_text)

results = index.query(vector=query_vec, top_k=5, include_metadata=True)
print("\n🔍 Search Results for:", query_text)
for match in results["matches"]:
    print(f"- {match['metadata'].get('task', 'N/A')} ({match['metadata'].get('company', 'N/A')}) → Score: {match['score']:.3f}")
