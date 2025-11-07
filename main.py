import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

# Import Pinecone and initialize it securely
from pinecone import init, Index, PodSpec, ServerlessSpec, list_indexes, describe_index

st.set_page_config(layout="wide")

st.title("🧠 AI Task Management System: Under the Hood")

st.write("This document explains the machine learning models powering your AI Task Management System.")

# --- Pinecone Initialization (Updated Addition) ---
st.subheader("Pinecone Setup")

# Define our specific index details
PINECONE_TASK_INDEX_NAME = "my-task-index"
PINECONE_INDEX_DIMENSION = 1024
# You might want to specify a pod_type (e.g., 'p1.x1') or cloud provider and region for ServerlessSpec
# For Serverless, you'd typically define cloud='aws' and region='us-west-2'
# For Pods, you'd define pod_type='p1.x1'
# Let's use ServerlessSpec as it's often simpler for development, but you can change this.
# Make sure the region matches your Pinecone environment's supported regions.

try:
    # Access Pinecone API key and environment from Streamlit secrets
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]
    pinecone_environment = st.secrets["PINECONE_ENVIRONMENT"]

    # Initialize Pinecone
    init(api_key=pinecone_api_key, environment=pinecone_environment)
    st.success("Pinecone initialized successfully!")

    # Check if the index exists
    if PINECONE_TASK_INDEX_NAME not in list_indexes():
        st.info(f"Pinecone index '{PINECONE_TASK_INDEX_NAME}' does not exist. Creating it now...")
        try:
            # Use ServerlessSpec for simplicity, adjust cloud and region as needed for your Pinecone setup
            init(api_key=pinecone_api_key, environment=pinecone_environment)
            cloud_provider = "aws" # Common default, change if your project uses GCP/Azure
            cloud_region = "us-west-2" # Example region, choose one supported by your Pinecone environment

            create_args = {
                "name": PINECONE_TASK_INDEX_NAME,
                "dimension": PINECONE_INDEX_DIMENSION,
                "metric": "cosine" # Common metric for embeddings, change if needed
            }

            # Choose between PodSpec and ServerlessSpec based on your Pinecone setup
            if pinecone_environment.startswith("gcp"): # Example for GCP
                create_args["spec"] = PodSpec(environment=pinecone_environment, pod_type="s1.x1") # Adjust pod_type
            elif pinecone_environment.startswith("aws"): # Example for AWS
                 # Attempt to use ServerlessSpec if supported by the environment and region
                try:
                    create_args["spec"] = ServerlessSpec(cloud=cloud_provider, region=cloud_region)
                except Exception as serverless_e:
                    st.warning(f"ServerlessSpec failed ({serverless_e}), falling back to PodSpec. Ensure your environment supports Serverless in region {cloud_region}.")
                    create_args["spec"] = PodSpec(environment=pinecone_environment, pod_type="s1.x1") # Adjust pod_type
            else: # Default fallback
                create_args["spec"] = PodSpec(environment=pinecone_environment, pod_type="s1.x1") # Adjust pod_type


            # Finally, create the index
            init(api_key=pinecone_api_key, environment=pinecone_environment) # Re-init just in case
            Index.create(**create_args)
            st.success(f"Pinecone index '{PINECONE_TASK_INDEX_NAME}' created successfully!")
        except Exception as e:
            st.error(f"Failed to create Pinecone index '{PINECONE_TASK_INDEX_NAME}': {e}")
            st.stop() # Stop if index creation fails
    else:
        st.info(f"Pinecone index '{PINECONE_TASK_INDEX_NAME}' already exists.")
        # You can also describe it to confirm dimensions, etc.
        index_info = describe_index(PINECONE_TASK_INDEX_NAME)
        st.write(f"Index Description: {index_info.status.state} (Dimension: {index_info.dimension})")
        if index_info.dimension != PINECONE_INDEX_DIMENSION:
            st.warning(f"Existing index dimension ({index_info.dimension}) does not match expected ({PINECONE_INDEX_DIMENSION}). This might cause issues.")


    # Now connect to the index
    pinecone_index = Index(PINECONE_TASK_INDEX_NAME)
    st.write(f"Connected to Pinecone index: `{PINECONE_TASK_INDEX_NAME}`")

except KeyError as e:
    st.error(f"Pinecone secret not found: {e}. Please add 'PINECONE_API_KEY' and 'PINECONE_ENVIRONMENT' to your .streamlit/secrets.toml file.")
    st.stop() # Stop the app if Pinecone secrets are missing
except Exception as e:
    st.error(f"An error occurred during Pinecone initialization or index setup: {e}")
    st.stop()
# --- End Pinecone Initialization ---


st.header("1. ML Models Overview")
st.write("Your system currently uses five ML-related components:")

models_overview = pd.DataFrame({
    "Model Type": ["Linear Regression", "Logistic Regression", "Random Forest Classifier", "Support Vector Machine (SVM)", "K-Means Clustering"],
    "Library": ["sklearn.linear_model.LinearRegression", "sklearn.linear_model.LogisticRegression", "sklearn.ensemble.RandomForestClassifier", "sklearn.svm.SVC", "sklearn.cluster.KMeans"],
    "Purpose": [
        "Predicts *marks* (performance score) based on task completion %.",
        "Predicts whether a task is *On Track* or *Delayed*.",
        "Estimates *deadline risk* (High/Low).",
        "Analyzes manager comments to classify sentiment (Positive/Negative).",
        "Groups employees into *performance clusters* for visual analytics (360° overview)."
    ]
})
st.table(models_overview)

st.header("2. Model Training Logic")
st.write("Let’s understand how each one is trained and what data it learns from:")

st.subheader("a. Linear Regression → Predicts Marks")
st.code("""
lin_reg = LinearRegression()
lin_reg.fit([[0], [50], [100]], [0, 2.5, 5])
""")
st.write("Training data:")
st.table(pd.DataFrame({"Completion (%)": [0, 50, 100], "Marks": [0, 2.5, 5]}))
st.write("The model learns a **linear mapping** between completion % and marks. So if a task is 80% done, it predicts around **4 marks**.")
st.write("🧩 *Usage:* When a manager adjusts completion, the system auto-predicts the corresponding marks.")

# Initialize and train Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(np.array([[0], [50], [100]]), np.array([0, 2.5, 5]))


st.subheader("b. Logistic Regression → Predicts On Track / Delayed")
st.code("""
log_reg = LogisticRegression()
log_reg.fit([[0], [40], [80], [100]], [0, 0, 1, 1])
""")
st.write("Training data:")
st.table(pd.DataFrame({"Completion": ["0%", "40%", "80%", "100%"], "Status": ["Delayed", "Delayed", "On Track", "On Track"]}))
st.write("Logistic regression models a **sigmoid probability curve**, outputting 1 for “On Track” when completion is sufficiently high.")
st.write("🧩 *Usage:* When a team member updates progress, the system instantly classifies it as **On Track** or **Delayed**.")

# Initialize and train Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(np.array([[0], [40], [80], [100]]), np.array([0, 0, 1, 1]))


st.subheader("c. Random Forest → Predicts Deadline Risk")
st.code("""
rf = RandomForestClassifier()
rf.fit(np.array([[10, 2], [50, 1], [90, 0], [100, 0]]), [0, 1, 0, 0])
""")
st.write("Training data (simplified):")
st.table(pd.DataFrame({"Completion": ["10%", "50%", "90%", "100%"], "Task Priority": [2, 1, 0, 0], "Risk": ["High (1)", "Low (0)", "Low (0)", "Low (0)"]}))
st.write("The random forest builds **decision trees** to predict *deadline risk* (High/Low) based on task progress.")
st.write("🧩 *Usage:* During task submission, each update is labeled as “High” or “Low” risk automatically.")

# Initialize and train Random Forest Classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(np.array([[10, 2], [50, 1], [90, 0], [100, 0]]), np.array([0, 1, 0, 0]))


st.subheader("d. SVM (Support Vector Machine) → Sentiment Analysis")
st.code("""
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform([
    "excellent work", "needs improvement", "bad performance", "great job", "average"
])
svm_clf = SVC()
svm_clf.fit(X_train, [1, 0, 0, 1, 0])
""")
st.write("Training data:")
st.table(pd.DataFrame({
    "Comment Text": ["excellent work", "needs improvement", "bad performance", "great job", "average"],
    "Sentiment": ["Positive (1)", "Negative (0)", "Negative (0)", "Positive (1)", "Neutral (0)"]
}))
st.write("The model learns to separate “positive\" and \"negative\" comment patterns using **text vectorization** and **support vectors**.")
st.write("🧩 *Usage:* When a manager writes comments, the system predicts whether feedback sentiment is **Positive** or **Negative**, and saves it along with the review.")

# Initialize and train SVM
vectorizer = CountVectorizer()
X_train_svm = vectorizer.fit_transform([
    "excellent work", "needs improvement", "bad performance", "great job", "average"
])
svm_clf = SVC()
svm_clf.fit(X_train_svm, np.array([1, 0, 0, 1, 0]))


st.subheader("e. K-Means Clustering → 360° Performance Visualization")
st.code("""
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
kmeans.fit(df[["completion", "marks"]])
""")
st.write("* **Input:** Completion % and Marks for all employees.")
st.write("* **Output:** 3 clusters (e.g., High, Medium, Low performers).")
st.write("🧩 *Usage:* This model powers the scatter plot showing employee performance clusters, giving a visual 360° company-level insight.")

# Initialize and train K-Means (dummy data for demonstration)
# In a real app, df would come from actual employee data
dummy_df_kmeans = pd.DataFrame({
    "completion": np.random.randint(0, 101, 50),
    "marks": np.random.uniform(0, 5, 50)
})
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
kmeans.fit(dummy_df_kmeans[["completion", "marks"]])



        "Task stored in Pinecone",
        "Predict marks & On Track status",
        "Sentiment + final marks",
        "Final sentiment summary",
        "Visual department-level analytics"
    ]
})
st.header("Summary")
st.markdown("""> Your system uses **AI to automate performance review, feedback sentiment, and risk assessment**, storing insights in Pinecone for real-time, data-driven dashboards — giving it a top-tier, enterprise-level “HR + project intelligence” feel.""")
