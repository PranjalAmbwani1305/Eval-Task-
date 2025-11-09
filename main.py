import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.svm import SVC

# --- PINE CONE INTEGRATION ---
try:
    from pinecone import Pinecone, Index
    st.session_state.pinecone_mock = False
except ImportError:
    st.warning("Pinecone client not found. Running in mock (Pandas DataFrame) mode.")
    st.session_state.pinecone_mock = True

# Configuration
INDEX_NAME = "task"
VECTOR_DIMENSION = 1024

# --- Helper Functions for Pinecone/Mock ---

@st.cache_resource
def init_pinecone_connection():
    """Initializes Pinecone and connects to the 'task' index."""
    if st.session_state.pinecone_mock:
        return None, "Mock Index"
        
    try:
        # 1. Initialize
        pc = Pinecone(
            api_key=st.secrets["PINE_CONE_API_KEY"],
            environment=st.secrets["PINE_CONE_ENVIRONMENT"]
        )
        
        # 2. Check and Create Index if necessary
        if INDEX_NAME not in pc.list_indexes().names:
            pc.create_index(
                INDEX_NAME, 
                dimension=VECTOR_DIMENSION, 
                metric='cosine', 
                spec=pc.IndexSpec() # Or your specific cloud spec
            )
            st.toast(f"Pinecone Index '{INDEX_NAME}' created.", icon='🚀')
            
        # 3. Connect to Index
        index = pc.Index(INDEX_NAME)
        st.toast(f"Connected to Pinecone Index '{INDEX_NAME}'.", icon='🔗')
        return index
    except Exception as e:
        st.error(f"Pinecone connection failed: {e}. Falling back to mock mode.")
        st.session_state.pinecone_mock = True
        return None
        
def mock_get_embedding(text):
    """Mocks a 1024-dimension embedding for task text."""
    # In a real app, use a model like 'BGE-large-en-v1.5' or similar 1024-dim model.
    np.random.seed(hash(text) % (2**32 - 1)) # Simple way to get repeatable "embeddings"
    return np.random.rand(VECTOR_DIMENSION).astype('float32').tolist()

# --- Data Loading and Processing (Pinecone/Mock) ---
if 'df' not in st.session_state:

    @st.cache_data(show_spinner="Loading and processing task vectors from Pinecone/DataFrame...")
    def load_and_process_data(pinecone_index):
        """Simulates fetching from Pinecone and applying ML logic."""
        
        # 0. Core Data Generation (Used for both Pinecone Upsert and Mock DF)
        np.random.seed(42)
        n_tasks = 100
        
        data = {
            'task_id': [f'TASK-{i:03d}' for i in range(1, n_tasks + 1)],
            'task_name': [f'Project Alpha Task {i}' for i in range(1, n_tasks + 1)],
            'employee': np.random.choice(['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'], n_tasks),
            'department': np.random.choice(['Dev', 'Design', 'Ops', 'HR'], n_tasks),
            'priority': np.random.choice(['High', 'Medium', 'Low'], n_tasks, p=[0.3, 0.5, 0.2]),
            'status': np.random.choice(['Pending', 'In-Progress', 'Completed', 'Overdue'], n_tasks, p=[0.25, 0.45, 0.20, 0.10]),
            'completion%': np.random.uniform(0, 100, n_tasks).round(0),
            'marks': np.random.uniform(60, 100, n_tasks).round(1),
            'timestamp': [datetime.now() - timedelta(days=int(np.random.rand() * 60)) for _ in range(n_tasks)],
            'deadline': [datetime.now() + timedelta(days=int(np.random.rand() * 30 - 15)) for _ in range(n_tasks)],
            'feedback_text': [np.random.choice(["Great job, highly efficient.", "Needs to improve focus.", "Completed on time.", "Some minor issues, but overall fine.", "Excellent quality work."], 1)[0] for _ in range(n_tasks)],
            'reviewed': np.random.choice([True, False], n_tasks, p=[0.1, 0.9])
        }
        df = pd.DataFrame(data)

        # Apply data consistency rules
        df.loc[df['status'] == 'Completed', 'completion%'] = 100
        df.loc[df['status'] == 'Pending', 'completion%'] = 0

        # --- Vectorization & Upsert to Pinecone ---
        if not st.session_state.pinecone_mock:
            vectors_to_upsert = []
            for _, row in df.iterrows():
                task_text = f"{row['task_name']} by {row['employee']} in {row['department']}"
                vector = mock_get_embedding(task_text)
                
                # Metadata (Simplified to ensure no conflicting types)
                metadata = row.drop(labels=['task_id', 'timestamp', 'deadline']).to_dict()
                metadata['timestamp'] = row['timestamp'].isoformat()
                metadata['deadline'] = row['deadline'].isoformat()
                
                vectors_to_upsert.append((row['task_id'], vector, metadata))
            
            # Upsert in batches
            try:
                pinecone_index.upsert(vectors=vectors_to_upsert)
            except Exception as e:
                 st.error(f"Pinecone Upsert failed: {e}")

        # --- ML Logic (Applied to local DF/metadata after fetch) ---

        # 1. Logistic Regression / Random Forest (Status Classification)
        def mock_status_prediction(row):
            if row['status'] == 'Completed': return 'On Track', 0.05
            if row['status'] == 'Overdue': return 'Delayed', np.random.uniform(0.7, 0.99)
            if row['completion%'] < 50 and row['deadline'] < (datetime.now() + timedelta(days=3)):
                return 'At Risk', np.random.uniform(0.5, 0.8)
            return 'On Track', np.random.uniform(0.01, 0.4)

        df[['predicted_status', 'prediction_probability']] = df.apply(
            lambda row: pd.Series(mock_status_prediction(row)), axis=1
        )

        # 2. K-Means Clustering (Performance Grouping)
        perf_data = df.groupby('employee')[['marks', 'completion%']].mean().reset_index()
        X = perf_data[['marks', 'completion%']].values
        
        if len(X) >= 3:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            perf_data['cluster'] = kmeans.fit_predict(X)
            cluster_means = perf_data.groupby('cluster')['marks'].mean()
            mapping = {
                cluster_means.idxmax(): 'High Performers',
                cluster_means.idxmin(): 'Needs Support',
                cluster_means.drop([cluster_means.idxmax(), cluster_means.idxmin()]).index[0]: 'Average'
            }
            perf_data['performance_group'] = perf_data['cluster'].map(mapping)
            df = df.merge(perf_data[['employee', 'performance_group']], on='employee', how='left')
        else:
             df['performance_group'] = np.random.choice(['High Performers', 'Average', 'Needs Support'], n_tasks)

        # 3. SVM (Sentiment Analysis - Mock)
        def mock_sentiment_analysis(text):
            text_lower = text.lower()
            if 'great' in text_lower or 'excellent' in text_lower or 'efficient' in text_lower:
                return 'Positive'
            if 'improve' in text_lower or 'minor issues' in text_lower or 'poor' in text_lower:
                return 'Negative'
            return 'Neutral'
        
        df['feedback_sentiment'] = df['feedback_text'].apply(mock_sentiment_analysis)

        return df

    # --- PINE CONE / MOCK INITIALIZATION ---
    pinecone_index = init_pinecone_connection()
    st.session_state.df = load_and_process_data(pinecone_index)
    st.session_state.pinecone_index = pinecone_index
    
# Helper function to update the DataFrame and Pinecone (Mock Pinecone Metadata Update)
def update_task_status(task_id, new_status, reviewed=False):
    # 1. Update Local Cache (DataFrame)
    st.session_state.df.loc[st.session_state.df['task_id'] == task_id, 'status'] = new_status
    if reviewed:
        st.session_state.df.loc[st.session_state.df['task_id'] == task_id, 'reviewed'] = True
        
    # 2. Update Pinecone Metadata
    if not st.session_state.pinecone_mock and st.session_state.pinecone_index:
        try:
            st.session_state.pinecone_index.update(
                id=task_id,
                set_metadata={'status': new_status, 'reviewed': reviewed}
            )
            st.toast(f"Task {task_id} status updated in Pinecone!", icon='💾')
        except Exception as e:
            st.error(f"Pinecone update failed for {task_id}: {e}")
            
    st.toast(f"Task {task_id} status updated to {new_status}!", icon='✅')

# --- UI/Layout & Custom CSS (Unchanged from previous version for brevity) ---
st.set_page_config(
    page_title="Agentic AI & EvalTrack Manager View",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""<style>...</style>""", unsafe_allow_html=True) # CSS omitted for brevity
st.title("Agentic AI & EvalTrack Manager View")

# --- Main Tabs Layout ---
tab_dashboard, tab_task_manager, tab_feedback_ai, tab_admin = st.tabs(
    ["📊 Dashboard", "📋 Task Manager", "💬 Feedback & AI Insights", "🛠️ Admin Tools"]
)

with tab_dashboard:
    st.subheader("Executive View: Real-time Performance & Risk")
    # ... (Task Summary Panel, AI Alerts, Goal Tracker, Heatmap logic - unchanged) ...

with tab_task_manager:
    st.subheader("Operational Manager View: Task Creation & Visualization")

    col_create, col_suggest = st.columns([2, 1])
    
    with col_create:
        with st.form("new_task_form"):
            # ... (Form inputs) ...
            
            # Agentic AI Logic - Auto-Suggest Assignment (Mock using Pinecone Query Pattern)
            available_employees = st.session_state.df['employee'].unique().tolist()
            workload = st.session_state.df[st.session_state.df['status'] == 'In-Progress'].groupby('employee')['task_id'].count()
            least_busy_employee = workload.idxmin() if not workload.empty else available_employees[0]
            
            # MOCK PINE CONE AGENTIC QUERY
            if not st.session_state.pinecone_mock and st.session_state.pinecone_index:
                # 1. Create a "Skill Vector" for the required task
                required_skill_text = f"{task_name} {task_desc} High Priority"
                required_skill_vector = mock_get_embedding(required_skill_text)
                
                # 2. Query Pinecone for the closest "Employee Skill/Past Task" vectors
                # (This simulates a vector-based query for optimal assignment)
                query_results = st.session_state.pinecone_index.query(
                    vector=required_skill_vector,
                    top_k=3,
                    filter={'status': 'Completed'} # Only query vectors of completed tasks (demonstrated skills)
                )
                if query_results.matches:
                    top_employee_from_vector = query_results.matches[0].metadata['employee']
                    suggested_employee = f"🤖 Suggested (Vector Search): {top_employee_from_vector}"
                else:
                    suggested_employee = f"🤖 Suggested (Workload): {least_busy_employee}"
            else:
                 suggested_employee = f"🤖 Suggested (Workload): {least_busy_employee} (Workload: {workload.min() if not workload.empty else 0})"
            
            selected_employee = st.selectbox(
                "Assign To", 
                options=available_employees, 
                help=suggested_employee
            )
            
            # ... (Other form fields) ...
            submitted = st.form_submit_button("Create Task")
            
            if submitted:
                # ... (New task creation logic) ...
                new_task_id = f"TASK-{len(st.session_state.df) + 1:03d}"
                new_row = pd.DataFrame([{...}]) # Create new DF row
                st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
                
                # PINE CONE UPSERT FOR NEW TASK
                if not st.session_state.pinecone_mock and st.session_state.pinecone_index:
                    new_task_text = f"{task_name} by {selected_employee}"
                    new_vector = mock_get_embedding(new_task_text)
                    
                    # Prepare metadata (ensure all ML outputs are saved back as metadata)
                    new_metadata = new_row.iloc[0].drop(labels=['task_id', 'timestamp', 'deadline']).to_dict()
                    new_metadata['timestamp'] = new_row['timestamp'].iloc[0].isoformat()
                    new_metadata['deadline'] = new_row['deadline'].iloc[0].isoformat()
                    
                    try:
                        st.session_state.pinecone_index.upsert(vectors=[
                            (new_task_id, new_vector, new_metadata)
                        ])
                        st.success(f"Task vector {new_task_id} upserted to Pinecone.")
                    except Exception as e:
                        st.error(f"Pinecone Upsert for new task failed: {e}")
                
                st.success(f"Task '{task_name}' created and assigned to {selected_employee}!")
    
    # ... (Predictive Deadlines and Task Visualizations - unchanged) ...

with tab_feedback_ai:
    # ... (Feedback Input and Sentiment Analysis - unchanged) ...
    # Note: Saving feedback also involves an update_task_status/index.update call
    # which is handled by a similar pattern to the one in tab_admin.
    pass

with tab_admin:
    st.subheader("Managerial Actions & Appraisal Tools")

    # --- 7. Managerial Actions & Approvals ---
    # ... (Task selection logic - unchanged) ...
    
    # The action buttons call the Pinecone-integrated helper:
    # update_task_status(task_to_act, 'Completed', reviewed=True) 

    # ... (Rest of the Appraisal Card logic - unchanged) ...
