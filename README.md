# EVAL AI TASK 🤖📊

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://evaltask.streamlit.app/)

An AI-Powered Framework for Automated Task Management, Scheduling, and Role-Based Performance Review. This project, developed as part of a Master's curriculum, showcases a full-stack application for modern organizational workflows using Machine Learning.

---

## 📋 Project Overview

EVAL AI TASK is designed to solve two primary challenges in project management: the manual overhead of assigning and scheduling tasks, and the subjectivity in performance evaluation. It integrates an **Agentic AI** for automation and an **EvalTrack** system for AI-powered, role-based reviews.

### Workflow Diagram

![Workflow](image_e7350e.png)
*(To make this image appear, make sure you have uploaded the diagram file, e.g., `workflow.png`, to your repository and updated the filename in the markdown above)*

---

## ✨ Key Features

* **🤖 Agentic AI Simulation:** A "Project Planner" role that automatically breaks down high-level projects into smaller tasks and assigns them to employees using a scheduling algorithm.
* **🔐 Role-Based Access Control:** A secure, multi-tenant interface with four distinct roles:
    * **Project Planner:** Assigns and schedules all initial tasks.
    * **Team Member:** Updates the completion percentage of their assigned tasks.
    * **Manager:** Reviews submissions, adjusts completion scores, adds comments, and approves tasks.
    * **Client:** Has a read-only view to track the progress of manager-approved tasks for their company.
* **🧠 AI & Machine Learning Layer:**
    * **Linear Regression:** Predicts performance marks (e.g., 0-5) based on the task completion percentage.
    * **Logistic Regression:** Classifies tasks as "On Track" or "Delayed" based on progress.
    * **Support Vector Machine (SVM):** Performs NLP sentiment analysis (Positive/Negative) on manager's written feedback.
* **⚡ Real-time Vector Database:** Uses **Pinecone** to store, filter, and retrieve task data instantly, allowing for efficient queries based on company, employee, or review status.
* **🖥️ Interactive UI:** A user-friendly web interface built with **Streamlit**.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **Database:** Pinecone (Serverless Vector DB)
* **Machine Learning:** Scikit-learn
* **Core Libraries:** Pandas, NumPy

---

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

* Python 3.8+
* Git

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/PranjalAmbwani1305/Eval-Task-.git](https://github.com/PranjalAmbwani1305/Eval-Task-.git)
    cd Eval-Task-
    ```

2.  **Create and activate a virtual environment:**
    * On macOS/Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    * On Windows:
        ```bash
        python -m venv venv
        venv\Scripts\activate
        ```

3.  **Install the required packages:**
    (First, make sure you have a `requirements.txt` file)
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your API Keys:**
    * Create a folder named `.streamlit` in your project's root directory.
    * Inside this folder, create a file named `secrets.toml`.
    * Add your Pinecone API key to this file:
        ```toml
        PINECONE_API_KEY = "YOUR_PINECONE_API_KEY_HERE"
        ```

5.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

Open your web browser and navigate to `http://localhost:8501`.

---

## 📝 Future Improvements

* Integrate real text embedding models (e.g., Sentence-BERT) instead of random vectors for semantic search capabilities.
* Train ML models on a historical dataset for more accurate predictions.
* Implement the employee performance classification model (High/Medium/Low) using aggregated task data.
* Add data visualizations and dashboards for managers and clients.

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
