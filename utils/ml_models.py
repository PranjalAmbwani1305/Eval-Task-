from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# --- Linear Regression (marks prediction) ---
lin_reg = LinearRegression().fit([[0], [50], [100]], [0, 2.5, 5])

# --- Logistic Regression (On Track / Delayed) ---
log_reg = LogisticRegression().fit([[0], [40], [80], [100]], [0, 0, 1, 1])

# --- Random Forest (Deadline Risk) ---
rf = RandomForestClassifier().fit(np.array([[10, 2], [50, 1], [90, 0], [100, 0]]), [1, 0, 0, 0])

# --- SVM for Sentiment ---
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform([
    "excellent work", "needs improvement", "bad performance", "great job", "average"
])
svm = SVC()
svm.fit(X_train, [1, 0, 0, 1, 0])

# --- KMeans for 360° clustering ---
def train_kmeans(df):
    if "completion" in df.columns and "marks" in df.columns:
        kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
        df["cluster"] = kmeans.fit_predict(df[["completion", "marks"]])
        return kmeans, df
    return None, df
