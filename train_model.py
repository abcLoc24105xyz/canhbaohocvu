import pandas as pd
import numpy as np
import re
import pickle

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from lightgbm import LGBMClassifier


# ===============================
# TEXT CLEAN
# ===============================
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ===============================
# LOAD
# ===============================
df = pd.read_csv("train.csv")

# Chỉ giữ cột cần thiết
selected_cols = [
    "Age",
    "Training_Score_Mixed",
    "Count_F",
    "Tuition_Debt",
    "Gender",
    "Admission_Mode",
    "Club_Member",
    "Advisor_Notes",
    "Personal_Essay",
    "Academic_Status"
]

df = df[selected_cols]

# Clean text
df["Advisor_Notes"] = df["Advisor_Notes"].apply(clean_text)
df["Personal_Essay"] = df["Personal_Essay"].apply(clean_text)
df["combined_text"] = df["Advisor_Notes"] + " " + df["Personal_Essay"]

y = df["Academic_Status"]
X = df.drop(["Academic_Status"], axis=1)

# ===============================
# FEATURE GROUPS
# ===============================
text_col = "combined_text"

num_cols = [
    "Age",
    "Training_Score_Mixed",
    "Count_F",
    "Tuition_Debt"
]

cat_cols = [
    "Gender",
    "Admission_Mode",
    "Club_Member"
]

# ===============================
# PIPELINE
# ===============================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),

        ("text", TfidfVectorizer(max_features=3000, ngram_range=(1,2)), text_col)
    ]
)

model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    class_weight="balanced",
    random_state=42
)

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", model)
])

pipeline.fit(X, y)

with open("academic_warning_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Model retrained & saved.")