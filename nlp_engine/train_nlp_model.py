import pandas as pd
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score


# ==========================================
# 1. LOAD DATA
# ==========================================

file_path = "dataset/PS_20174392719_1491204439457_log.csv"

print("Loading PaySim dataset...")

df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)


# ==========================================
# 2. SORT TRANSACTIONS BY USER AND TIME
# ==========================================

print("\nSorting transactions...")

df = df.sort_values(
    ["nameOrig", "step"]
).reset_index(drop=True)

print("Sorting completed.")


# ==========================================
# 3. CREATE USER TRANSACTION DOCUMENTS
# ==========================================

print("\nCreating user transaction histories...")

user_documents = (
    df.groupby("nameOrig")["type"]
    .apply(lambda x: " ".join(x))
)

print("Number of users:", len(user_documents))


# ==========================================
# 4. CREATE USER FRAUD LABEL
# ==========================================

print("\nCreating user-level fraud labels...")

user_labels = (
    df.groupby("nameOrig")["isFraud"]
    .max()
)

print("\nUser fraud distribution:")
print(user_labels.value_counts())


# ==========================================
# 5. ALIGN DOCUMENTS AND LABELS
# ==========================================

data = pd.DataFrame({
    "document": user_documents,
    "isFraud": user_labels
})

print("\nUser-level dataset:")
print(data.shape)


# ==========================================
# 6. TRAIN / TEST SPLIT
# ==========================================

print("\nCreating train/test split...")

X = data["document"]
y = data["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training users:", len(X_train))
print("Testing users:", len(X_test))


# ==========================================
# 7. TF-IDF
# ==========================================

print("\nCreating TF-IDF representation...")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    lowercase=False
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF testing shape:", X_test_tfidf.shape)


# ==========================================
# 8. TRAIN LOGISTIC REGRESSION
# ==========================================

print("\n====================================")
print("TRAINING NLP MODEL")
print("====================================")

model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

print("Training model...")

model.fit(
    X_train_tfidf,
    y_train
)

print("NLP model training completed!")


# ==========================================
# 9. PREDICTIONS
# ==========================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test_tfidf)

y_probability = model.predict_proba(
    X_test_tfidf
)[:, 1]


# ==========================================
# 10. EVALUATION
# ==========================================

print("\n====================================")
print("NLP MODEL EVALUATION")
print("====================================")

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        digits=4,
        zero_division=0
    )
)


# ==========================================
# 11. ROC-AUC
# ==========================================

try:

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    print(
        "\nROC-AUC:",
        round(roc_auc, 4)
    )

except ValueError:

    print(
        "\nROC-AUC could not be calculated."
    )


# ==========================================
# 12. PR-AUC
# ==========================================

try:

    pr_auc = average_precision_score(
        y_test,
        y_probability
    )

    print(
        "PR-AUC:",
        round(pr_auc, 4)
    )

except ValueError:

    print(
        "PR-AUC could not be calculated."
    )


# ==========================================
# 13. TOP NLP FEATURES
# ==========================================

print("\n====================================")
print("TOP NLP FEATURES")
print("====================================")

feature_names = vectorizer.get_feature_names_out()

coefficients = model.coef_[0]

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients
})

feature_importance = feature_importance.sort_values(
    "coefficient",
    ascending=False
)

print("\nFeatures associated with fraud:")

print(
    feature_importance
    .head(20)
    .to_string(index=False)
)


# ==========================================
# 14. SAVE MODEL
# ==========================================

print("\nSaving NLP model...")

os.makedirs(
    "models",
    exist_ok=True
)

model_path = "models/nlp_tfidf_model.pkl"

vectorizer_path = "models/tfidf_vectorizer.pkl"

joblib.dump(
    model,
    model_path
)

joblib.dump(
    vectorizer,
    vectorizer_path
)


# ==========================================
# 15. COMPLETED
# ==========================================

print("\n====================================")
print("NLP ENGINE TRAINING COMPLETED")
print("====================================")

print("Model saved:")
print(model_path)

print("\nVectorizer saved:")
print(vectorizer_path)