import pandas as pd
import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score
)

# ==========================================
# 1. LOAD PROCESSED DATA
# ==========================================

file_path = "dataset/processed/bot_features.csv"

print("Loading processed dataset...")
df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)


# ==========================================
# 2. CHRONOLOGICAL TRAIN / TEST SPLIT
# ==========================================

print("\nCreating chronological train/test split...")

# PaySim step represents time.
# Use earlier transactions for training
# and later transactions for testing.

split_step = df["step"].quantile(0.80)

train_df = df[df["step"] <= split_step].copy()
test_df = df[df["step"] > split_step].copy()

print("Split step:", split_step)

print("\nTraining shape:", train_df.shape)
print("Testing shape:", test_df.shape)

print("\nTraining fraud distribution:")
print(train_df["isFraud"].value_counts())

print("\nTesting fraud distribution:")
print(test_df["isFraud"].value_counts())


# ==========================================
# 3. SELECT FEATURES
# ==========================================

features = [
    "step",
    "type",
    "amount",
    "time_gap",
    "transaction_count_before",
    "unique_receiver_before",
    "user_avg_amount_before",
    "amount_deviation",
    "same_hour_count",
    "rapid_transaction",
    "transfer_count_before",
    "cashout_count_before"
]

target = "isFraud"


# ==========================================
# 4. ONE-HOT ENCODE TRANSACTION TYPE
# ==========================================

print("\nEncoding transaction type...")

X_train = pd.get_dummies(
    train_df[features],
    columns=["type"],
    dtype=int
)

X_test = pd.get_dummies(
    test_df[features],
    columns=["type"],
    dtype=int
)

# Make sure both datasets have identical columns
X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)

y_train = train_df[target]
y_test = test_df[target]

print("Training features:", X_train.shape[1])
print("Feature columns:")
print(X_train.columns.tolist())


# ==========================================
# 5. TRAIN RANDOM FOREST
# ==========================================

print("\n====================================")
print("TRAINING RANDOM FOREST")
print("====================================")

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("Training model...")
model.fit(X_train, y_train)

print("Model training completed!")


# ==========================================
# 6. PREDICTIONS
# ==========================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# ==========================================
# 7. MODEL EVALUATION
# ==========================================

print("\n====================================")
print("MODEL EVALUATION")
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
# 8. ROC-AUC
# ==========================================

try:
    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    print("\nROC-AUC:", round(roc_auc, 4))

except ValueError:
    print("\nROC-AUC could not be calculated.")


# ==========================================
# 9. PR-AUC
# ==========================================

try:
    pr_auc = average_precision_score(
        y_test,
        y_probability
    )

    print("PR-AUC:", round(pr_auc, 4))

except ValueError:
    print("PR-AUC could not be calculated.")


# ==========================================
# 10. FEATURE IMPORTANCE
# ==========================================

print("\n====================================")
print("FEATURE IMPORTANCE")
print("====================================")

importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print(
    importance.to_string(index=False)
)


# ==========================================
# 11. SAVE MODEL
# ==========================================

print("\nSaving model...")

os.makedirs(
    "models",
    exist_ok=True
)

model_path = "models/bot_model.pkl"

joblib.dump(
    model,
    model_path
)

# Save feature column order
feature_path = "models/bot_model_features.pkl"

joblib.dump(
    X_train.columns.tolist(),
    feature_path
)

print("\n====================================")
print("BOT ENGINE TRAINING COMPLETED")
print("====================================")

print("Model saved:")
print(model_path)

print("\nFeature list saved:")
print(feature_path)