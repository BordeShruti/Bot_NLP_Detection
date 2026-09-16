import pandas as pd
import joblib


# ==========================================
# 1. LOAD DATA
# ==========================================

print("Loading PaySim dataset...")

df = pd.read_csv(
    "dataset/PS_20174392719_1491204439457_log.csv"
)

print("Dataset loaded:", df.shape)


# ==========================================
# 2. LOAD NLP MODEL
# ==========================================

print("\nLoading NLP model...")

model = joblib.load(
    "models/nlp_tfidf_model.pkl"
)

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

print("NLP model loaded!")


# ==========================================
# 3. CREATE USER HISTORIES
# ==========================================

print("\nCreating user transaction histories...")

df = df.sort_values(
    ["nameOrig", "step"]
)

user_documents = (
    df.groupby("nameOrig")["type"]
    .apply(lambda x: " ".join(x))
)


# ==========================================
# 4. CREATE USER LABELS
# ==========================================

user_labels = (
    df.groupby("nameOrig")["isFraud"]
    .max()
)


users = pd.DataFrame({
    "document": user_documents,
    "isFraud": user_labels
})


# ==========================================
# 5. SELECT USERS
# ==========================================

print("\nSelecting test users...")

normal_users = users[
    users["isFraud"] == 0
].sample(
    5,
    random_state=42
)

fraud_users = users[
    users["isFraud"] == 1
].sample(
    5,
    random_state=42
)

test_users = pd.concat([
    normal_users,
    fraud_users
])


# ==========================================
# 6. TF-IDF
# ==========================================

print("\nConverting histories to TF-IDF...")

X_test = vectorizer.transform(
    test_users["document"]
)


# ==========================================
# 7. PREDICTION
# ==========================================

print("\nRunning NLP predictions...")

predictions = model.predict(X_test)

probabilities = model.predict_proba(
    X_test
)[:, 1]


# ==========================================
# 8. DISPLAY RESULTS
# ==========================================

print("\n==============================================")
print("           NLP ENGINE TEST RESULTS")
print("==============================================")


for i, ((user_id, row), prediction, probability) in enumerate(
    zip(test_users.iterrows(), predictions, probabilities)
):

    print("\n----------------------------------------------")

    print("User:", user_id)

    print(
        "Transaction History:",
        row["document"]
    )

    print(
        "Actual Fraud:",
        row["isFraud"]
    )

    print(
        "NLP Risk Score:",
        round(probability * 100, 2),
        "%"
    )

    if prediction == 1:

        print(
            "MODEL PREDICTION: HIGH RISK USER"
        )

    else:

        print(
            "MODEL PREDICTION: NORMAL USER"
        )


print("\n==============================================")
print("NLP ENGINE TEST COMPLETED")
print("==============================================")