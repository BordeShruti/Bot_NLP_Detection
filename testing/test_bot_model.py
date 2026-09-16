import pandas as pd
import joblib


# ==========================================
# 1. LOAD DATA
# ==========================================

print("Loading processed Bot Engine data...")

df = pd.read_csv(
    "dataset/processed/bot_features.csv"
)

print("Dataset loaded:", df.shape)


# ==========================================
# 2. LOAD MODEL
# ==========================================

print("\nLoading Bot Engine model...")

model = joblib.load(
    "models/bot_model.pkl"
)

feature_columns = joblib.load(
    "models/bot_model_features.pkl"
)

print("Bot model loaded!")


# ==========================================
# 3. PREPARE FEATURES
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


X = pd.get_dummies(
    df[features],
    columns=["type"],
    dtype=int
)

X = X.reindex(
    columns=feature_columns,
    fill_value=0
)


# ==========================================
# 4. SELECT TEST TRANSACTIONS
# ==========================================

print("\nSelecting sample transactions...")


# 5 normal transactions
normal = df[
    df["isFraud"] == 0
].sample(
    5,
    random_state=42
)


# 5 fraud transactions
fraud = df[
    df["isFraud"] == 1
].sample(
    5,
    random_state=42
)


test_df = pd.concat(
    [normal, fraud]
)


X_test = X.loc[test_df.index]


# ==========================================
# 5. PREDICT
# ==========================================

print("\nRunning Bot Engine predictions...")

predictions = model.predict(
    X_test
)

probabilities = model.predict_proba(
    X_test
)[:, 1]


# ==========================================
# 6. DISPLAY RESULTS
# ==========================================

print("\n==============================================")
print("          BOT ENGINE TEST RESULTS")
print("==============================================")

for i, (idx, row) in enumerate(test_df.iterrows()):

    actual = row["isFraud"]

    prediction = predictions[i]

    probability = probabilities[i]

    print("\n----------------------------------------------")

    print("Transaction:", i + 1)

    print("Transaction Type:", row["type"])

    print(
        "Amount:",
        round(row["amount"], 2)
    )

    print(
        "Time Gap:",
        row["time_gap"]
    )

    print(
        "Previous Transactions:",
        row["transaction_count_before"]
    )

    print(
        "Amount Deviation:",
        round(
            row["amount_deviation"],
            2
        )
    )

    print(
        "Rapid Transaction:",
        row["rapid_transaction"]
    )

    print(
        "Actual Fraud:",
        actual
    )

    print(
        "Fraud Probability:",
        round(
            probability * 100,
            2
        ),
        "%"
    )

    if prediction == 1:

        print(
            "MODEL PREDICTION: FRAUD / HIGH RISK"
        )

    else:

        print(
            "MODEL PREDICTION: NORMAL"
        )


print("\n==============================================")
print("BOT ENGINE TEST COMPLETED")
print("==============================================")