import pandas as pd

file_path = "dataset/processed/bot_features.csv"

print("Loading processed dataset...")
df = pd.read_csv(file_path)

print("\n========== SHAPE ==========")
print(df.shape)

print("\n========== COLUMNS ==========")
print(df.columns.tolist())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== FRAUD DISTRIBUTION ==========")
print(df["isFraud"].value_counts())

print("\n========== RAPID TRANSACTION ==========")
print(df["rapid_transaction"].value_counts())

print("\n========== SAME-HOUR TRANSACTIONS ==========")
print(df["same_hour_count"].value_counts())

print("\n========== TIME GAP ==========")
print(df["time_gap"].describe())

print("\n========== HISTORICAL TRANSACTION COUNT ==========")
print(df["transaction_count_before"].describe())

print("\n========== UNIQUE RECEIVERS BEFORE ==========")
print(df["unique_receiver_before"].describe())

print("\n========== HISTORICAL AVERAGE AMOUNT ==========")
print(df["user_avg_amount_before"].describe())

print("\n========== AMOUNT DEVIATION ==========")
print(df["amount_deviation"].describe())

print("\n========== TRANSACTION TYPES ==========")
print(df["type"].value_counts())

print("\n========== FRAUD BY TYPE ==========")
print(
    pd.crosstab(
        df["type"],
        df["isFraud"],
        margins=True
    )
)

print("\n========== FRAUD FEATURE COMPARISON ==========")

comparison_features = [
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

print(
    df.groupby("isFraud")[comparison_features]
    .mean()
    .T
)

print("\n========== FIRST 10 ROWS ==========")
print(df.head(10).to_string())

print("\n========== CHECK COMPLETED ==========")