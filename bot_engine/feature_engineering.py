import pandas as pd
import os

# ==========================================
# 1. LOAD DATASET
# ==========================================

file_path = "dataset/PS_20174392719_1491204439457_log.csv"

print("Loading PaySim dataset...")

df = pd.read_csv(file_path)

print("Dataset loaded!")
print("Original shape:", df.shape)


# ==========================================
# 2. SORT BY USER AND TIME
# ==========================================

print("\nSorting transactions...")

df = df.sort_values(
    ["nameOrig", "step"]
).reset_index(drop=True)

print("Sorting completed.")


# ==========================================
# 3. TIME GAP
# ==========================================

print("\nCalculating time gaps...")

df["time_gap"] = (
    df.groupby("nameOrig")["step"]
    .diff()
    .fillna(0)
)

print("Time gap created.")


# ==========================================
# 4. TRANSACTION COUNT BEFORE CURRENT TX
# ==========================================

print("\nCalculating historical transaction count...")

df["transaction_count_before"] = (
    df.groupby("nameOrig").cumcount()
)

print("Historical transaction count created.")


# ==========================================
# 5. UNIQUE RECEIVERS BEFORE CURRENT TX
# ==========================================

print("\nCalculating historical unique receivers...")

# Create indicator for whether this receiver
# has already appeared for the same sender.

df["receiver_seen_before"] = (
    df.groupby(["nameOrig", "nameDest"])
    .cumcount()
)

df["unique_receiver_before"] = (
    df["receiver_seen_before"]
    .eq(0)
    .groupby(df["nameOrig"])
    .cumsum()
    - 1
)

df["unique_receiver_before"] = (
    df["unique_receiver_before"].clip(lower=0)
)

print("Historical unique receiver feature created.")


# ==========================================
# 6. HISTORICAL AVERAGE AMOUNT
# ==========================================

print("\nCalculating historical average amount...")

df["amount_sum_before"] = (
    df.groupby("nameOrig")["amount"]
    .cumsum()
    - df["amount"]
)

df["user_avg_amount_before"] = (
    df["amount_sum_before"] /
    df["transaction_count_before"].replace(0, pd.NA)
)

df["user_avg_amount_before"] = (
    df["user_avg_amount_before"]
    .fillna(0)
)

print("Historical average amount created.")


# ==========================================
# 7. AMOUNT DEVIATION
# ==========================================

print("\nCalculating amount deviation...")

df["amount_deviation"] = (
    df["amount"] -
    df["user_avg_amount_before"]
).abs()

print("Amount deviation created.")


# ==========================================
# 8. SAME-HOUR TRANSACTION COUNT
# ==========================================

print("\nCalculating same-hour transaction count...")

df["same_hour_count"] = (
    df.groupby(["nameOrig", "step"])
    .cumcount()
)

print("Same-hour transaction count created.")


# ==========================================
# 9. RAPID TRANSACTION
# ==========================================

print("\nCreating rapid transaction feature...")

df["rapid_transaction"] = (
    (
        (df["time_gap"] == 0) &
        (df["transaction_count_before"] > 0)
    )
    |
    (df["time_gap"] == 1)
).astype(int)

print("Rapid transaction feature created.")


# ==========================================
# 10. HISTORICAL TRANSFER COUNT
# ==========================================

print("\nCalculating historical transfer count...")

df["transfer_indicator"] = (
    df["type"] == "TRANSFER"
).astype(int)

df["transfer_count_before"] = (
    df.groupby("nameOrig")["transfer_indicator"]
    .cumsum()
    - df["transfer_indicator"]
)

print("Historical transfer count created.")


# ==========================================
# 11. HISTORICAL CASH-OUT COUNT
# ==========================================

print("\nCalculating historical cash-out count...")

df["cashout_indicator"] = (
    df["type"] == "CASH_OUT"
).astype(int)

df["cashout_count_before"] = (
    df.groupby("nameOrig")["cashout_indicator"]
    .cumsum()
    - df["cashout_indicator"]
)

print("Historical cash-out count created.")


# ==========================================
# 12. SELECT FEATURES
# ==========================================

print("\nSelecting Bot Engine features...")

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
    "cashout_count_before",
    "isFraud"
]

bot_df = df[features].copy()


# ==========================================
# 13. CREATE OUTPUT FOLDER
# ==========================================

output_folder = "dataset/processed"

os.makedirs(
    output_folder,
    exist_ok=True
)


# ==========================================
# 14. SAVE PROCESSED DATA
# ==========================================

output_file = (
    f"{output_folder}/bot_features.csv"
)

print("\nSaving processed dataset...")

bot_df.to_csv(
    output_file,
    index=False
)


# ==========================================
# 15. FINAL INFORMATION
# ==========================================

print("\n====================================")
print("FEATURE ENGINEERING COMPLETED")
print("====================================")

print("\nOutput file:")
print(output_file)

print("\nProcessed shape:")
print(bot_df.shape)

print("\nFeatures:")
print(bot_df.columns.tolist())

print("\nFraud distribution:")
print(
    bot_df["isFraud"].value_counts()
)

print("\nRapid transaction distribution:")
print(
    bot_df["rapid_transaction"].value_counts()
)

print("\nFirst 5 rows:")
print(
    bot_df.head().to_string()
)