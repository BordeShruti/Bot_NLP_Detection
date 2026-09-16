import pandas as pd

file_path = "dataset/PS_20174392719_1491204439457_log.csv"

df = pd.read_csv(file_path)

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())

print("\n========== TRANSACTION TYPES ==========")
print(df["type"].value_counts())

print("\n========== FRAUD DISTRIBUTION ==========")
print(df["isFraud"].value_counts())

print("\n========== FRAUD PERCENTAGE ==========")
print(df["isFraud"].value_counts(normalize=True) * 100)

print("\n========== FLAGGED FRAUD ==========")
print(df["isFlaggedFraud"].value_counts())

print("\n========== AMOUNT STATISTICS ==========")
print(df["amount"].describe())

print("\n========== TIME RANGE ==========")
print("Minimum step:", df["step"].min())
print("Maximum step:", df["step"].max())

print("\n========== FRAUD BY TRANSACTION TYPE ==========")
print(
    pd.crosstab(
        df["type"],
        df["isFraud"],
        margins=True
    )
)