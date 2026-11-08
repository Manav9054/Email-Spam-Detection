import pandas as pd

# Load the Enron email spam dataset
file_path = "data/enron_spam_data.csv"

df = pd.read_csv(file_path)

# Display dataset information
print("\n========== DATASET INFORMATION ==========")

print("\n1. Dataset Shape:")
print(df.shape)

print("\n2. Column Names:")
print(df.columns.tolist())

print("\n3. First 5 Email Records:")
print(df.head().to_string(index=False, max_colwidth=100))

print("\n4. Missing Values:")
print(df.isnull().sum())

print("\n5. Dataset Information:")
df.info()

print("\n========== INSPECTION COMPLETED ==========")
# Check Spam and Ham Distribution
print("\n========== EMAIL LABEL DISTRIBUTION ==========")

print("\nNumber of Emails:")
print(df["Spam/Ham"].value_counts())

print("\nPercentage Distribution:")
print(df["Spam/Ham"].value_counts(normalize=True) * 100)