from preprocess import load_dataset, combine_email_text, clean_dataset

# Load the Enron dataset
df = load_dataset("data/enron_spam_data.csv")

print("\n========== PREPROCESSING TEST ==========")

print("\nOriginal Dataset Shape:")
print(df.shape)

# Combine subject and message
df = combine_email_text(
    df,
    subject_column="Subject",
    message_column="Message"
)

# Clean the dataset
df = clean_dataset(
    df,
    text_column="email_text",
    label_column="Spam/Ham"
)

print("\nProcessed Dataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df["email_text"].isnull().sum())

print("\nEmail Label Distribution:")
print(df["Spam/Ham"].value_counts())

print("\nSample Processed Emails:")
print(df[["email_text", "Spam/Ham"]].head(3).to_string(index=False))

print("\n========== TEST COMPLETED ==========")