import pandas as pd


def load_dataset(file_path):
    """Load an email dataset from a CSV file."""
    return pd.read_csv(file_path)


def combine_email_text(df, subject_column, message_column):
    """Combine subject and message into one text column."""

    subject = df[subject_column].fillna("").astype(str)
    message = df[message_column].fillna("").astype(str)

    df = df.copy()
    df["email_text"] = subject + " " + message

    return df


def clean_dataset(df, text_column, label_column):
    """Remove empty records and standardize labels."""

    df = df.copy()

    df[text_column] = df[text_column].fillna("").astype(str).str.strip()
    df[label_column] = df[label_column].astype(str).str.strip().str.lower()

    df = df[df[text_column] != ""]
    df = df[df[label_column].isin(["spam", "ham"])]

    return df.reset_index(drop=True)