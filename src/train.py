
import numpy as np
import tensorflow as tf
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def train_model(
    df,
    text_column="email_text",
    label_column="Spam/Ham",
    test_size=0.20,
    max_words=10000,
    embedding_dim=32,
    epochs=5,
    batch_size=32
):
    """
    Train a neural network for email spam detection.
    """

    
    # Prepare text and labels
    texts = (
        df[text_column]
        .fillna("")
        .astype("string[python]")
        .str.replace(
            r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]",
            " ",
            regex=True
        )
        .to_numpy(dtype=object)
    )

    labels_series = (
        df[label_column]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"ham": 0, "spam": 1})
    )

    if labels_series.isna().any():
        raise ValueError(
            "Labels must contain only 'spam' and 'ham'."
        )

    labels = labels_series.to_numpy(dtype=np.int32)


    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=42,
        stratify=labels
    )

    # Create text vectorization layer
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_words,
        output_mode="int",
        output_sequence_length=200
    )

    # Adapt vocabulary using training data only
    text_dataset = tf.data.Dataset.from_tensor_slices(
        X_train
    ).batch(64)

    vectorizer.adapt(text_dataset)

    # Build neural network
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(), dtype=tf.string),
        vectorizer,
        tf.keras.layers.Embedding(
            input_dim=max_words,
            output_dim=embedding_dim
        ),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    # Compile model
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Train model
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Evaluate on unseen testing data
    predictions = model.predict(X_test, verbose=0)
    predicted_labels = (predictions >= 0.5).astype(int).flatten()

    results = {
        "accuracy": accuracy_score(y_test, predicted_labels),
        "precision": precision_score(
            y_test, predicted_labels, zero_division=0
        ),
        "recall": recall_score(
            y_test, predicted_labels, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predicted_labels, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predicted_labels
        ),
        "classification_report": classification_report(
            y_test,
            predicted_labels,
            target_names=["Ham", "Spam"],
            zero_division=0
        )
    }

    return model, history, results
