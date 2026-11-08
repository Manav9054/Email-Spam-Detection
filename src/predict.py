
import numpy as np
import tensorflow as tf


def load_trained_model(model_path="models/email_spam_model.keras"):
    """Load the previously saved neural network."""
    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )
    return model


def predict_email(model, subject, message):
    """Predict whether an email is spam or ham."""

    # Combine subject and message
    email_text = str(subject) + " " + str(message)
    email_text = email_text.strip()

    if not email_text:
        raise ValueError("Please enter an email subject or message.")

    # Prepare input for the neural network
    email_input = np.array([email_text], dtype=object)

    # Get spam probability
    prediction = model.predict(email_input, verbose=0)

    spam_probability = float(prediction[0][0])

    # Classify email
    if spam_probability >= 0.5:
        label = "SPAM"
        confidence = spam_probability * 100
    else:
        label = "HAM"
        confidence = (1 - spam_probability) * 100

    return {
        "prediction": label,
        "confidence": confidence,
        "spam_probability": spam_probability * 100
    }
