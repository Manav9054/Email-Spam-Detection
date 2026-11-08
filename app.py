
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from src.predict import load_trained_model, predict_email
from src.train import train_model

st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧",
    layout="wide"
)
# Initialize prediction history
if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []

st.title("📧 Email Spam Detection System")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a page",
    ["Dashboard", "Dataset Management", "Model Training", "Email Prediction","Prediction History"]
)

if page == "Dashboard":

    st.header("Welcome to Email Spam Detection")

    st.write(
        "An interactive Neural Network application "
        "that identifies spam and legitimate emails."
    )

    st.divider()

    st.subheader("Project Overview")

    col1, col2, col3 = st.columns(3)

    # Dataset Status
    with col1:
        if "prepared_df" in st.session_state:
            st.metric(
                "Dataset Status",
                "Prepared"
            )
        else:
            st.metric(
                "Dataset Status",
                "Not Loaded"
            )

    # Total Emails
    with col2:
        if "prepared_df" in st.session_state:
            st.metric(
                "Total Emails",
                len(st.session_state["prepared_df"])
            )
        else:
            st.metric(
                "Total Emails",
                0
            )

    # Model Status
    with col3:
        if "trained_model" in st.session_state:
            st.metric(
                "Model Status",
                "Trained"
            )
        else:
            st.metric(
                "Model Status",
                "Not Trained"
            )
                # Spam and Ham Statistics
    if "prepared_df" in st.session_state:

        dashboard_df = st.session_state["prepared_df"]
        label_column = st.session_state["label_column"]

        spam_count = int(
            (dashboard_df[label_column] == "spam").sum()
        )

        ham_count = int(
            (dashboard_df[label_column] == "ham").sum()
        )

        total_emails = len(dashboard_df)

        spam_percentage = (spam_count / total_emails) * 100
        ham_percentage = (ham_count / total_emails) * 100

        st.divider()
        st.subheader("Email Distribution")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Spam Emails", spam_count)
        col2.metric("Ham Emails", ham_count)
        col3.metric("Spam Percentage", f"{spam_percentage:.2f}%")
        col4.metric("Ham Percentage", f"{ham_percentage:.2f}%")
                # Spam vs Ham Distribution Chart
        st.subheader("Spam vs Ham Distribution")

        fig, ax = plt.subplots(figsize=(2, 2))

        ax.pie(
            [spam_count, ham_count],
            labels=["Spam", "Ham"],
            autopct="%1.1f%%",
            startangle=90
        )

        ax.axis("equal")

        st.pyplot(fig, use_container_width=False)

        plt.close(fig)

    st.divider()

    st.subheader("Application Features")

    st.write("1. Upload and analyse email datasets.")
    st.write("2. Configure dataset labels.")
    st.write("3. Train a Neural Network model.")
    st.write("4. Predict whether an email is SPAM or HAM.")
    st.write("5. View prediction history.")

    st.info(
        "To get started, visit Dataset Management "
        "and prepare your email dataset."
    )

elif page == "Dataset Management":

    st.header("Dataset Management")

    dataset_option = st.radio(
        "Choose Dataset Source",
        ["Use Enron Dataset", "Upload Your Own CSV"]
    )

    df = None

    if dataset_option == "Use Enron Dataset":

        try:
            df = pd.read_csv("data/enron_spam_data.csv")
            st.success("Enron dataset loaded successfully!")

        except FileNotFoundError:
            st.error("Enron dataset file not found.")

    else:

        uploaded_file = st.file_uploader(
            "Upload your email dataset",
            type=["csv"]
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("Dataset uploaded successfully!")

            except Exception as error:
                st.error(f"Unable to read CSV: {error}")

    if df is not None:

        st.subheader("Dataset Preview")

        st.write("Dataset Shape:", df.shape)

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        st.subheader("Select Dataset Columns")

        columns = df.columns.tolist()

        subject_column = st.selectbox(
            "Select Subject Column",
            ["None"] + columns
        )

        message_column = st.selectbox(
            "Select Message Column",
            ["None"] + columns
        )

        label_column = st.selectbox(
            "Select Spam/Ham Label Column",
            columns
        )
        # Get all unique label values from the selected column
        label_values = sorted(
        df[label_column].dropna().astype(str).str.strip().unique().tolist()
        )

        # Automatically identify likely SPAM and HAM labels
        spam_default_index = next(
        (i for i, value in enumerate(label_values)
        if value.lower() in ["spam", "junk", "1", "true", "yes"]),
        0
        )

        ham_default_index = next(
        (i for i, value in enumerate(label_values)
        if value.lower() in ["ham", "legitimate", "not spam", "0", "false", "no"]),
        1 if len(label_values) > 1 else 0
        )

        # Let the user configure label mapping
        spam_label = st.selectbox(
        "Which value represents SPAM?",
        label_values,
        index=spam_default_index
        )

        ham_label = st.selectbox(
        "Which value represents HAM (Not Spam)?",
        label_values,
        index=ham_default_index
        )
        if spam_label == ham_label:
            st.error(
            "SPAM and HAM labels cannot be the same. "
            "Please select different values."
            )

        st.write("Selected Configuration")
        st.write("Subject:", subject_column)
        st.write("Message:", message_column)
        st.write("Label:", label_column)

        st.divider()

        if st.button("Prepare Dataset", type="primary") and spam_label !=ham_label:

            if subject_column == "None" and message_column == "None":
                st.error("Please select at least one text column.")

            elif subject_column == label_column or message_column == label_column:
                st.error("The label column cannot also be used as email text.")

            else:

                text_parts = []

                if subject_column != "None":
                    text_parts.append(
                        df[subject_column].fillna("").astype(str)
                    )

                if message_column != "None":
                    text_parts.append(
                        df[message_column].fillna("").astype(str)
                    )

                df["email_text"] = text_parts[0]

                for part in text_parts[1:]:
                    df["email_text"] = df["email_text"] + " " + part

                df["email_text"] = df["email_text"].str.strip()

                # Standardize labels
                df[label_column] = (
                    df[label_column]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .str.lower()
                )

                # Get selected SPAM and HAM values
                spam_value = str(spam_label).strip().lower()
                ham_value = str(ham_label).strip().lower()

                # Keep only selected labels and non-empty emails
                df = df[
                    (df["email_text"] != "") &
                    (df[label_column].isin([spam_value, ham_value]))
                ].copy()

                # Convert selected labels into standard SPAM and HAM
                df[label_column] = df[label_column].map({
                    spam_value: "spam",
                    ham_value: "ham"
                })
                    # Save prepared dataset in Streamlit session state
                st.session_state["prepared_df"] = df.copy()
                st.session_state["label_column"] = label_column

                st.success("Prepared dataset saved for model training!")

                st.success("Dataset prepared successfully!")

                st.subheader("Prepared Dataset")
                st.write("Total records:", len(df))

                st.dataframe(
                    df[["email_text", label_column]].head(10),
                    use_container_width=True
                )

                st.subheader("Label Distribution")

                st.bar_chart(
                    df[label_column].value_counts()
                )
              
                # Dataset Analysis Metrics
                label_counts = df[label_column].value_counts()

                spam_count = int(label_counts.get("spam", 0))
                ham_count = int(label_counts.get("ham", 0))
                total_emails = len(df)

                spam_percentage = (
                    (spam_count / total_emails) * 100
                    if total_emails > 0 else 0
                )

                ham_percentage = (
                    (ham_count / total_emails) * 100
                    if total_emails > 0 else 0
                )

                st.subheader("Dataset Analysis")

                col1, col2, col3, col4 = st.columns(4)

                col1.metric("Total Emails", total_emails)
                col2.metric("Spam Emails", spam_count)
                col3.metric("Ham Emails", ham_count)
                col4.metric("Missing Values", df.isnull().sum().sum())

                st.write("Spam Percentage:", f"{spam_percentage:.2f}%")
                st.write("Ham Percentage:", f"{ham_percentage:.2f}%")

elif page == "Model Training":

    st.header("Neural Network Model Training")

    if "prepared_df" not in st.session_state:

        st.warning(
            "Please prepare a dataset first in Dataset Management."
        )

    else:

        training_df = st.session_state["prepared_df"]

        st.success("Prepared dataset is available for training!")

        st.write("Total training records:", len(training_df))

        st.subheader("Configure Training Parameters")

        test_size = st.slider(
            "Testing Data Percentage",
            min_value=10,
            max_value=40,
            value=20,
            step=5
        )

        max_words = st.slider(
            "Maximum Vocabulary Size",
            min_value=5000,
            max_value=50000,
            value=10000,
            step=5000
        )

        embedding_dim = st.select_slider(
            "Embedding Dimensions",
            options=[16, 32, 64, 128],
            value=32
        )

        epochs = st.slider(
            "Number of Training Epochs",
            min_value=3,
            max_value=20,
            value=5
        )

        batch_size = st.select_slider(
            "Batch Size",
            options=[16, 32, 64, 128],
            value=32
        )

        st.subheader("Selected Training Configuration")

        st.write("Testing Data:", f"{test_size}%")
        st.write("Vocabulary Size:", max_words)
        st.write("Embedding Dimensions:", embedding_dim)
        st.write("Epochs:", epochs)
        st.write("Batch Size:", batch_size)
        
        st.divider()

        st.subheader("Start Neural Network Training")

        if st.button("Train Neural Network", type="primary"):

            try:
                with st.spinner("Training neural network... Please wait."):

                    model, history, results = train_model(
                        df=training_df,
                        text_column="email_text",
                        label_column=st.session_state["label_column"],
                        test_size=test_size / 100,
                        max_words=max_words,
                        embedding_dim=embedding_dim,
                        epochs=epochs,
                        batch_size=batch_size
                    )

                    st.session_state["trained_model"] = model
                    st.session_state["training_history"] = history
                    st.session_state["training_results"] = results

                st.success("Neural network training completed!")

            except Exception as e:
                st.error(f"Training failed: {e}")
                st.exception(e)

        if "training_results" in st.session_state:

            results = st.session_state["training_results"]

            st.subheader("Model Performance")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Accuracy", f"{results['accuracy'] * 100:.2f}%")
            col2.metric("Precision", f"{results['precision'] * 100:.2f}%")
            col3.metric("Recall", f"{results['recall'] * 100:.2f}%")
            col4.metric("F1 Score", f"{results['f1_score'] * 100:.2f}%")

            st.subheader("Confusion Matrix")

            st.write(
                pd.DataFrame(
                    results["confusion_matrix"],
                    index=["Actual Ham", "Actual Spam"],
                    columns=["Predicted Ham", "Predicted Spam"]
                )
            )

            
            with st.container(border=True):

                st.subheader("Classification Report")

                st.text(results["classification_report"])
            

            # Training Performance Graphs
            if "training_history" in st.session_state:

                st.subheader("Training Performance Analysis")

                history = st.session_state["training_history"]

                col1, col2 = st.columns(2)

                # Accuracy Graph
                with col1:

                    st.write("Training Accuracy")

                    fig, ax = plt.subplots(figsize=(5, 3))

                    ax.plot(
                        history.history["accuracy"],
                        label="Training"
                    )

                    ax.plot(
                        history.history["val_accuracy"],
                        label="Validation"
                    )

                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Accuracy")
                    ax.legend()
                    ax.grid(True)

                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                # Loss Graph
                with col2:

                    st.write("Training Loss")

                    fig2, ax2 = plt.subplots(figsize=(5, 3))

                    ax2.plot(
                        history.history["loss"],
                        label="Training"
                    )

                    ax2.plot(
                        history.history["val_loss"],
                        label="Validation"
                    )

                    ax2.set_xlabel("Epoch")
                    ax2.set_ylabel("Loss")
                    ax2.legend()
                    ax2.grid(True)

                    fig2.tight_layout()
                    st.pyplot(fig2, use_container_width=True)
                    plt.close(fig2)

            # Save Trained Model
            if "trained_model" in st.session_state:

                if st.button("Save Trained Model"):

                    st.session_state["trained_model"].save(
                        "models/email_spam_model.keras"
                    )

                    st.success(
                        "Trained model saved successfully!"
                    )
                
elif page == "Email Prediction":

    st.header("Email Spam Prediction")
    st.write("Enter an email subject and message to check whether it is spam.")

    subject = st.text_input("Email Subject")

    message = st.text_area(
        "Email Message",
        height=200,
        placeholder="Paste the email message here..."
    )

    if st.button("Predict Email", type="primary"):

        if not subject.strip() and not message.strip():
            st.warning("Please enter an email subject or message.")

        else:
            try:
                model = load_trained_model(
                    "models/email_spam_model.keras"
                )

                result = predict_email(
                    model,
                    subject,
                    message
                )
                # Save prediction to history
                st.session_state["prediction_history"].append({
                    "Date & Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Subject": subject,
                    "Message": message,
                    "Prediction": result["prediction"],
                    "Confidence": round(result["confidence"], 2),
                    "Spam Probability": round(result["spam_probability"], 2)
                })

                st.subheader("Prediction Result")

                if result["prediction"] == "SPAM":
                    st.error("This email is likely SPAM.")
                else:
                    st.success("This email is likely HAM (Not Spam).")

                col1, col2 = st.columns(2)

                col1.metric(
                    "Prediction Confidence",
                    f"{result['confidence']:.2f}%"
                )

                col2.metric(
                    "Spam Probability",
                    f"{result['spam_probability']:.2f}%"
                )

            except FileNotFoundError:
                st.error(
                    "Trained model not found. Please train and save the model first."
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.exception(e)
    

elif page == "Prediction History":

    st.header("Prediction History")

    history = st.session_state["prediction_history"]

    if history:

        history_df = pd.DataFrame(history)

        total_predictions = len(history_df)

        spam_predictions = (
            history_df["Prediction"] == "SPAM"
        ).sum()

        ham_predictions = (
            history_df["Prediction"] == "HAM"
        ).sum()

        st.subheader("Prediction Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Predictions", total_predictions)
        col2.metric("Spam Detected", spam_predictions)
        col3.metric("Ham Detected", ham_predictions)

        st.divider()

        st.subheader("Previous Email Predictions")

        st.dataframe(
            history_df,
            use_container_width=True
        )

        st.write("Total Predictions:", total_predictions)

    else:

        st.info(
            "No prediction history available yet. Please test some emails first."
        )
