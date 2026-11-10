# Email Spam Detection System

An interactive email spam classification application built with Python,
TensorFlow/Keras and Streamlit. The application allows a user to prepare
an email dataset, configure and train a neural-network text classifier,
evaluate its performance, classify individual emails, and review
predictions made during the current app session.

**GitHub repository:** https://github.com/Manav9054/Email-Spam-Detection

## 1. Project objectives

The project was developed as a portfolio project to demonstrate an
end-to-end machine-learning workflow for text classification:

1.  Import an email dataset from a CSV file.
2.  Select the columns containing the subject, message and class label.
3.  Combine email text and standardise the SPAM/HAM labels.
4.  Inspect the prepared dataset and its class distribution.
5.  Configure and train a neural-network model.
6.  Evaluate the model on a held-out test split.
7.  Save and reload the trained model.
8.  Enter an email and obtain a SPAM or HAM prediction.
9.  Review prediction history within the Streamlit application.

## 2. Features

### Dashboard

-   Shows whether a dataset has been prepared in the current app
    session.
-   Shows model status for the current session.
-   Displays the prepared dataset's spam and ham counts and percentages.
-   Displays a spam-versus-ham distribution chart.

### Dataset Management

-   Supports the included Enron CSV dataset or a user-uploaded CSV.
-   Displays a dataset preview.
-   Lets the user select the subject, message and label columns.
-   Supports subject-only, message-only, or combined subject-and-message
    text, according to the selected columns.
-   Lets the user specify which source label represents SPAM and which
    represents HAM. This supports labels such as `spam`/`ham`,
    `junk`/`legitimate`, and `1`/`0`.
-   Checks that SPAM and HAM are assigned different source values.
-   Combines the selected text fields into an `email_text` field,
    handles missing text values, removes empty email text and retains
    the two selected classes.
-   Displays the prepared data, class distribution and dataset analysis
    metrics.

### Model Training

-   Provides controls for the test-data percentage, maximum vocabulary
    size, embedding dimensions, number of epochs and batch size.
-   Uses a stratified train/test split with a fixed random state.
-   Fits the text vectorisation vocabulary using training text, rather
    than the test set.
-   Trains a TensorFlow/Keras neural network.
-   Displays evaluation metrics, a confusion matrix, a classification
    report, and training/validation accuracy and loss graphs.
-   Allows the trained model to be saved as
    `models/email_spam_model.keras`.

### Email Prediction

-   Accepts an email subject and message.
-   Loads the saved Keras model and predicts whether the email is SPAM
    or HAM.
-   Displays the prediction, confidence and spam probability.
-   Rejects an empty subject-and-message input.

### Prediction History

-   Displays predictions made during the current Streamlit session.
-   Shows date and time, subject, message, prediction, confidence and
    spam probability.
-   Displays summary counts for recorded predictions.
-   History is held in Streamlit session state. It is not a permanent
    database and can be cleared when the app session restarts or is
    reset.
-   Prediction history is displayed in the application; the project does
    not provide an Excel or CSV history-download feature.

## 3. Technology stack

-   **Python 3.11.9**
-   **Streamlit** -- interactive application interface
-   **TensorFlow / Keras** -- neural-network model
-   **Pandas and NumPy** -- dataset handling and numerical operations
-   **Scikit-learn** -- train/test splitting and evaluation metrics
-   **Matplotlib** -- charts and training plots
-   **Seaborn** -- included in the project dependencies
-   **NLTK** -- included in the project dependencies; the described
    model pipeline uses Keras text vectorisation

## 4. Dataset

The project was developed and tested using the Enron spam dataset from:

https://github.com/MWiechmann/enron_spam_data

The source CSV used in the project contained **33,716 rows and five
columns**:

-   `Message ID`
-   `Subject`
-   `Message`
-   `Spam/Ham`
-   `Date`

The source labels contained 17,171 spam emails and 16,545 ham emails.
After preparing the data, combining the selected email text and removing
records with empty email text, the application produced **33,665 usable
records**:

  Class         Prepared records
  ----------- ------------------
  SPAM                    17,120
  HAM                     16,545
  **Total**           **33,665**

The application is not restricted to this dataset. Users can upload
another CSV and map its columns and source labels in Dataset Management.
The uploaded dataset must contain usable email text and two classes that
the user maps to SPAM and HAM.

### Dataset availability in the repository

The dataset is intentionally excluded from GitHub through `.gitignore`
rules for `data/`, CSV files and ZIP files. A person cloning the
repository must obtain a dataset separately or upload their own CSV
through the application.

## 5. Neural-network approach

The training pipeline in `src/train.py` performs the following steps:

1.  Normalises class labels and maps HAM to `0` and SPAM to `1`.
2.  Cleans control characters from email text.
3.  Splits the data into training and test sets using stratification.
4.  Creates a Keras `TextVectorization` layer with a configurable
    vocabulary limit and a sequence length of 200.
5.  Adapts the vectorisation layer on the training text only.
6.  Builds and trains the neural network.

The model architecture is:

``` text
Input email text
      |
TextVectorization
      |
Embedding
      |
GlobalAveragePooling1D
      |
Dense layer (64 units, ReLU)
      |
Dropout (0.3)
      |
Dense output (1 unit, Sigmoid)
      |
SPAM probability
```

The model is compiled with the Adam optimiser, binary cross-entropy loss
and accuracy as a training metric. A sigmoid output is interpreted as
the probability of SPAM; a value of 0.5 or above is classified as SPAM,
otherwise as HAM.

## 6. Model evaluation

In the recorded experiment using the prepared Enron dataset, the model
achieved the following results on the held-out test split:

  Metric        Result
  ----------- --------
  Accuracy      99.14%
  Precision     98.78%
  Recall        99.53%
  F1-score      99.16%

The reported test set contained **6,733 records**. These figures
describe that particular split and configuration. They do not establish
performance on every email source, future data or real-world deployment.
Results can vary with the dataset, label quality, split and training
configuration.

## 7. Project structure

``` text
Email-Spam-Detection/
├── data/                       # Local datasets (not committed)
├── models/                     # Saved models (not committed)
├── notebooks/                  # Notebook workspace
├── src/
│   ├── preprocess.py           # Dataset loading and text/label cleaning helpers
│   ├── train.py                # Neural-network training and evaluation
│   └── predict.py              # Model loading and email prediction
├── Screenshots/                # Application screenshots shown in this README
├── app.py                      # Streamlit application and page navigation
├── requirements.txt            # Python dependencies
├── README.md
└── .gitignore
```

The `venv/` virtual environment is created locally and is not committed
to GitHub.

## 8. Installation and setup

### Requirements

-   Windows, macOS or Linux
-   Python 3.11 recommended (the project was developed with Python
    3.11.9)
-   Git
-   An email dataset in CSV format

### Clone the repository

``` bash
git clone https://github.com/Manav9054/Email-Spam-Detection.git
cd Email-Spam-Detection
```

### Create and activate a virtual environment

**Windows PowerShell:**

``` powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation for the current terminal session, run:

``` powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**

``` bash
python3.11 -m venv venv
source venv/bin/activate
```

### Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The dependency file lists the packages used by the project. It does not
pin exact package versions.

## 9. Running the application

From the project root, with the virtual environment activated:

``` bash
streamlit run app.py
```

Streamlit will provide a local URL in the terminal and normally open the
application in a browser.

### First-time workflow

1.  Open **Dataset Management**.
2.  Choose the Enron dataset if you have placed the expected CSV in the
    local `data/` folder, or upload your own CSV.
3.  Select the subject, message and label columns that match your file.
4.  Choose the source value representing SPAM and the source value
    representing HAM.
5.  Prepare the dataset and inspect the preview and class counts.
6.  Open **Model Training** and configure the test split, vocabulary
    size, embedding dimensions, epochs and batch size.
7.  Train the model and inspect the evaluation results and graphs.
8.  Save the trained model.
9.  Open **Email Prediction**, enter a subject and/or message, and
    select the prediction action.
10. Open **Prediction History** to review predictions made during the
    current session.

For a new or uploaded dataset, prepare it and train a model before
expecting predictions for that dataset. The saved model file is not
included in the GitHub repository.

## 10. Saved files and GitHub exclusions

The `.gitignore` file excludes local virtual environments, datasets,
model files, caches and IDE settings. In particular:

-   `data/` and CSV/ZIP files are not committed.
-   `models/` and saved model formats such as `.keras` are not
    committed.
-   `venv/` is not committed.

This keeps large or machine-specific files out of the source repository.
After cloning, users need to supply a dataset and train/save their own
model before using model-based prediction.

## 11. Application screenshots

The repository includes screenshots of the implemented application:

### Dashboard

![Dashboard](Screenshots/dashboard.png)

### Dataset Management

![Dataset Management](Screenshots/Dataset_management_1.png) ![Dataset
Management](Screenshots/dataset_management_2.png) ![Dataset
Management](Screenshots/dataset_management_3.png) ![Dataset
Management](Screenshots/dataset_management_4.png)

### Model Training

![Model Training Settings](Screenshots/model_training_settings.png)
![Model Training Results](Screenshots/model_training_result.png)
![Training Graphs](Screenshots/model_traning_graphs.png)

### Email Prediction

![Email Prediction](Screenshots/email_prediction.png)

## 12. Current limitations and possible future work

The implemented application is an educational/portfolio project, not a
production email-security service. Areas that could be developed further
include:

-   Persisting prediction history in a database so it survives app
    restarts.
-   Adding automated tests and broader validation against different
    datasets.
-   Evaluating the model on an independent dataset from a different
    email source.
-   Versioning saved models and recording the training configuration
    with each model.
-   Improving deployment, monitoring and security before any real-world
    use.

## 13. Author

**Manav Parmar**

GitHub: https://github.com/Manav9054
