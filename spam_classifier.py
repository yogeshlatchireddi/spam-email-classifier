"""
Spam Email Classification
TF-IDF + Logistic Regression

Author: Yogesh Latchireddi
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -------------------------------------------------
# 1. Load Dataset
# -------------------------------------------------
def load_data(filepath="mail_data.csv"):
    try:
        data = pd.read_csv(filepath)
        print("Dataset loaded successfully.")
        return data
    except FileNotFoundError:
        print("Error: 'mail_data.csv' not found in project directory.")
        exit()


# -------------------------------------------------
# 2. Preprocess Data
# -------------------------------------------------
def preprocess(data):
    data = data.copy()

    # Replace null values
    data = data.where(pd.notnull(data), '')

    # Encode labels
    data['Category'] = data['Category'].map({'spam': 0, 'ham': 1})

    X = data['Message']
    y = data['Category']

    return X, y


# -------------------------------------------------
# 3. Train Model
# -------------------------------------------------
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2
    )

    vectorizer = TfidfVectorizer(
        min_df=1,
        stop_words='english',
        lowercase=True
    )

    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_features, y_train)

    # Predictions
    train_predictions = model.predict(X_train_features)
    test_predictions = model.predict(X_test_features)

    # Accuracy
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    print("\nModel Performance:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, test_predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, test_predictions))

    return model, vectorizer


# -------------------------------------------------
# 4. Predict New Email
# -------------------------------------------------
def predict_message(message, model, vectorizer):
    input_data = vectorizer.transform([message])
    prediction = model.predict(input_data)

    print("\nPrediction Result:")
    if prediction[0] == 0:
        print("Spam")
    else:
        print("Ham (Not Spam)")


# -------------------------------------------------
# 5. Main Execution
# -------------------------------------------------
if __name__ == "__main__":
    print("Starting Spam Email Classification...")

    data = load_data()
    X, y = preprocess(data)
    model, vectorizer = train_model(X, y)

    # Example message
    sample_message = "Congratulations! You have won a free lottery ticket."
    predict_message(sample_message, model, vectorizer)

