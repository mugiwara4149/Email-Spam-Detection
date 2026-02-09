# ===============================
# Import Required Libraries
# ===============================

# Numpy Library for Numerical Calculations
import numpy as np

# Pandas Library for Dataframe Handling
import pandas as pd

# Pickle Library for Saving Models
import pickle

# Regular Expression Library
import re

# NLTK Library for Natural Language Processing
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

# Train-Test Split
from sklearn.model_selection import train_test_split

# Machine Learning Models
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# Evaluation Metrics
from sklearn.metrics import accuracy_score, confusion_matrix

# Utilities
from scipy.sparse import hstack
from sklearn.preprocessing import MinMaxScaler
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))


# ===============================
# Global Objects
# ===============================

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))


# ===============================
# Text Preprocessing Function
# ===============================

def Pre(text):
    # Remove subject line
    text = re.sub(r'^(s\s*u\s*b\s*j\s*e\s*c\s*t)\s*:', '', text, flags=re.I)
    text = re.sub(r'^subject\s*:', '', text, flags=re.I)

    # Remove non-alphabet characters
    text = re.sub('[^a-zA-Z]', ' ', text)

    # Convert to lowercase and split
    text = text.lower().split()

    # Remove stopwords and apply stemming
    text = [ps.stem(word) for word in text if word not in stop_words]

    return ' '.join(text)


# ===============================
# Feature Extraction Function
# ===============================

def extract_features(text):
    return {
        "length": len(text),
        "num_excl": text.count("!"),
        "num_caps": sum(1 for c in text if c.isupper()),
        "num_digits": sum(1 for c in text if c.isdigit()),
        "has_free": int("free" in text.lower()),
        "has_urgent": int("urgent" in text.lower()),
        "has_win": int("win" in text.lower())
    }


# ===============================
# Main Function
# ===============================

def main():

    # ---------------------------
    # Load Dataset
    # ---------------------------
    df = pd.read_csv('spam_ham_dataset.csv')
    print(df.isnull().sum())
    print(df.head())

    df = df[['label', 'text']]
    print(df.shape)

    # ---------------------------
    # Text Preprocessing
    # ---------------------------
    corpus = [Pre(text) for text in df['text']]

    # ---------------------------
    # Feature Engineering
    # ---------------------------
    feature_df = pd.DataFrame(df['text'].apply(extract_features).tolist())

    scaler = MinMaxScaler()
    feature_scaled = scaler.fit_transform(feature_df)

    # ---------------------------
    # TF-IDF Vectorization
    # ---------------------------
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    X_text = vectorizer.fit_transform(corpus)

    # Combine text & numeric features
    X = hstack([X_text, feature_scaled])

    # Label Encoding
    Y = pd.get_dummies(df['label']).iloc[:, 1].values

    # ---------------------------
    # Train-Test Split
    # ---------------------------
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # ---------------------------
    # Hybrid Voting Classifier
    # ---------------------------
    Hybrid_Model = VotingClassifier(
        estimators=[
            ('svm', LinearSVC(max_iter=10000, random_state=0)),
            ('nb', MultinomialNB()),
            ('lr', LogisticRegression(max_iter=10000))
        ],
        voting='hard'
    )

    Hybrid_Model.fit(x_train, y_train)

    # ---------------------------
    # Random Forest Model
    # ---------------------------
    rf_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    rf_model.fit(x_train.toarray(), y_train)
    rf_pred = rf_model.predict(x_test.toarray())

    print("Accuracy Of Random Forest :", accuracy_score(y_test, rf_pred))
    print("Confusion Matrix of Random Forest :\n", confusion_matrix(y_test, rf_pred))

    # ---------------------------
    # Logistic Regression Probability
    # ---------------------------
    lr_prob_model = LogisticRegression(max_iter=1000)
    lr_prob_model.fit(x_train, y_train)

    # Save probability model
    pickle.dump(lr_prob_model, open("prob_model.pkl", "wb"))

    proba = lr_prob_model.predict_proba(x_test)
    print("Spam confidence of first email:", proba[0][1])

    # ---------------------------
    # Save Models
    # ---------------------------
    pickle.dump(Hybrid_Model, open('Hybrid_Model.pkl', 'wb'))
    pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
    pickle.dump(scaler, open('scaler.pkl', 'wb'))

    # ---------------------------
    # Hybrid Model Evaluation
    # ---------------------------
    hybrid_pred = Hybrid_Model.predict(x_test)

    print("Accuracy Of Hybrid Model :", accuracy_score(y_test, hybrid_pred))
    print("Confusion Matrix of Hybrid Model :\n", confusion_matrix(y_test, hybrid_pred))


# ===============================
# Program Entry Point
# ===============================

if __name__ == "__main__":
    main()
