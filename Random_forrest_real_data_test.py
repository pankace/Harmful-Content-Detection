import time

import nltk
from sklearn.ensemble import RandomForestClassifier
start_time = time.time()

nltk.download('stopwords')
nltk.download('wordnet')

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, recall_score
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.decomposition import PCA, TruncatedSVD
from nltk.tokenize import word_tokenize
import pandas as pd

def binary_classifier(X, y):
    def error_metrics(y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        return accuracy, precision, recall, f1

    upgrade_deployment = 0
    if type(y) != np.ndarray:
        y = np.array(y)

    if upgrade_deployment == 1:
        svd = TruncatedSVD(n_components=75)
        X = svd.fit_transform(X)
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prediction
    prediction = model.predict(X_test)
    print(f"Prediction: {prediction}")
    accuracy_metric, precision_metric, recall_metric, f1_metric = error_metrics(y_test, prediction)
    return accuracy_metric, precision_metric, recall_metric, f1_metric

def remove_stopwords_from_text(tokens, stop_words_applied):
    assert(type(tokens) == list or type(tokens) == np.ndarray)
    if type(tokens) != np.ndarray:
        tokens = np.array(tokens)

    filtered_tokens = []
    for token in tokens:
        if token.lower() not in stop_words_applied and token.lower() not in spanish_stop_words:
            filtered_tokens.append(token)
    return filtered_tokens


purified_data = pd.read_csv(r"C:\Users\Tomy\PycharmProjects\Experiment - 7\Industrial machine learning course files\Racism classification\Data analysis\Results\Token based\Post_purging\32\Purification verification data analysis results.csv")

cleaned_tweet = purified_data["Cleaned tweet"].to_numpy()
tags = purified_data["Tag"].to_numpy()

# Tokenization
token_conversion = []
for tweets in cleaned_tweet:
    post_conversion = word_tokenize(tweets)
    token_conversion.append(post_conversion)

print("Tokenization complete")
for i in range(3):
    print(token_conversion[i])

# Stopwords purge
english_stop_words = set(stopwords.words("English"))
spanish_stop_words = set(stopwords.words("Spanish"))
print("<------------->")
print("Purging of spanish and english stop words in progress...")
print(f"English stop words(length: {len(english_stop_words)}): {english_stop_words}")
print(f"Spanish stop words(length: {len(spanish_stop_words)}): {spanish_stop_words}")

stage_0 = []
# English stopwords removal
for stage_0_element in token_conversion:
    english_cleaned_text = remove_stopwords_from_text(stage_0_element, english_stop_words)
    stage_0.append(english_cleaned_text)

stop_words_stage_1 = []
# Spanish stop words removal
stage_1 = []
for stage_1_elements in stage_0:
    spanish_cleaned_text = remove_stopwords_from_text(stage_1_elements, spanish_stop_words)
    stage_1.append(spanish_cleaned_text)
print("Purge of Spanish and english stop words completed...")
print("<------------->")


# lammination
lemmatizer = WordNetLemmatizer()
lamminized_tokens = []
for post_processed_tokens in stage_1:
    processed_tokens = []
    for token in post_processed_tokens:
        if token not in english_stop_words and token not in spanish_stop_words and token.isalpha():
            # Lemmatize the token and append to the result list
            lemmatized_token = lemmatizer.lemmatize(token)
            processed_tokens.append(lemmatized_token)
    lamminized_tokens.append(processed_tokens)

preprocessed_texts = [" ".join(tokens) for tokens in lamminized_tokens]
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(preprocessed_texts)
y = tags
if type(y) != np.ndarray:
    y = np.array(y)
print(f"X type: {type(X)}")
print(f"y type: {type(y)}")
accuracy_metric, precision_metric, recall_metric, f1_metric = binary_classifier(X, y)
end_time = time.time()
print(f"Accuracy: {round(accuracy_metric, 4)}")
print(f"Precision: {round(precision_metric, 4)}")
print(f"Recall: {round(recall_metric, 4)}")
print(f"F1: {round(f1_metric, 4)}")
processing_time = end_time - start_time
if processing_time < 60:
    print(f"Processing time: {processing_time} seconds")
elif processing_time >= 60:
    hours_conversion = (processing_time)/60
    if hours_conversion > 1 and hours_conversion < 2:
        print(f"Processing time: {hours_conversion} hour")
    else:
        print(f"Processing time: {hours_conversion} hours")
"""
Benchmark to beat:
------------------
Prediction: [0 1 1 1 1 0 1 0 1 1 0 0 0 1 1 0 1 0 0 0 1 0 0 1 0 0 0 1 1 0 1 1 1 1 0 1 0
 0 1 1 0 0 1 1 0 1 1 1 0 0 1 1 0 1 1 0 1 0 1 1 0 1 0 0 0 1 0 0 0 1 0 1 0 0
 1 1 0 1 1 0 1 1 0 1 0 0 0 1 1 0 1 0 0 1 0 0 1 0 0 0 1 0 1 0 0 1 1 1 0 1 1
 0 0 0 0 1 0 0 1 0 0 0 1 1 1 0 1 1 0 1 0 0 0 1 1 0 0 0 0 0 1 1 0 1 0 1 1 1
 1 0 0 1 0 0 1 1 1 0 0 0 1 1 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 0 1 0 0 1 1 1 1
 1 0 0 0 0 1 0 1 1 1 1 0 1 0 0 0 0 0 1 1 1 0 0 0 1 0 1 0 0 1 0 0 1 1 1 1 1
 0 0 0 1 1 1 0 1 0 0 1 1 1 1 0 0 1 1 0 0 1 1 0 0 1 0 0 0 1 1 1 1 0 0 1 1 1
 0 0 0 1 0 1 0 1 1 0 0 0 0 1 0 0 0 1 1 0 0 0 0 1 0 0 1 1 0 0 1 0 0 0 0 0 1
 1 0 0 1]
Accuracy: 79.0%
Precision: 79.43262%
Recall: 76.71233%
F1-Score: 78.04878%
Processing time: 3.4483914375305176 seconds

Observation: Any attempt to increase accuracy makes these metrics decrease. Any assist is welcomed.
Note: The processing time fluctuates drasticallly from 3.45 seconds to sometimes 8 seconds. Do not consider that an important consideration or anything if it stays in the seconds for processing time.
"""
