import nltk
nltk.download('stopwords')
nltk.download('wordnet')

import xgboost as xgb
import numpy as np
from joblib import dump
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd

def remove_stopwords_from_text(tokens, stop_words_applied):
    assert(type(tokens) == list or type(tokens) == np.ndarray)
    if type(tokens) != np.ndarray:
        tokens = np.array(tokens)

    filtered_tokens = []
    for token in tokens:
        if token.lower() not in stop_words_applied:
            filtered_tokens.append(token)
    return filtered_tokens

def ml_diagnostics(y_test, predictions):
    assert(type(y_test) == list or type(y_test) == np.ndarray)
    assert(type(predictions) == list or type(predictions) == np.ndarray)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    details = ["Accuracy", "Precision", "Recall", "F1"]
    results = [round(accuracy, 4), round(precision, 4), round(recall, 4), round(f1, 4)]
    if len(details) == len(results):
        print("Diagnostic metrics")
        for elements in range(len(results)):
            print(f"{details[elements]}: {results[elements]}")

    if (accuracy > 0.9) and (precision > 0.9):
        print()
        print(f"Go touch come grass. You got accuracy to reach {round(accuracy, 4)} and precision to reach {round(precision, 4)}")
        if (recall > 0.8) and (f1 > 0.8):
            print("Bruv. Go live life outside. This is already accurate like you had OCD writing this.")
            print(f"Recall is {round(recall, 4)} and F1 score is {round(f1, 4)}. Go touch some grass. Seriously.")



def gradient_booster(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5)
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    return y_test, predictions, clf

def upgraded_gradient_booster(X, y):
    if type(y) != np.ndarray:
        y = np.array(y)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    #Baseline classifier
    base_clf = DecisionTreeClassifier(max_depth=1)
    #AdaBoost paired with blase classifier
    ada_clf = AdaBoostClassifier(base_estimator=base_clf, n_estimators=50, algorithm="SAMME.R", learning_rate=0.666)
    ada_clf.fit(X_train, y_train)
    predictions  = ada_clf.predict(X_test)

    return y_test, predictions, ada_clf


purified_data = pd.read_csv(r"C:\Users\Tomy\PycharmProjects\Experiment - 7\Industrial machine learning course files\Racism classification\Data analysis\Results\Token based\Post_purging\38\Purification verification data analysis results.csv")

cleaned_tweet = purified_data["Cleaned tweet"].to_numpy()
tags = purified_data["Tags"].to_numpy()

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
print("...")
print(f"X type: {type(X)}")
print(f"y type: {type(y)}")
print("...")

standard = 0
if standard == 1:
    print("Baseline deployed")
    y_test_output, predicted_results, clf = gradient_booster(X, y)
    ml_diagnostics(y_test_output, predicted_results)
    dump(clf, "Standard_baseline_boosted_model.joblib")
elif standard == 0:
    print("Upgraded baseline deployed")
    y_test_2_output, predicted_2, ada_clf = upgraded_gradient_booster(X, y)
    ml_diagnostics(y_test_2_output, predicted_2)
    dump(ada_clf,"ada_boosted_model.joblib")


"""
Benchmark(label encoder = True)
--------------
Diagnostic metrics
Accuracy: 0.74
Precision: 0.7405405405405405
Recall: 0.7098445595854922
F1: 0.7248677248677249

Alternative alteration
n_estimators=100, learning_rate=0.1, max_depth=5
------------------
Diagnostic metrics
Accuracy: 0.76
Precision: 0.7975460122699386
Recall: 0.6735751295336787
F1: 0.7303370786516852

Benchmark to beat
----------
Upgraded baseline deployed
Diagnostic metrics
Accuracy: 0.9133333333333333
Precision: 0.9466666666666667
Recall: 0.8875
F1: 0.9161290322580645
"""
