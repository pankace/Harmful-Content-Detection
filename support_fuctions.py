import nltk
nltk.download('stopwords')
nltk.download('wordnet')

import pandas as pd
import numpy as np
import statistics
import re
from nltk.corpus import stopwords
import scipy.stats as stats
import matplotlib.pyplot as plt

def tokenization(input):
    assert(type(input) == list or type(input) == np.ndarray)

    processing_arm = [sub.split() for sub in input]
    extracted_tokens = []
    for elements in processing_arm:
        if type(elements) == list:
            for value in elements:
                extracted_tokens.append(value)

    release_stage = np.array(extracted_tokens)

    return release_stage



def stop_words_purge(stored_tokens):
    def remove_stopwords_from_text(tokens, stop_words_applied):
        assert (type(tokens) == list or type(tokens) == np.ndarray)
        if type(tokens) != np.ndarray:
            tokens = np.array(tokens)

        filtered_tokens = []
        for token in tokens:
            if token.lower() not in stop_words_applied:
                filtered_tokens.append(token)
        return filtered_tokens

    assert(type(stored_tokens) == list or type(stored_tokens) == np.ndarray)
    dev_mode = 0

    english_stop_words = set(stopwords.words("English"))
    spanish_stop_words = set(stopwords.words("Spanish"))
    if dev_mode == 1:
        print("<------------->")
        print("Purging of spanish and english stop words in progress...")
        print(f"English stop words(length: {len(english_stop_words)}): {english_stop_words}")
        print(f"Spanish stop words(length: {len(spanish_stop_words)}): {spanish_stop_words}")

    post_purge = []
    # streamlined version
    for tokens in stored_tokens:
        tokens_lower = [token.lower() for token in tokens]
        english_clearance = remove_stopwords_from_text(tokens_lower, english_stop_words)
        spanish_clearance = remove_stopwords_from_text(english_clearance, spanish_stop_words)
        post_purge.append(spanish_clearance)
    if dev_mode == 1:
        print("Purge of Spanish and english stop words completed...")
        print("<------------->")

    return post_purge


def replace_additional_dollar_signs(text):
    assert(type(text) == str)
    parts = text.split('$', 1)

    if len(parts) > 1:
        # If there was at least one '$', replace the rest and reconstruct the string
        parts[1] = re.sub(r"\$", "", parts[1])
        return '$'.join(parts)
    else:
        return text

def dataframe_generator(description_column, results_column, c1, c2, column_name, developer_mode):
    def data_cataloging(mutation_col_1, mutation_col_2, data_logging_01, data_logging_02):
        assert(type(mutation_col_1) == str and type(mutation_col_2) == str)
        assert(type(data_logging_01) == list or type(data_logging_01) == np.ndarray)
        assert(type(data_logging_01) == list or type(data_logging_02) == np.ndarray)

        dataframe = pd.DataFrame({
            mutation_col_1: data_logging_01,
            mutation_col_2: data_logging_02
        })
        return dataframe

    assert(type(description_column) == list or type(description_column) == np.ndarray)
    assert(type(results_column) == list or type(results_column) == np.ndarray)
    assert(type(c1) == str)
    assert(type(c2) == str)
    assert(type(column_name) == str)
    assert(type(developer_mode) == int)
    assert(len(description_column) == len(results_column))

    dataframe_desription = description_column
    dataframe_specifics = results_column

    dataframe = data_cataloging(c1, c2, dataframe_desription, dataframe_specifics)
    file_name = f"{column_name.capitalize()} data analysis results.csv"

    #saving it as a csv.
    #-------------------
    dataframe.to_csv(file_name, index=True)
    # -------------------
    if developer_mode == 1:
        print(f"File name: {file_name}")

def enhanced_diagnostics(column_name, input_data, developer_mode):
    assert (type(column_name) == str)
    assert (type(input_data) == list or type(input_data) == np.ndarray)
    assert (type(developer_mode) == int)
    if type(input_data) != np.ndarray:
        input_data = np.array(input_data)

    zeros = 0
    positives = 0
    negatives = 0
    for values in input_data:
        if values == 0:
            zeros += 1
        elif values > 0:
            positives += 1
        elif values < 0:
            negatives += 1
    percentage_of_unique = (len(set(input_data)) / len(input_data))*100

    updated_name = column_name + " enhanced diagnostics"
    description = ["Number of unique values", "Percentage of unique values", "Zeros", "Negatives", "Positives", "Total number of raw input values"]
    outputs = [len(set(input_data)), f"{percentage_of_unique}%", zeros, negatives, positives, len(input_data)]
    dataframe_generator(description,outputs, "Analysis metric", "Result",updated_name.upper(),developer_mode)

    if developer_mode == 1:
        if len(description) == len(outputs):
            for element_A175 in range(len(description)):
                print(f"{description[element_A175]}: {outputs[element_A175]}")

def numeric_analysis_arm(column_name, input_data, developer_mode):
    assert(type(column_name) == str)
    assert(type(input_data) == list or type(input_data) == np.ndarray)
    assert(type(developer_mode) == int)
    if type(input_data) != np.ndarray:
        input_data = np.array(input_data)

    analysis_description = ["Maximum", "Minimum", "Mean", "Median", "Mode", "Standard deviation", "Range", "Skew", "Kurtosis", "Variance"]
    analysis_results = [round(np.max(input_data), 4), round(np.min(input_data), 4), round(np.mean(input_data), 4), round(np.median(input_data), 4),
                        round(statistics.mode(input_data), 4), round(np.std(input_data), 4), round(np.max(input_data) - np.min(input_data), 4),
                        stats.skew(input_data),round(stats.kurtosis(input_data),4), round(statistics.variance(input_data),4)]
    if developer_mode == 1:
        for analysis_outputs in range(len(analysis_results)):
            print(f"{analysis_description[analysis_outputs]}: {analysis_results[analysis_outputs]}")

    dataframe_generator(analysis_description,analysis_results,"Analysis metric", "Result",column_name.upper(), developer_mode)

def remove_emojis(text):
    assert(type(text))
    # Regex pattern to match all emojis
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U0001F700-\U0001F77F"  # alchemical symbols
                           u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                           u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                           u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                           u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                           u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                           u"\U00002702-\U000027B0"  # Dingbats
                           u"\U000024C2-\U0001F251" 
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def link_mention_purge(text):
    assert(type(text) == str)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove Mentions
    text = re.sub(r'@\w+', '', text)
    # Remove everything except letters and necessary whitespace
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def inpurity_purging_protocol(input_storage):
    assert(type(input_storage) == list or type(input_storage) == np.ndarray)
    if type(input_storage) != np.ndarray:
        input_storage = np.array(input_storage)
    cleared = []
    for element in input_storage:
        text_no_urls = link_mention_purge(element)
        baseline = remove_emojis(text_no_urls).replace("(","").replace(")","").strip('"')
        updated_baseline = re.sub(r"(\w)([,.!?;:()-])", r"\1 \2", baseline)
        purged_S01 = updated_baseline.replace("..", "")
        purged_S02 = purged_S01.replace('"',"")
        purged_S03 = purged_S02.replace(";)","")
        purged_S04 = purged_S03.replace("*", "")
        purged_S05 = re.sub(r"@\w+", " ", purged_S04)
        purged_S06 = purged_S05.replace("  "," ")
        purged_S07 = purged_S06.replace("!!", "!")
        purged_S08 = purged_S07.replace("!!!", "!")
        purged_S09 = replace_additional_dollar_signs(purged_S08)
        purged_S10 = purged_S09.replace(",,","")
        purged_S11 = purged_S10.replace("=(","")
        purged_S12 = purged_S11.replace("=>","")
        purged_S13 = purged_S12.replace(" .", "")
        purged_S14 = purged_S13.replace("!","")
        cleared.append(purged_S14)

    cleared_numpy_conversion = np.array(cleared)

    return cleared_numpy_conversion

def advanced_specialist_purification(stored_input):
    assert(type(stored_input) == list or type(stored_input) == np.ndarray)
    if type(stored_input) != np.ndarray:
        stored_input_array = np.array(stored_input)
    # Handle stop words

    #