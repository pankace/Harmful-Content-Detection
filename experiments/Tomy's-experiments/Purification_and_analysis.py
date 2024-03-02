import threading
import pandas as pd
import numpy as np
from support_fuctions import numeric_analysis_arm, enhanced_diagnostics, dataframe_generator, tokenization, remove_emojis, inpurity_purging_protocol

file_path = r"C:\Users\Tomy\PycharmProjects\Experiment - 7\Industrial machine learning course files\Racism classification\Data analysis\tranlated_tweets.csv"
source_data = pd.read_csv(file_path)
developer_mode = 0
loop_viewer = 0

print("<------------------>")
column_names = list(source_data.columns)
column_data_types_detected = []
object_data_columns = []
analysis_compatible_columns = []

for c_names in column_names:
    column_data = source_data[c_names].to_numpy()
    types_detected = column_data.dtype
    column_data_types_detected.append(types_detected)
    if types_detected == "int64":
        analysis_compatible_columns.append(c_names)
    elif types_detected == "object":
        object_data_columns.append(c_names)

if developer_mode == 1:
    if len(column_names) == len(column_data_types_detected):
        for element in range(len(column_names)):
            print(f"Column name: {column_names[element]} | Data type detected: {column_data_types_detected[element]}")
elif developer_mode == 0:
    print("Developer mode inactive")
print("<------------------>")
print()
print(f"Analysis compatible columns: {analysis_compatible_columns}")
numeric_columns = []
object_columns = []

input_mode = True
print("Type '-1' when no more target columns are needed.")
while input_mode == True:
    target = "-1" #input("Column name: ")
    if target == "-1":
        input_mode = False
    elif target != "-1":
        numeric_columns.append(target)

locked_in_columns = numeric_columns
print("Numeric analysis columns:")
print(locked_in_columns)

for names in locked_in_columns:
    c_name_analysis_input = names
    column_isolation = source_data[c_name_analysis_input].to_numpy()
    if type(c_name_analysis_input) == str and (type(column_isolation) == list or type(column_isolation) == np.ndarray):
        numeric_analysis_arm(c_name_analysis_input,column_isolation,developer_mode)
        enhanced_diagnostics(c_name_analysis_input,column_isolation,developer_mode)

text_isolation = source_data["Description Cleaned Translated"].to_numpy()
text_isolation = inpurity_purging_protocol(text_isolation)
unique_locations = []
for tweets in text_isolation:
    if tweets not in unique_locations:
        unique_locations.append(tweets)

target_isolation = source_data["Analysis results"].to_numpy()
numbers_detected = []
for numbers in target_isolation:
    if numbers not in numbers_detected:
        numbers_detected.append(numbers)
    elif numbers in numbers_detected:
        continue

if len(target_isolation) == len(text_isolation):
    print(f"Target isolated data: {len(target_isolation)}")
    print(f"Text isolated data: {len(text_isolation)}")


zero = 0
one = 0
asociation_text = []
asociation_target = []
if len(target_isolation) == len(text_isolation):
    for elements in range(len(target_isolation)):
        asociation_text.append(text_isolation[elements])
        asociation_target.append(target_isolation[elements])

        if target_isolation[elements] == 0:
            zero += 1
        elif target_isolation[elements] == 1:
            one += 1

full_set = one + zero
print(f"Full set: {full_set}")

if developer_mode == 1:
    print(f"Zero targets: {zero}")
    print(f"Ones targets: {one}")

if full_set != 0:
    zero_percentage = (zero / full_set) * 100
    one_percentage = (one / full_set) * 100
else:
    zero_percentage = "Infinity"
    one_percentage = "Infinity"

if one > zero:
    description = ["Cleared", "Flagged", "Percentage ratio of Cleared", "Percentage ratio of Flagged"]
    results = [zero, one, f"{round(zero_percentage, 2)} %", f"{round(one_percentage, 2)} %"]
else:
    description = ["Cleared", "Flagged", "Percentage ratio of flagged", "Percentage ratio of Cleared"]
    results = [zero, one, f"{round(one_percentage, 2)} %", f"{round(zero_percentage, 2)} %"]

# Token analysis with stop words purge integrated
text_isolation = text_isolation.flatten()
tokenized_conversion = tokenization(text_isolation)

# Emoji removal in classic for-loop format
post_purge_storage = []
for pre_purge in tokenized_conversion:
    post_purge = remove_emojis(pre_purge)
    post_purge_storage.append(post_purge)

# Calculate unique tokens after stop words removal
unique_tokens, counts = np.unique(post_purge_storage, return_counts=True)
token_counts = dict(zip(unique_tokens, counts))

# Prepare for dataframe generation
token_keys = list(token_counts.keys())
token_values = list(token_counts.values())
token_des_column = ["Unique tokens", "Total tokens", "Percentage of unique tokens"]
token_res_column = [len(unique_tokens), sum(counts), f"{(len(unique_tokens) / sum(counts) * 100)} %"]
multithreading_deployment = 1
if multithreading_deployment == 1:
    if __name__ == "__main__":
        t1 = threading.Thread(target=dataframe_generator(asociation_text, asociation_target, c1="Cleaned tweet", c2="Tag", column_name="Purification verification", developer_mode=developer_mode))
        t2 = threading.Thread(target=dataframe_generator(description, results, c1="Cleared/Flagged quantity", c2="Analysis output", column_name="Tags analysis", developer_mode=developer_mode))
        t3 = threading.Thread(target=dataframe_generator(token_des_column, token_res_column, c1="Token details", c2="Token count", column_name="Token analysis", developer_mode=developer_mode))
        t4 = threading.Thread(target=dataframe_generator(token_keys, token_values, c1="Individual words", c2="Occurrences", column_name="Frequency of token usage", developer_mode=developer_mode))

        threads = [t1, t2, t3, t4]
        for individual_threads in threads:
            individual_threads.start()
        for initiated_threads in threads:
            initiated_threads.join()
