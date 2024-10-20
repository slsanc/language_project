import pandas as pd
from timeit import default_timer as timer

from cosine_similarity import Method2
from fingerprint_method import FingerprintMethod
from scotts_method import ScottsMethod
import os
import csv

# Paths
DATA_PATH = './resources/data/train2.csv'
OUTPUT_PATH = './output/similarity_results.csv'
FUNCTION_WORDLIST_PATH = 'resources/words_lists/function_words.txt'
CORE_VOCAB_WORDLIST_PATH = 'resources/words_lists/core_vocab_words.txt'

# Function to load essays from the CSV (returns a dictionary with essay_id as keys and full_text as values)
def load_essays(csv_path):
    df = pd.read_csv(csv_path)
    return dict(zip(df['essay_id'], df['full_text']))  # Returns a dictionary {essay_id: full_text}

def normalize_data(results):
    """
    Normalize results to a range between 0 and 1.
    :param results:
    :return:
    """
    all_scores = [measurement[1] for measurement in results.values()]
    max_value = max(all_scores)
    min_value = min(all_scores)

    if max_value == min_value:
        # If all values are the same, normalize them to 1.0. We do this
        # manually, because running the calculation would require dividing by
        # zero.
        for key in results:
            time_taken, similarity_score = results[key]
            results[key] = (time_taken, 1)
    else:
        # Otherwise, apply the Min-Max normalization formula
        for key in results:
            time_taken, similarity_score = results[key]

            normalized_score = (similarity_score - min_value) / (max_value - min_value)
            results[key] = (time_taken, normalized_score)


# Function to save results to a CSV file
def save_results_to_csv(results, output_path):
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Method', 'Essay A ID', 'Essay B ID', 'Time Taken', 'Similarity Score'])

        for method_name, method_results in results.items():
            for pair, measurement in method_results.items():
                essay_a_id, essay_b_id = pair
                time_taken, similarity_score = measurement
                writer.writerow([method_name, essay_a_id, essay_b_id, time_taken, similarity_score])


def main():
    """
    Using each method, compare each essay to every other essay.
    """
    essays = load_essays(DATA_PATH)  # Load essays as {essay_id: full_text}
    results = {}  # Results for each method

    # Create an instance of the class that uses Method 1 (Scott's Method) to compare texts.
    ScottsMethod.load_wordlists(FUNCTION_WORDLIST_PATH, CORE_VOCAB_WORDLIST_PATH)

    method1_results = {}
    method2_results = {}
    method3_results = {}

    essay_ids = list(essays.keys())  # List of essay IDs

    # Compare each pair of essays based on their essay IDs
    for i, essay_a_id in enumerate(essay_ids):
        for essay_b_id in essay_ids[i + 1:]:  # Compare only with essays after essay_a_id
                essay_a_text = essays[essay_a_id]
                essay_b_text = essays[essay_b_id]

                start = timer()
                similarity_score_1 = ScottsMethod.compare_texts(essay_a_text, essay_b_text)
                end = timer()
                time_taken = end - start
                method1_results[(essay_a_id, essay_b_id)] = (time_taken, similarity_score_1)

                start = timer()
                similarity_score_2 = Method2.compare_texts(essay_a_text, essay_b_text)
                end = timer()
                time_taken = end - start
                method2_results[(essay_a_id, essay_b_id)] = (time_taken, similarity_score_2)

                start = timer()
                similarity_score_3 = FingerprintMethod.compare_texts(essay_a_text, essay_b_text)
                end = timer()
                time_taken = end - start
                method3_results[(essay_a_id, essay_b_id)] = (time_taken, similarity_score_3)




    normalize_data(method1_results)

    # Store Method1 results
    results['Method1'] = method1_results
    results['Method2'] = method2_results
    results['Method3'] = method3_results


    # Save all results to CSV
    save_results_to_csv(results, OUTPUT_PATH)


if __name__ == '__main__':
    main()
