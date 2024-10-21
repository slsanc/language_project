import pandas as pd
from timeit import default_timer as timer

from cosine_similarity import Method2
from fingerprint_method import FingerprintMethod
from semantically_matching_paragraph_counter_method import SmpcMethod
import os
import csv

"""
This program compares each essay in the database with every other essay.

Each pair of essays is compared three times - once with each of the following methods:

    * The Fingerprint Method
    * The Cosine Similarity Method
    * The Semantically Matching Paragraph Counter (SMPC) Method

The results are then saved to a CSV file for further analysis.

Made by Scott Sanchez and Mihir Bhakta for CS5300: Introduction to Artificial Intelligence.
"""
# Paths
DATA_PATH = './resources/data/train2.csv'
OUTPUT_PATH = './output/similarity_results.csv'
FUNCTION_WORDLIST_PATH = 'resources/words_lists/function_words.txt'
CORE_VOCAB_WORDLIST_PATH = 'resources/words_lists/core_vocab_words.txt'

def load_essays(csv_path):
    """
    Load essays from a CSV file and return them as a dictionary

    Args:
        csv_path (str): The path to the CSV file containing the essays.

    Returns:
        dict:
            A dictionary whose keys are essay IDs, and whose values are the
            full texts.
    """
    data_file = pd.read_csv(csv_path)
    return dict(zip(data_file['essay_id'], data_file['full_text']))  # Returns a dictionary {essay_id: full_text}

def normalize_data(results):
    """
    Normalize similarity scores in the results to a range between 0 and 1 using
    Min-Max normalization.

    Args:
        results (dict):
            A dict whose keys are tuples of (essay_id_a,essay_id_b) and whose 
            values are tuples of (time_taken, similarity_score). The similarity
            scores will be normalized. All else is left alone.
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

def save_results_to_csv(results, output_path):
    """
    Save all results to a csv.
    """
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

    Save the results as a csv.
    """
    essays = load_essays(DATA_PATH)  # Load essays as {essay_id: full_text}
    results = {}  # Results for each method

    # Load the wordlists used by the Semantically Matching Paragraph Counter method.
    SmpcMethod.load_wordlists(FUNCTION_WORDLIST_PATH, CORE_VOCAB_WORDLIST_PATH)

    cosine_method_results = {}
    fingerprint_method_results = {}
    smpc_method_results = {}

    essay_ids = list(essays.keys())  # List of essay IDs

    # Compare each pair of essays based on their essay IDs
    for i, essay_a_id in enumerate(essay_ids):
        for essay_b_id in essay_ids[i + 1:]:  # Compare only with essays after essay_a_id
                essay_a_text = essays[essay_a_id]
                essay_b_text = essays[essay_b_id]

                start = timer()
                similarity_score = SmpcMethod.compare_texts(essay_a_text, essay_b_text)
                end = timer()
                time_taken = end - start
                cosine_method_results[(essay_a_id, essay_b_id)] = (time_taken, similarity_score)

                start = timer()
                similarity_score = Method2.compare_texts(essay_a_text, essay_b_text)
                end = timer()
                time_taken = end - start
                fingerprint_method_results[(essay_a_id, essay_b_id)] = (time_taken, similarity_score)

                start = timer()
                similarity_score = FingerprintMethod.compare_texts(essay_a_text, essay_b_text)
                end = timer()
                time_taken = end - start
                smpc_method_results[(essay_a_id, essay_b_id)] = (time_taken, similarity_score)




    normalize_data(cosine_method_results)

    # Store Method1 results
    results['Method1'] = cosine_method_results
    results['Method2'] = fingerprint_method_results
    results['Method3'] = smpc_method_results


    # Save all results to CSV
    save_results_to_csv(results, OUTPUT_PATH)


if __name__ == '__main__':
    main()
