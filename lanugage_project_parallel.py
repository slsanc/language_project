import pandas as pd
from timeit import default_timer as timer
from multiprocessing import Pool
from cosine_similarity import CosineSimilarityMethod
from fingerprint_method import FingerprintMethod
from semantically_matching_paragraph_counter_method import SmpcMethod
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
DATA_PATH = './resources/data/train150.csv'
OUTPUT_PATH = './output/similarity_results.csv'
FUNCTION_WORDLIST_PATH = 'resources/words_lists/function_words.txt'
CORE_VOCAB_WORDLIST_PATH = 'resources/words_lists/core_vocab_words.txt'


def load_essays(csv_path):
    """
    Load essays from a CSV file and return them as a dictionary.

    Args:
        csv_path (str): The path to the CSV file containing the essays.

    Returns:
        dict:
            A dictionary whose keys are essay IDs and whose values are the
            full texts.
    """
    data_file = pd.read_csv(csv_path)
    return dict(zip(data_file['essay_id'], data_file['full_text']))


def normalize_data(results):
    """
    Normalize similarity scores in the results to a range between 0 and 1 using Min-Max normalization.

    Args:
        results (dict): A dict with (essay_id_a, essay_id_b) keys and similarity score values.
    """
    max_value = max(results.values())
    min_value = min(results.values())

    if max_value == min_value:
        for key in results.keys():
            results[key] = 1.0
    else:
        for key in results:
            results[key] = (results[key] - min_value) / (max_value - min_value)


def save_results_to_csv(results, output_path):
    """
    Save all results to a CSV file.

    Args:
        results (dict): The dictionary containing results from different methods.
        output_path (str): Path to save the output CSV.
    """
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Method', 'Essay A ID', 'Essay B ID', 'Similarity Score'])

        for method_name, method_results in results.items():
            for pair, similarity_score in method_results.items():
                essay_a_id, essay_b_id = pair
                writer.writerow([method_name, essay_a_id, essay_b_id, similarity_score])

def run_comparisons_in_parallel(essay_ids, essay_texts, method, num_docs):
    """
    Run comparisons for a given method in parallel.

    Args:
        essay_ids (list): A list of essay IDs.
        essay_texts (list): A list of all essay texts.
        method (class): The comparison method class to use.
        num_docs (int): Total number of documents.

    Returns:
        dict: A dictionary where keys are (essay_id_a, essay_id_b) tuples and
              values are similarity scores.
    """
    tasks = [(essay_ids[i], essay_ids[j], essay_texts[i], essay_texts[j]) for i in range(num_docs) for j in range(i + 1, num_docs)]

    with Pool(processes=8) as pool:
        results = pool.starmap(method.compare_texts, [(task[2], task[3]) for task in tasks], chunksize=100)

    # Reformat results as a dictionary with essay ID pairs
    comparison_results = {(tasks[i][0], tasks[i][1]): results[i] for i in range(len(results))}
    return comparison_results


def main():
    """
    Using each method, compare each essay to every other essay.

    Save the results as a CSV.
    """
    # Load essays
    essays = load_essays(DATA_PATH)
    essay_ids = list(essays.keys())
    essay_texts = [essays[essay_id] for essay_id in essay_ids]
    num_docs = len(essay_texts)

    # Load the wordlists for SMPC
    SmpcMethod.load_wordlists(FUNCTION_WORDLIST_PATH, CORE_VOCAB_WORDLIST_PATH)

    # Store results
    results = {}

    # Run comparisons for each method
    for method_name, method_class in [("Cosine", CosineSimilarityMethod),
                                      ("SMPC", SmpcMethod),
                                      ("Fingerprint", FingerprintMethod)]:
        print(f"Running {method_name} comparisons...")

        start_time = timer()
        method_results = run_comparisons_in_parallel(essay_ids, essay_texts, method_class, num_docs)
        end_time = timer()

        results[method_name] = method_results
        print(f"{method_name} comparisons completed in {end_time - start_time} seconds.")

    # Normalize the results for SMPC method
    normalize_data(results['SMPC'])

    # Save all results to CSV
    save_results_to_csv(results, OUTPUT_PATH)


if __name__ == '__main__':
    main()
