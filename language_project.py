import pandas as pd

from cosine_similarity import Method2
from scotts_method import ScottsMethod
import os
import csv

# Paths
DATA_PATH = './resources/data/test.csv'
OUTPUT_PATH = './output/similarity_results.csv'
WORDLIST_PATH = './resources/common_words.txt'

# Function to load essays from the CSV (returns a dictionary with essay_id as keys and full_text as values)
def load_essays(csv_path):
    df = pd.read_csv(csv_path)
    return dict(zip(df['essay_id'], df['full_text']))  # Returns a dictionary {essay_id: full_text}


# Function to save results to a CSV file
def save_results_to_csv(results, output_path):
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Method', 'Essay A ID', 'Essay B ID', 'Similarity Score'])

        for method_name, method_results in results.items():
            for pair, similarity_score in method_results.items():
                essay_a_id, essay_b_id = pair
                writer.writerow([method_name, essay_a_id, essay_b_id, similarity_score])


def main():
    """
    Using each method, compare each essay to every other essay.
    """
    essays = load_essays(DATA_PATH)  # Load essays as {essay_id: full_text}
    results = {}  # Results for each method

    # Create an instance of the class that uses Method 1 (Scott's Method) to compare texts.
    ScottsMethod.load_wordlist(WORDLIST_PATH)

    method1_results = {}
    method2_results = {}
    # method3_results = {}


    essay_ids = list(essays.keys())  # List of essay IDs

    # Compare each pair of essays based on their essay IDs
    for i, essay_a_id in enumerate(essay_ids):
        for essay_b_id in essay_ids[i + 1:]:  # Compare only with essays after essay_a_id
                essay_a_text = essays[essay_a_id]
                essay_b_text = essays[essay_b_id]

                method1_results[(essay_a_id, essay_b_id)] = ScottsMethod.compare_texts(essay_a_text, essay_b_text)
                # method2_results[(essay_a_id, essay_b_id)] = Method2.compare_texts(essay_a_text, essay_b_text)
                # method3_results[(essay_a_id, essay_b_id)] = Method3.compare_texts(essay_a_text, essay_b_text)

    # Store Method1 results
    results['Method1'] = method1_results
    # results['Method2'] = method2_results
    # results['Method3'] = method3_results


    # Save all results to CSV
    save_results_to_csv(results, OUTPUT_PATH)


if __name__ == '__main__':
    main()
