import pandas as pd
from scotts_method import ScottsMethod
import os
import csv

# Paths
DATA_PATH = './resources/data/essays.csv'
OUTPUT_PATH = './output/similarity_results.csv'
WORDLIST_PATH = './resources/common_words.txt'


# Function to load essays from the CSV
def load_essays(csv_path):
    df = pd.read_csv(csv_path)
    return df['essay'].tolist()  # Assuming there's a column 'essay' with the text


# Function to save results to a CSV file
def save_results_to_csv(results, output_path):
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Method', 'Essay A', 'Essay B', 'Similarity Score'])

        for method_name, method_results in results.items():
            for pair, similarity_score in method_results.items():
                essay_a, essay_b = pair
                writer.writerow([method_name, essay_a, essay_b, similarity_score])


def main():
    """
    Using each method, compare each essay to every other essay.
    """
    essays = load_essays(DATA_PATH) # Load essays from the csv file
    results = {} # results for each method

    # Create an instance of the class that uses Method 1 (Scott's Method) to compare texts.
    # Feed it the list of 100 most common words.
    method1 = ScottsMethod(WORDLIST_PATH)
    # method2 = Method2()

    # Each of the following dictionaries will hold all the results for one of the methods used.
    method1_results = {}

    # Compare each pair of essays
    for i, essay_a in enumerate(essays):
        for j, essay_b in enumerate(essays):
            if i != j:  # Avoid comparing an essay with itself
                similarity_score = method1.compare_texts(essay_a, essay_b)

                method1_results[(f'Essay {i + 1}', f'Essay {j + 1}')] = similarity_score
                # method2_results[(f'Essay {i + 1}', f'Essay {j + 1}')] = similarity_score
                # method3_results[(f'Essay {i + 1}', f'Essay {j + 1}')] = similarity_score
                # method4_results[(f'Essay {i + 1}', f'Essay {j + 1}')] = similarity_score
                # method5_results[(f'Essay {i + 1}', f'Essay {j + 1}')] = similarity_score

    # Store Method1 results
    results['Method1'] = method1_results

    # Add more methods similarly if needed

    # Save all results to CSV
    save_results_to_csv(results, OUTPUT_PATH)


if __name__ == '__main__':
    main()
