import numpy as np
class Method2:
    @staticmethod
    def calc_word_frequencies(unique_words, text1_arr, text2_arr):
        """Calculate how frequently words appear in the essays."""
        list_frequencies_1 = []
        list_frequencies_2 = []
        for word in unique_words:
            list_frequencies_1.append(text1_arr.count(word))
            list_frequencies_2.append(text2_arr.count(word))

        return list_frequencies_1, list_frequencies_2

    @staticmethod
    def calc_similarity_score(vector_1, vector_2):
        """
        Calculate the similarity score using 2 vectors represented as numpy arrays using cosine similarity algorithm
        """

        # Calculate the dot product of the vectors and the magnitude of each vector
        dot_prod_vectors = np.dot(vector_1, vector_2)
        vector_1_magnitude = np.linalg.norm(vector_1)
        vector_2_magnitude = np.linalg.norm(vector_2)

        # Calculate the similarity score
        similarity_score = dot_prod_vectors / (vector_1_magnitude * vector_2_magnitude)
        return similarity_score

    @staticmethod
    def text_to_words(essay1, essay2):
        """Split the texts into lists of words"""
        return essay1.split(), essay2.split()

    @staticmethod
    def compare_texts(text1, text2):
        """Main function to compare two texts using the cosine similarity algorithm"""

        # Split the essays into two lists containing the words from each essay
        text1_arr, text2_arr = Method2.text_to_words(text1, text2)

        # Combine the words from both lists of words into a unique list.
        unique_words = list(set(text1_arr + text2_arr))

        # Compare each list of words to the unique words to check the frequency of the words in each list
        text1_vector, text2_vector = Method2.calc_word_frequencies(unique_words, text1_arr, text2_arr)

        # Casting the lists to numpy arrays because it is more efficient for proceeding cosine similarity calculations.
        text1_vector_np = np.array(text1_vector)
        text2_vector_np = np.array(text2_vector)

        # Calculating the similarity score using cosine similarity algorithm
        similarity_score = Method2.calc_similarity_score(text1_vector_np, text2_vector_np)

        return similarity_score # Return the similarity score.