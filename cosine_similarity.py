import numpy as np

from comparison_util import ComparisonUtil

"""
This class compares the similarities of two texts by using the fingerprint method.

Made By Scott Sanchez and Mihir Bhakta for CS5300.
"""

class Method2:
    @staticmethod
    def calc_word_frequencies(unique_words, text1_arr, text2_arr):
        """
        Given the set of all the words from Essay A and Essay B, calculate:
            * The number of times that each word occurs in essay A
            * The number of times that each word occurs in essay B
        And return those frequencies.

        Args:
            unique_words (list of string):
                The set of all unique words from Essay A and Essay B.

            text1_arr (list of string): Essay A's words as an array.
            text2_arr (list of string): Essay B's words as an array.

        Returns:
            tuple: A tuple containing two lists:
                - list_frequencies_1 (list of int):
                    Frequencies of the words in essay A. The position of each
                    element corresponds to the word at the same location in
                    `unique_words`.

                    In other words, if `unique_words[5]` were "foo", and if
                    "foo" occurred 7 times in Essay A, then
                    `list_frequencies_1[5]` would have a value of 7.

                - list_frequencies_2 (list of int):
                    Frequencies of the words in the second essay, in the same
                    format as `list_frequencies_1`.
        """
        list_frequencies_1 = []
        list_frequencies_2 = []
        for word in unique_words:
            list_frequencies_1.append(text1_arr.count(word))
            list_frequencies_2.append(text2_arr.count(word))

        return list_frequencies_1, list_frequencies_2

    @staticmethod
    def calc_similarity_score(vector_1, vector_2):
        """
        Use the cosine similarity formula to calculate a similarity score
        between the two vectors.

        Args:
            vector_1 (numpy.array):
                The frequency vector of the first essay (for more information
                on what this is, see the documentation for
                `calc_word_frequencies()`).
            vector_2 (numpy.array):
                The frequency vector of the second essay.

        Returns:
            float:
                The similarity score between the two vectors.
        """
        # Calculate the dot product of the vectors and the magnitude of each vector
        dot_prod_vectors = np.dot(vector_1, vector_2)
        vector_1_magnitude = np.linalg.norm(vector_1)
        vector_2_magnitude = np.linalg.norm(vector_2)

        # Calculate the similarity score
        similarity_score = dot_prod_vectors / (vector_1_magnitude * vector_2_magnitude)
        return similarity_score

    @staticmethod
    def compare_texts(text1, text2):
        """
        Compare two texts using the cosine similarity method.

        Args:
            text1 (string) : The first text
            text2 (string) : The second text

        Returns:
            float:
                The similarity score between the two texts
        """
        # Clean the texts before splitting them into words.
        text1 = ComparisonUtil.clean_text(text1)
        text2 = ComparisonUtil.clean_text(text2)

        # Split the essays into two lists containing the words from each essay
        text1_arr = text1.split()
        text2_arr = text2.split()

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