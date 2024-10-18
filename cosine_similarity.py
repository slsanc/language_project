import numpy as np
class Method2:
    @staticmethod
    def compare_texts(text1, text2):
        text1_arr = text1.split()
        text2_arr = text2.split()

        # Combine the words from both strings.
        unique_words = list(set(text1_arr + text2_arr))

        # Compare each string to check the frequency of the words in each string
        text1_vector = []
        text2_vector = []
        for word in unique_words:
            text1_vector.append(text1_arr.count(word))
            text2_vector.append(text2_arr.count(word))

        # Casting the lists to numpy arrays because it is more efficient.
        text1_vector_np = np.array(text1_vector)
        text2_vector_np = np.array(text2_vector)

        # Calculations
        similarity_score = 5

        # Return the similarity score.
        return similarity_score

