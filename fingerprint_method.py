"""
This class compares the similarities of two texts by using the fingerprint method.

Made for CS5300.
"""
import hashlib

class FingerprintMethod:
    N_GRAM_SIZE = 4  # Class-level constant for n-gram size
    PRIME_MOD = 3    # Class-level constant for prime modulus

    @staticmethod
    def generate_n_grams(text):
        """
        Convert the given text into a list of n-grams. For instance,
        "the quick brown fox jumps over the lazy dog" is converted to:

            ["theq" , "uick" , "brow" , "nfox" , ... , "ydog"]

        The default size of an n-gram is 4 letters. This is set by a
        constant above.

        Args:
            text (str): The text to convert
        Return:
             n_grams (list of str):
                 A list of n-grams generated from the input text
        """
        text = text.replace(" ", "")
        n_grams = [text[i:i + FingerprintMethod.N_GRAM_SIZE] for i in range(len(text) - FingerprintMethod.N_GRAM_SIZE + 1)]
        return n_grams

    @staticmethod
    def hash_ngrams(ngrams_arr):
        """
        Hash each n-gram in the given list using MD5, and return the resulting
        hashes.

        This lets us convert the text from a list of strings to a list
        of integers. This, in turn, makes the data easier for the computer
        to process.

        Args:
            ngrams_arr (list of str): A list of the ngrams to hash.

        Return:
            array of string:
                A list of hash values corresponding to the given list of
                n-grams.
        """
        return [int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16) for word in ngrams_arr]

    @staticmethod
    def select_fingerprints(hash_values):
        """
        Return a list of all the given hash values that are evenly divisible
        by the prime mod. That is, return all hashes where

        hash_value % FingerprintMethod.PRIME_MOD == 0

        Args:
            hash_values (list of int):
                a list of hash values corresponding to the ngrams generated
                from the test.
        Return:
            list of str:
                A list of fingerprints, selected from the given hashes.
        """
        return [hash_value for hash_value in hash_values if hash_value % FingerprintMethod.PRIME_MOD == 0]

    @staticmethod
    def dice_coefficient(fingerprints_a, fingerprints_b):
        """
        Calculate the similarity score (i.e. the Dice Coefficient) between two
        sets of fingerprints.

        Args:
            fingerprints_a (list of int): The list of fingerprint hashes from text A
            fingerprints_b (list of int): The list of fingerprint hashes from text B
        Return:
            float: The Dice Coefficient between text A and text B.
        """
        set_a = set(fingerprints_a)
        set_b = set(fingerprints_b)
        intersection_absolute_value = len(set_a.intersection(set_b))
        return (2 * intersection_absolute_value) / (len(set_a) + len(set_b))

    @staticmethod
    def compare_texts(text_a, text_b):
        """
        Compare the documents using the fingerprint method.

        Args:
            text_a (string) : The first text
            text_b (string) : The second text

        Returns:
            float:
                The similarity score between the two texts
        """
        # Generate and hash n-grams for both documents
        n_grams_a = FingerprintMethod.generate_n_grams(text_a)
        n_grams_b = FingerprintMethod.generate_n_grams(text_b)
        hash_values_a = FingerprintMethod.hash_ngrams(n_grams_a)
        hash_values_b = FingerprintMethod.hash_ngrams(n_grams_b)

        # Select fingerprints
        fingerprints_a = FingerprintMethod.select_fingerprints(hash_values_a)
        fingerprints_b = FingerprintMethod.select_fingerprints(hash_values_b)

        # Compute and return the similarity
        return FingerprintMethod.dice_coefficient(fingerprints_a, fingerprints_b)
