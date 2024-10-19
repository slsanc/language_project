import hashlib

"""
This class compares the similarities of two texts by using the fingerprint method.

Made for CS5300
"""

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

        :return: A list of n-grams
        """
        text = text.replace(" ", "")
        n_grams = [text[i:i + FingerprintMethod.N_GRAM_SIZE] for i in range(len(text) - FingerprintMethod.N_GRAM_SIZE + 1)]
        return n_grams

    @staticmethod
    def hash_n_grams(n_grams):
        """
        Hash each n-gram using MD5 and return the sequence of hashes.

        This lets us convert the text from a list of strings to a list
        of integers. This, in turn, makes the data easier for the computer
        to process.

        :return: A list of hash values
        """
        return [int(hashlib.md5(n_gram.encode('utf-8')).hexdigest(), 16) for n_gram in n_grams]

    @staticmethod
    def select_fingerprints(hash_values):
        """
        Return a list of all the given hash values that are evenly divisible
        by the prime mod. That is, return all hashes where

        hash_value % FingerprintMethod.PRIME_MOD == 0

        :return: A list of fingerprints
        """
        return [hash_value for hash_value in hash_values if hash_value % FingerprintMethod.PRIME_MOD == 0]

    @staticmethod
    def dice_coefficient(fingerprints_a, fingerprints_b):
        """
        Calculate the Dice Coefficient between two sets of fingerprints.

        :return: Dice Coefficient similarity score
        """
        set_a = set(fingerprints_a)
        set_b = set(fingerprints_b)
        intersection_absolute_value = len(set_a.intersection(set_b))
        return (2 * intersection_absolute_value) / (len(set_a) + len(set_b))

    @staticmethod
    def compare_texts(text_a, text_b):
        """
        Compare the documents using the fingerprint method.

        :return: The similarity score (i.e. The Dice Coefficient)
        """
        # Generate and hash n-grams for both documents
        n_grams_a = FingerprintMethod.generate_n_grams(text_a)
        n_grams_b = FingerprintMethod.generate_n_grams(text_b)
        hash_values_a = FingerprintMethod.hash_n_grams(n_grams_a)
        hash_values_b = FingerprintMethod.hash_n_grams(n_grams_b)

        # Select fingerprints
        fingerprints_a = FingerprintMethod.select_fingerprints(hash_values_a)
        fingerprints_b = FingerprintMethod.select_fingerprints(hash_values_b)

        # Compute similarity using Dice Coefficient
        return FingerprintMethod.dice_coefficient(fingerprints_a, fingerprints_b)
