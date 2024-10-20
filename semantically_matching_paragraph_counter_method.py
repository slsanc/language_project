import hashlib
from collections import Counter
import re
from nltk.corpus import wordnet as wn

from comparison_util import ComparisonUtil


class SmpcMethod:
    function_words = None  # Class-level constant for the common words wordlist
    core_vocab_words = None  # Class-level constant for the medium-frequency wordlist

    @classmethod
    def load_wordlists(cls, function_words_path, core_vocab_path):
        """Load the list of common words as a class-level constant"""
        if cls.function_words is None:  # Only load it once
            with open(function_words_path, 'r') as f:
                cls.function_words = set(word.strip().lower() for word in f.readlines())

        if cls.core_vocab_words is None:  # Only load it once
            with open(core_vocab_path, 'r') as f:
                cls.core_vocab_words = set(word.strip().lower() for word in f.readlines())

    @staticmethod
    def remove_function_words(paragraphs):
        """
        Remove the most common English words from an array of arrays of words.

        Args:
            paragraphs (list of list of str): An array of arrays, where each sub-array contains
            words from a paragraph.

        Returns:
            list of list of str: The array of arrays with common words removed from each paragraph.
        """
        return [
            [word for word in paragraph if word.lower() not in SmpcMethod.function_words]
            for paragraph in paragraphs
        ]

    @staticmethod
    def replace_core_vocab_with_synonyms(paragraphs):
        """
        Replace words in the core vocabulary (medium-frequency words) with their
        most common synonym, preserving the structure of the input paragraphs.

        Args:
            paragraphs (list of list of str): A list of paragraphs, where each paragraph
            is represented as a list of words.

        Returns:
            list of list of str: A list of paragraphs with core vocabulary words
            replaced by their most common synonym.
        """
        synonym_replaced_paragraphs = []

        for paragraph in paragraphs:
            synonym_replaced = []
            for word in paragraph:
                word_lower = word.lower()

                # Check if the word is in the core vocabulary (medium-frequency words)
                if word_lower in SmpcMethod.core_vocab_words:
                    # Replace with the most common synonym
                    synsets = wn.synsets(word_lower)
                    if synsets:
                        most_common_synonym = synsets[0].lemmas()[0].name()  # Get the most common synonym
                        synonym_replaced.append(most_common_synonym)
                    else:
                        synonym_replaced.append(word)  # Keep the word if no synonym is found
                else:
                    synonym_replaced.append(word)  # Keep non-core vocabulary words unchanged

            # Append the modified paragraph to the final result
            synonym_replaced_paragraphs.append(synonym_replaced)

        return synonym_replaced_paragraphs

    @staticmethod
    def text_to_paragraphs(text):
        """
        Convert the input text into an array of arrays, where each array contains
        the words of a paragraph. Any occurrence of one or more consecutive newlines
        is treated as a paragraph separator. Multiple newlines will not result in
        empty arrays.

        Args:
            text (str): The input text to be converted into paragraphs.

        Returns:
            list of list of str: A list where each element is a list of words
            representing a paragraph.
        """
        # Use regex to split the text into paragraphs, by one or more consecutive
        # newlines. This gives us an array of strings, each string being a
        # paragraph.
        paragraphs = re.split(r'\n+', text)

        # Convert each paragraph into a list of words, ignoring empty paragraphs.
        # This gives an array of arrays of strings - each string being a word.
        paragraph_arrays = [paragraph.split() for paragraph in paragraphs if paragraph.strip()]

        return paragraph_arrays

    @staticmethod
    def paragraph_to_ints(paragraph):
        """Convert each word in the paragraph to an integer"""
        return ComparisonUtil.hash_words()

    @staticmethod
    def compare_paragraphs(para1, para2):
        """Compare the word frequencies between two paragraphs"""
        freq1 = Counter(para1)
        freq2 = Counter(para2)
        shared_words = freq1 & freq2
        similarity_score = sum(shared_words.values())  # Number of shared words
        return similarity_score

    @staticmethod
    def most_frequent_words(paragraphs, top_n=10):
        """
        Find and return a list of the top N most frequent words the given input.

        Args:
            paragraphs (list of list of str): A list of paragraphs, where each paragraph
            is represented as a list of words.
            top_n (int): The number of most-frequent words to return.

        Returns:
            list of str: The N most frequent words across all paragraphs.
        """
        # Flatten the list of paragraphs into a single list of words
        all_words = [word.lower() for paragraph in paragraphs for word in paragraph]

        # Count the frequency of each word across the entire text
        word_counts = Counter(all_words)

        # Return the top N most frequent words
        return [word for word, _ in word_counts.most_common(top_n)]

    @staticmethod
    def compare_texts(text1, text2):
        """
        Main function to compare two texts using the method steps
        """
        # Step 1: "Clean" the text, converting it to lowercase and removing punctuation.
        text1 = ComparisonUtil.clean_text(text1)
        text2 = ComparisonUtil.clean_text(text2)

        # Step 2: Split the text into an array of arrays of strings.
        paragraphs1 = SmpcMethod.text_to_paragraphs(text1)
        paragraphs2 = SmpcMethod.text_to_paragraphs(text2)

        # Step 2: Remove common words; they don't add much to a text's meaning.
        paragraphs1 = SmpcMethod.remove_function_words(paragraphs1)
        paragraphs2 = SmpcMethod.remove_function_words(paragraphs2)

        # Step 3: Replace medium-frequency words with synonyms
        paragraphs1 = SmpcMethod.replace_core_vocab_with_synonyms(paragraphs1)
        paragraphs2 = SmpcMethod.replace_core_vocab_with_synonyms(paragraphs2)

        # Step 5: Initial large-scale check (compare most frequent words) across the whole of both texts
        most_freq_words_1 = SmpcMethod.most_frequent_words(paragraphs1)
        most_freq_words_2 = SmpcMethod.most_frequent_words(paragraphs2)

        if len(set(most_freq_words_1).intersection(most_freq_words_2)) < 3:
            return 0  # Not similar if fewer than 3 common frequent words

        # Step 6: Compare paragraphs and count matching pairs

        # Start by finding the most common words in each paragraph of both texts.
        most_freq_words_by_paragraph_1 = [SmpcMethod.most_frequent_words([paragraph]) for paragraph in paragraphs1]
        most_freq_words_by_paragraph_2 = [SmpcMethod.most_frequent_words([paragraph]) for paragraph in paragraphs2]

        # Compare the most frequent words in paragraph A with the most common words in paragraph B.
        # If they share at least 3 words on their top ten words, then the two paragraphs are said
        # to be a "matching pair".
        matching_pairs = 0

        for para1_freq_words in most_freq_words_by_paragraph_1:
            for para2_freq_words in most_freq_words_by_paragraph_2:
                if len(set(para1_freq_words).intersection(para2_freq_words)) > 2:
                    matching_pairs += 1

        return matching_pairs  # The final similarity score
