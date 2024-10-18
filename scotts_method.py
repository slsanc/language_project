import hashlib
from collections import Counter
import nltk
from nltk.corpus import wordnet as wn

class ScottsMethod:
    def __init__(self, wordlist_path):
        # Load the common words (stop words)
        self.common_words = self.load_wordlist(wordlist_path)

    def load_wordlist(self, wordlist_path):
        """Load the list of common words"""
        with open(wordlist_path, 'r') as f:
            return set(word.strip().lower() for word in f.readlines())

    def remove_common_words(self, text):
        """Remove common words from the text"""
        words = text.split()
        return [word for word in words if word.lower() not in self.common_words]

    def replace_with_synonyms(self, text):
        """Replace medium-frequency words with their most common synonym"""
        words = text.split()
        synonym_replaced = []
        for word in words:
            synsets = wn.synsets(word)
            if synsets:
                most_common_synonym = synsets[0].lemmas()[0].name()
                synonym_replaced.append(most_common_synonym)
            else:
                synonym_replaced.append(word)
        return synonym_replaced

    def hash_word(self, word):
        """Convert a word to a hashed integer for easier computation"""
        return int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)

    def text_to_paragraphs(self, text):
        """Split the text into paragraphs"""
        return text.split("\n\n")

    def paragraph_to_int(self, paragraph):
        """Convert each word in the paragraph to an integer"""
        words = paragraph.split()
        return [self.hash_word(word) for word in words]

    def compare_paragraphs(self, para1, para2):
        """Compare the word frequencies between two paragraphs"""
        freq1 = Counter(para1)
        freq2 = Counter(para2)
        shared_words = freq1 & freq2
        similarity_score = sum(shared_words.values())  # Number of shared words
        return similarity_score

    def most_frequent_words(self, text, top_n=10):
        """Return the top N most frequent words in the text"""
        words = text.split()
        word_counts = Counter(words)
        return [word for word, _ in word_counts.most_common(top_n)]

    def compare_texts(self, text1, text2):
        """Main function to compare two texts using the method steps"""
        # Step 1: Remove common words
        text1_filtered = self.remove_common_words(text1)
        text2_filtered = self.remove_common_words(text2)

        # Step 2: Replace medium-frequency words with synonyms
        text1_synonym_replaced = self.replace_with_synonyms(' '.join(text1_filtered))
        text2_synonym_replaced = self.replace_with_synonyms(' '.join(text2_filtered))

        # Step 3: Split into paragraphs
        paragraphs1 = self.text_to_paragraphs(' '.join(text1_synonym_replaced))
        paragraphs2 = self.text_to_paragraphs(' '.join(text2_synonym_replaced))

        # Step 4: Convert paragraphs into arrays of integers
        para1_int_arrays = [self.paragraph_to_int(p) for p in paragraphs1]
        para2_int_arrays = [self.paragraph_to_int(p) for p in paragraphs2]

        # Step 5: Initial large-scale check (compare most frequent words)
        most_freq_words_1 = self.most_frequent_words(' '.join(text1_filtered))
        most_freq_words_2 = self.most_frequent_words(' '.join(text2_filtered))

        if len(set(most_freq_words_1).intersection(most_freq_words_2)) < 3:
            return 0  # Not similar if fewer than 3 common frequent words

        # Step 6: Compare paragraphs and count matching pairs
        matching_pairs = 0
        for para1 in para1_int_arrays:
            for para2 in para2_int_arrays:
                similarity_score = self.compare_paragraphs(para1, para2)
                if similarity_score > 0:  # You can set a threshold if necessary
                    matching_pairs += 1

        return matching_pairs  # The final similarity score

