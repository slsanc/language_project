import hashlib

class ComparisonUtil:
    @staticmethod
    def hash_words(words_arr):
        """
        Hash each word in the given list using MD5, and return the resulting
        hashes.

        This lets us convert the text from a list of strings to a list
        of integers. This, in turn, makes the data easier for the computer
        to process.

        Return:
            array of string:
                A list of hash values corresponding to the given list of words
        """
        return [int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16) for word in words_arr]

    @staticmethod
    def clean_text(text):
        """
        Remove all characters, except alphanumeric characters, spaces, and
        linebreaks from the text. Then, Convert the text to lowercase.

        This cleanup makes it easier to analyze the text. For instance, it
        allows the computer to recognize two instances of a word as the
        same, even if one is capitalized (i.e. "You're" and "you're" are the
        same word)

        Args:
            text (str): The input text to be cleaned.

        Returns:
            str: The text, cleaned as described above.
        """
        # Keep only alphanumeric characters, spaces, and newlines.
        cleaned_text = re.sub(r"[^a-zA-Z0-9'\s\n]", "", text)

        # Convert the cleaned text to lowercase
        cleaned_text = cleaned_text.lower()

        return cleaned_text