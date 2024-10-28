import re

class ComparisonUtil:
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
            text (str): The text to be cleaned.

        Returns:
            str: The text, cleaned as described above.
        """
        # Keep only alphanumeric characters, spaces, and newlines.
        cleaned_text = re.sub(r"[^a-zA-Z0-9'\s\n]", "", text)

        # Convert the cleaned text to lowercase
        cleaned_text = cleaned_text.lower()

        return cleaned_text