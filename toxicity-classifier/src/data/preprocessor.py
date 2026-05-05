import re
import unicodedata

class TextPreprocessor:
    def __init__(self):
        pass

    def clean_text(self, text, lang="en"):
        """
        Cleans input text: removes URLs, HTML tags, normalizes whitespace.
        Preserves original casing for Hindi/Hinglish.
        """
        if not isinstance(text, str):
            return ""

        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Normalize Unicode (handles Devanagari edges)
        text = unicodedata.normalize('NFC', text)
        
        # Strip excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Lowercase only for purely English texts
        if lang == "en":
            text = text.lower()
            
        return text

    def clean_series(self, series, lang_series):
        """
        Apply cleaning over a Pandas Series efficiently.
        """
        return series.combine(lang_series, lambda txt, lang: self.clean_text(txt, lang))
