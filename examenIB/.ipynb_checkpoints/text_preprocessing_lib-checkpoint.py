"""
Text Preprocessing Library
===========================
A comprehensive library for text preprocessing including cleaning,
normalization, tokenization, stopword removal, and stemming.
"""

import re
from typing import List, Set, Tuple
from nltk.corpus import stopwords
from nltk.stem import porter


class TextPreprocessor:
    """
    A complete text preprocessing pipeline for NLP tasks.
    
    Attributes:
        stop_words (Set[str]): Set of stopwords for filtering
        stemmer: Porter stemmer for word stemming
    """
    
    def __init__(self, language: str = 'english'):
        """
        Initialize the preprocessor with stopwords and stemmer.
        
        Args:
            language: Language for stopwords (default: 'english')
        """
        self.stop_words = set(stopwords.words(language))
        self.stemmer = porter.PorterStemmer()
    
    def clean_text(self, text: str) -> str:
        """
        Remove HTML tags and special characters from text.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        # Remove HTML tags
        text = re.sub(pattern=r'<.*?>', repl='', string=text)
        
        # Remove special characters and punctuation
        text = text.replace('.', ' ')
        text = text.replace(',', '')
        text = text.replace('(', '')
        text = text.replace(')', '')
        text = text.replace('"', '')
        text = text.replace("'", '')
        text = text.replace('\x08', '')
        
        return text
    
    def normalize(self, text: str) -> str:
        """
        Convert text to lowercase for normalization.
        
        Args:
            text: Text string to normalize
            
        Returns:
            Normalized (lowercase) text
        """
        return text.lower()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Split text into individual tokens (words).
        
        Args:
            text: Text string to tokenize
            
        Returns:
            List of tokens
        """
        return text.split()
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of tokens without stopwords
        """
        return [word for word in tokens if word not in self.stop_words]
    
    def apply_stemming(self, tokens: List[str]) -> List[str]:
        """
        Apply Porter stemming to reduce words to their root form.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of stemmed tokens
        """
        return [self.stemmer.stem(token) for token in tokens]
    
    def preprocess(self, text: str, 
                   clean: bool = True,
                   normalize: bool = True,
                   remove_stops: bool = True,
                   stem: bool = True) -> List[str]:
        """
        Complete preprocessing pipeline.
        
        Args:
            text: Raw text string
            clean: Apply text cleaning
            normalize: Apply normalization (lowercase)
            remove_stops: Remove stopwords
            stem: Apply stemming
            
        Returns:
            List of preprocessed tokens
        """
        # Clean
        if clean:
            text = self.clean_text(text)
        
        # Normalize
        if normalize:
            text = self.normalize(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        if remove_stops:
            tokens = self.remove_stopwords(tokens)
        
        # Stemming
        if stem:
            tokens = self.apply_stemming(tokens)
        
        return tokens
    
    def get_vocabulary(self, token_lists: List[List[str]]) -> Set[str]:
        """
        Extract unique vocabulary from multiple token lists.
        
        Args:
            token_lists: List of token lists
            
        Returns:
            Set of unique tokens (vocabulary)
        """
        vocab = set()
        for tokens in token_lists:
            vocab.update(tokens)
        return vocab
    
    def compare_vocabularies(self, 
                            original_tokens: List[List[str]], 
                            processed_tokens: List[List[str]]) -> Tuple[int, int, int, float]:
        """
        Compare vocabulary size before and after preprocessing.
        
        Args:
            original_tokens: List of original token lists
            processed_tokens: List of preprocessed token lists
            
        Returns:
            Tuple of (original_size, processed_size, reduction, percentage)
        """
        vocab_original = self.get_vocabulary(original_tokens)
        vocab_processed = self.get_vocabulary(processed_tokens)
        
        original_size = len(vocab_original)
        processed_size = len(vocab_processed)
        reduction = original_size - processed_size
        percentage = (reduction / original_size) * 100 if original_size > 0 else 0
        
        return original_size, processed_size, reduction, percentage


# Convenience functions for direct usage
def clean_text(text: str) -> str:
    """Clean text by removing HTML tags and special characters."""
    preprocessor = TextPreprocessor()
    return preprocessor.clean_text(text)


def normalize_text(text: str) -> str:
    """Normalize text to lowercase."""
    preprocessor = TextPreprocessor()
    return preprocessor.normalize(text)


def tokenize_text(text: str) -> List[str]:
    """Tokenize text into words."""
    preprocessor = TextPreprocessor()
    return preprocessor.tokenize(text)


def remove_stopwords(tokens: List[str], language: str = 'english') -> List[str]:
    """Remove stopwords from token list."""
    preprocessor = TextPreprocessor(language=language)
    return preprocessor.remove_stopwords(tokens)


def stem_tokens(tokens: List[str]) -> List[str]:
    """Apply stemming to token list."""
    preprocessor = TextPreprocessor()
    return preprocessor.apply_stemming(tokens)


def preprocess_text(text: str, **kwargs) -> List[str]:
    """
    Preprocess text with configurable steps.
    
    Args:
        text: Raw text string
        **kwargs: Options for preprocessing steps (clean, normalize, remove_stops, stem)
        
    Returns:
        List of preprocessed tokens
    """
    preprocessor = TextPreprocessor()
    return preprocessor.preprocess(text, **kwargs)


