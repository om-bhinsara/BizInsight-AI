"""
Sentiment Filter Utility
Analyzes raw review text and extracts negative reviews using lexicon-based sentiment analysis
"""

import pandas as pd
from typing import Tuple, Dict, Optional

def filter_negative_reviews(df: pd.DataFrame, sentiment_column: str = "sentiment") -> Dict:
    """
    Filter dataframe to keep only negative reviews
    
    Args:
        df: DataFrame with review and sentiment columns
        sentiment_column: Name of sentiment column (default: "sentiment")

    Returns:
        Dictionary containing:
        - negative_reviews: List of negative review texts
        - negative_indices: Original indices of negative reviews
        - negative_count: Number of negative reviews
        - positive_count: Number of positive reviews
        - neutral_count: Number of neutral reviews
        - total_count: Total reviews
        - success: Boolean indicating if enough data
        - message: Status message    
    """

