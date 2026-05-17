"""
Vectorize Module
Converts cleaned text reviews into numerical embeddings using Sentence Transformers
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Optional
import streamlit as st

# Global variable to cache the model (load once)
_model = None

@st.cache_resource
def load_model(model_name: str = "all-mpnet-base-v2"):
    """
    Load the sentence transformer model.
    Model is cached after first load for performance.
    """
    global _model
    
    if _model is None:
        with st.spinner("Loading AI model (first time only)..."):
            _model = SentenceTransformer(model_name)
    
    return _model

def get_embeddings(reviews: List[str], model: Optional[SentenceTransformer] = None) -> np.ndarray:
    """
    Convert list of reviews to vector embeddings.
    """
    if model is None:
        model = load_model()
    
    progress_bar = st.progress(0)
    embeddings = model.encode(reviews, show_progress_bar=False)
    progress_bar.progress(100)
    
    return embeddings