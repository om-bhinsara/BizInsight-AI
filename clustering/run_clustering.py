"""
Run Clustering Module - BERTopic with outlier reduction & unique business naming
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from bertopic import BERTopic
import hdbscan
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st
from collections import Counter

from clustering.preprocess import preprocess_reviews
from clustering.vectorize import load_model


def make_unique_business_name(topic_words: List[tuple], examples: List[str]) -> str:
    """
    Create a short, unique business name from topic words (top 2-3 words).
    Returns e.g. "Payment Gateway Failed" instead of generic "Payment Issues".
    """
    if not topic_words:
        return "Other Issues"
    
    # Extract top 2 meaningful words (ignore very common ones)
    words = [w for w, _ in topic_words[:2] if len(w) > 2]
    if not words:
        return "Other Issues"
    
    # Capitalize and join
    base_name = " ".join(words).title()
    
    # Add a suffix like "Issues" or "Problems" if not already there
    if not any(suffix in base_name for suffix in ["Issues", "Problems", "Error", "Failed", "Delay"]):
        base_name += " Issues"
    
    return base_name


def merge_duplicate_clusters(clusters: List[Dict]) -> List[Dict]:
    """
    Merge clusters that have exactly the same name (by summing counts and combining examples).
    """
    merged = {}
    for c in clusters:
        name = c["name"]
        if name not in merged:
            merged[name] = {
                "name": name,
                "count": 0,
                "examples": [],
                "sample_review": c["sample_review"],
                "action": c["action"]
            }
        merged[name]["count"] += c["count"]
        merged[name]["examples"].extend(c["example_reviews"][:2])
    
    # Rebuild list, truncate examples to 3 per cluster
    result = []
    for name, data in merged.items():
        result.append({
            "id": hash(name) % 10000,   # dummy id
            "name": name,
            "count": data["count"],
            "percentage": 0.0,  # will recalc later
            "example_reviews": data["examples"][:3],
            "sample_review": data["sample_review"],
            "action": data["action"]
        })
    return result


def run_pipeline(
    reviews: List[str],
    embedding_model: Optional[SentenceTransformer] = None,
    min_topic_size: int = 8,        # lower = fewer outliers
    verbose: bool = True
) -> Dict:
    """
    Main clustering pipeline using BERTopic with:
    - custom HDBSCAN to reduce noise (cluster_selection_epsilon)
    - unique naming via top topic words
    - duplicate name merging
    """
    if len(reviews) < 10:
        return {
            "success": False,
            "message": f"Only {len(reviews)} reviews. Need at least 10.",
            "clusters": [],
            "total_reviews": len(reviews),
            "n_topics": 0,
            "silhouette_score": None,
            "noise_count": len(reviews),
            "noise_percentage": 100.0
        }

    # 1. Preprocess
    if verbose:
        st.write("🧹 Cleaning reviews...")
    cleaned_reviews = preprocess_reviews(reviews)
    
    if len(cleaned_reviews) < 10:
        return {
            "success": False,
            "message": f"After cleaning, only {len(cleaned_reviews)} reviews left. Need at least 10.",
            "clusters": [],
            "total_reviews": len(reviews),
            "n_topics": 0,
            "silhouette_score": None,
            "noise_count": len(reviews),
            "noise_percentage": 100.0
        }

    if verbose:
        st.write("**Sample cleaned reviews (first 3):**")
        for i in range(min(3, len(cleaned_reviews))):
            st.write(f"{i+1}. {cleaned_reviews[i]}")

    # 2. Load embedding model
    if embedding_model is None:
        embedding_model = load_model()

    # 3. Train BERTopic with custom HDBSCAN to reduce noise
    if verbose:
        st.write("🔍 Training BERTopic model (optimised for lower noise)...")
    
    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.8,
        min_df=2
    )
    
    # Custom HDBSCAN with epsilon to merge nearby clusters and reduce noise
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=3,
        cluster_selection_epsilon=0.3,   # <-- merges small nearby clusters, reduces outliers
        metric='euclidean',
        prediction_data=True
    )
    
    topic_model = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=min_topic_size,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        verbose=verbose,
        calculate_probabilities=True
    )
    
    topics, probs = topic_model.fit_transform(cleaned_reviews)
    
    # 4. Get topic info
    topic_info = topic_model.get_topic_info()
    valid_topics = topic_info[topic_info.Topic != -1]
    n_topics = len(valid_topics)
    total_reviews = len(cleaned_reviews)
    noise_count = topic_info[topic_info.Topic == -1]['Count'].values[0] if -1 in topic_info.Topic.values else 0
    noise_percentage = (noise_count / total_reviews) * 100 if total_reviews > 0 else 0
    
    # 5. Prepare clusters with unique names from topic words
    clusters = []
    for _, row in valid_topics.iterrows():
        topic_id = row['Topic']
        size = row['Count']
        
        # Get representative reviews
        try:
            representative_docs = topic_model.get_representative_docs(topic_id)
            examples = representative_docs[:3] if representative_docs else []
        except AttributeError:
            indices = [i for i, t in enumerate(topics) if t == topic_id]
            examples = [cleaned_reviews[i] for i in indices[:3]]
        
        # Get topic words from BERTopic (already filtered by CountVectorizer)
        topic_words = topic_model.get_topic(topic_id)
        business_name = make_unique_business_name(topic_words, examples)
        
        clusters.append({
            "id": int(topic_id),
            "name": business_name,
            "count": size,
            "percentage": (size / total_reviews) * 100,
            "example_reviews": examples,
            "sample_review": examples[0] if examples else "No sample",
            "action": f"Investigate {business_name}"
        })
    
    # 6. Merge duplicate names (if two clusters get same name)
    clusters = merge_duplicate_clusters(clusters)
    # Recalculate percentages after merging
    for c in clusters:
        c["percentage"] = (c["count"] / total_reviews) * 100
    clusters.sort(key=lambda x: x["count"], reverse=True)
    n_topics = len(clusters)
    
    # 7. Silhouette score
    try:
        reduced_embeddings = topic_model.umap_model.embedding_ if hasattr(topic_model, 'umap_model') else None
        if reduced_embeddings is not None and n_topics >= 2:
            non_noise_mask = np.array(topics) != -1
            if np.sum(non_noise_mask) > n_topics:
                sil_score = silhouette_score(
                    reduced_embeddings[non_noise_mask], 
                    np.array(topics)[non_noise_mask]
                )
            else:
                sil_score = None
        else:
            sil_score = None
    except Exception:
        sil_score = None
    
    # 8. Prepare 2D embeddings for plot
    try:
        reduced_2d = topic_model.umap_model.embedding_
        if isinstance(reduced_2d, np.ndarray):
            reduced_2d_list = reduced_2d.tolist()
        else:
            reduced_2d_list = reduced_2d if isinstance(reduced_2d, list) else []
    except AttributeError:
        reduced_2d_list = []
    
    # --- FIX: topics is already a list, no .tolist() needed ---
    return {
        "success": True,
        "message": f"Found {n_topics} complaint topics with {noise_percentage:.1f}% noise",
        "total_negative_reviews": total_reviews,
        "n_clusters": n_topics,
        "n_topics": n_topics,
        "reduced_embeddings": reduced_2d_list,
        "labels": topics,                     # <-- FIXED: no .tolist()
        "noise_count": noise_count,
        "noise_percentage": noise_percentage,
        "silhouette_score": round(sil_score, 4) if sil_score else None,
        "clusters": clusters,
        "topic_model": topic_model,
        "cleaned_reviews": cleaned_reviews
    }