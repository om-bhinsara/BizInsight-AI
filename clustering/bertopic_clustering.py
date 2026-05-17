"""
BERTopic Clustering Module
Replaces HDBSCAN + custom analysis with robust BERTopic pipeline.
"""

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
import streamlit as st

from clustering.preprocess import preprocess_reviews
from clustering.vectorize import load_model

def run_bertopic_pipeline(
    reviews: List[str],
    embedding_model: Optional[SentenceTransformer] = None,
    min_topic_size: int = 8,
    verbose: bool = True
) -> Dict:
    """
    Main clustering pipeline using BERTopic.
    
    Args:
        reviews: list of raw review strings
        embedding_model: SentenceTransformer model (if None, loads default)
        min_topic_size: minimum size of a topic (controls granularity)
        verbose: print progress
    
    Returns:
        dict with clustering results, topics, visualisations, etc.
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

    # ---------- 1. Preprocess ----------
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

    # ---------- 2. Load or use embedding model ----------
    if embedding_model is None:
        embedding_model = load_model()

    # ---------- 3. Train BERTopic ----------
    if verbose:
        st.write("🔍 Training BERTopic model...")
    
    vectorizer_model = CountVectorizer(
        stop_words="english",           # built-in English stopwords
        ngram_range=(1, 2),            # include bigrams for better phrases
        max_df=0.8,                     # ignore very common words
        min_df=2                        # ignore words that appear only once
    )
    
    topic_model = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=min_topic_size,
        vectorizer_model=vectorizer_model,
        verbose=verbose,
        calculate_probabilities=True
    )
    
    topics, probs = topic_model.fit_transform(cleaned_reviews)
    
    # ---------- 4. Get topic info ----------
    topic_info = topic_model.get_topic_info()
    
    # Filter out noise topic (-1)
    valid_topics = topic_info[topic_info.Topic != -1]
    n_topics = len(valid_topics)
    total_reviews = len(cleaned_reviews)
    noise_count = topic_info[topic_info.Topic == -1]['Count'].values[0] if -1 in topic_info.Topic.values else 0
    noise_percentage = (noise_count / total_reviews) * 100 if total_reviews > 0 else 0
    
    # ---------- 5. Prepare clusters in the format expected by app.py ----------
    clusters = []
    for _, row in valid_topics.iterrows():
        topic_id = row['Topic']
        size = row['Count']
        
        # Get representative reviews for this topic
        # topic_model.get_representative_docs() requires bertopic>=0.15.0
        try:
            representative_docs = topic_model.get_representative_docs(topic_id)
            examples = representative_docs[:3] if representative_docs else []
        except AttributeError:
            # Fallback: get random sample from documents in this topic
            indices = [i for i, t in enumerate(topics) if t == topic_id]
            examples = [cleaned_reviews[i] for i in indices[:3]]
        
        # Get topic words (top 5)
        topic_words = topic_model.get_topic(topic_id)
        if topic_words:
            topic_name = ", ".join([word for word, _ in topic_words[:3]])
        else:
            topic_name = f"Topic {topic_id}"
        
        clusters.append({
            "id": int(topic_id),
            "name": topic_name,
            "count": size,
            "percentage": (size / total_reviews) * 100,
            "example_reviews": examples,
            "sample_review": examples[0] if examples else "No sample",
            "action": f"Investigate: {topic_name}"
        })
    
    # Sort by count descending
    clusters.sort(key=lambda x: x["count"], reverse=True)
        
    # ---------- 6. Silhouette score (on reduced embeddings) ----------
    try:
        # Get reduced embeddings from BERTopic (if available)
        reduced_embeddings = topic_model.umap_model.embedding_ if hasattr(topic_model, 'umap_model') else None
        if reduced_embeddings is not None and n_topics >= 2:
            # Filter out noise for silhouette
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
    
    # ---------- 7. Prepare final result ----------
    # Get UMAP reduced embeddings for 2D plot (if needed by app.py)
    try:
        reduced_2d = topic_model.umap_model.embedding_
    except AttributeError:
        reduced_2d = None
    
    return {
        "success": True,
        "message": f"Found {n_topics} complaint topics with {noise_percentage:.1f}% noise",
        "total_negative_reviews": total_reviews,
        "n_clusters": n_topics,  # for compatibility with app.py variable names
        "n_topics": n_topics,
        "reduced_embeddings": reduced_2d.tolist() if reduced_2d is not None else [],
        "labels": topics.tolist(),
        "noise_count": noise_count,
        "noise_percentage": noise_percentage,
        "silhouette_score": round(sil_score, 4) if sil_score else None,
        "clusters": clusters,
        "topic_model": topic_model,  # optional for later use
        "cleaned_reviews": cleaned_reviews
    }

def merge_clusters_until_limit(
    topic_model,
    topics,
    cleaned_reviews,
    clusters,
    max_clusters=10
):
    """
    Merge the smallest clusters into the most similar larger cluster
    until the number of clusters <= max_clusters.
    Returns updated clusters and topics array.
    """
    if len(clusters) <= max_clusters:
        return clusters, topics
    
    # Get cluster centroids from UMAP space (if available)
    try:
        reduced_embeds = topic_model.umap_model.embedding_
    except AttributeError:
        # Fallback: use embeddings from the topic model
        reduced_embeds = topic_model.c_tf_idf_.toarray()  # may be large
    
    # Build mapping: cluster_id -> centroid, size, examples
    cluster_info = {}
    for c in clusters:
        cid = c['id']
        # Get all indices for this cluster
        indices = [i for i, t in enumerate(topics) if t == cid]
        if indices:
            centroid = np.mean(reduced_embeds[indices], axis=0)
            cluster_info[cid] = {
                'centroid': centroid,
                'size': c['count'],
                'examples': c['example_reviews'],
                'name': c['name']
            }
    
    # Merge loop
    while len(cluster_info) > max_clusters:
        # Find smallest cluster
        smallest_id = min(cluster_info.keys(), key=lambda x: cluster_info[x]['size'])
        smallest = cluster_info.pop(smallest_id)
        
        # Find most similar cluster (by centroid cosine similarity)
        best_sim = -1
        best_id = None
        for cid, info in cluster_info.items():
            sim = np.dot(smallest['centroid'], info['centroid']) / (
                np.linalg.norm(smallest['centroid']) * np.linalg.norm(info['centroid']) + 1e-9
            )
            if sim > best_sim:
                best_sim = sim
                best_id = cid
        
        # Merge smallest into best
        cluster_info[best_id]['size'] += smallest['size']
        # Merge examples (keep top 3 from both, but simple concatenation)
        cluster_info[best_id]['examples'] = (cluster_info[best_id]['examples'] + smallest['examples'])[:3]
        # Update name (use the larger cluster's name, or combine)
        cluster_info[best_id]['name'] = cluster_info[best_id]['name']  # keep larger name
        
        # Also update the topics array: reassign all smallest cluster topics to best_id
        for i in range(len(topics)):
            if topics[i] == smallest_id:
                topics[i] = best_id
    
    # Rebuild clusters list
    new_clusters = []
    for cid, info in cluster_info.items():
        new_clusters.append({
            "id": int(cid),
            "name": info['name'],
            "count": info['size'],
            "percentage": (info['size'] / len(cleaned_reviews)) * 100,
            "example_reviews": info['examples'],
            "sample_review": info['examples'][0] if info['examples'] else "No sample",
            "action": f"Investigate: {info['name']}"
        })
    
    new_clusters.sort(key=lambda x: x["count"], reverse=True)
    return new_clusters, topics

def make_business_name(example_reviews: List[str]) -> str:
    """
    Generate short business name from example reviews.
    Priority:
      1. Keyword matching (payment, login, delivery, technical, etc.)
      2. Most frequent meaningful word (length > 3, not in common stopwords)
      3. Fallback: first 3 words of first review
    """
    # Combine examples
    text = " ".join(example_reviews).lower()
    
    # Keyword mapping (extend as needed)
    keywords = {
        "payment": ["payment", "charge", "billing", "invoice", "refund", "duplicate", "credit card"],
        "login": ["login", "password", "access", "account", "verify", "authentication"],
        "delivery": ["delivery", "shipping", "courier", "package", "tracking", "arrived", "delay"],
        "technical": ["crash", "error", "timeout", "server", "app", "freeze", "bug"],
        "product": ["product", "item", "toy", "razor", "diaper", "battery", "collar", "shampoo"]
    }
    
    # Score each category
    scores = {}
    for cat, words in keywords.items():
        score = sum(text.count(w) for w in words)
        if score > 0:
            scores[cat] = score
    
    if scores:
        best_cat = max(scores, key=scores.get)
        return f"{best_cat.capitalize()} Issues"
    
    # Fallback: most frequent content word (len>3, not common stopword)
    from collections import Counter
    words = [w for w in text.split() if len(w) > 3 and w not in {'this', 'that', 'with', 'from', 'have', 'were'}]
    if words:
        common = Counter(words).most_common(1)[0][0]
        return common.capitalize() + " Issues"
    
    # Ultimate fallback: first few words of first example
    return " ".join(example_reviews[0].split()[:3]).capitalize()