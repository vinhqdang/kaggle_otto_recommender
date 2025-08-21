import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import logging
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)

class PopularityRecommender:
    """Simple popularity-based recommender as baseline."""
    
    def __init__(self, top_k: int = 20):
        self.top_k = top_k
        self.global_popularity = None
        self.type_popularity = None
        
    def fit(self, train_events: pd.DataFrame):
        """Fit the popularity model."""
        logger.info("Training popularity recommender...")
        
        # Global popularity (weighted by interaction type)
        weights = {'clicks': 1, 'carts': 3, 'orders': 5}
        train_events['weight'] = train_events['type'].map(weights)
        
        self.global_popularity = (
            train_events.groupby('aid')['weight']
            .sum()
            .sort_values(ascending=False)
            .index.tolist()
        )
        
        # Type-specific popularity
        self.type_popularity = {}
        for interaction_type in ['clicks', 'carts', 'orders']:
            type_events = train_events[train_events['type'] == interaction_type]
            popularity = (
                type_events.groupby('aid')
                .size()
                .sort_values(ascending=False)
                .index.tolist()
            )
            self.type_popularity[interaction_type] = popularity
        
        logger.info(f"Popularity model trained with {len(self.global_popularity)} items")
    
    def predict(self, session_events: pd.DataFrame) -> List[int]:
        """Predict next items for a session."""
        if self.global_popularity is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Get items already seen in session
        seen_items = set(session_events['aid'].unique())
        
        # Recommend popular items not yet seen
        recommendations = []
        for item in self.global_popularity:
            if item not in seen_items:
                recommendations.append(item)
            if len(recommendations) >= self.top_k:
                break
        
        # Fill with global popular items if needed
        while len(recommendations) < self.top_k and len(recommendations) < len(self.global_popularity):
            for item in self.global_popularity:
                if item not in recommendations:
                    recommendations.append(item)
                if len(recommendations) >= self.top_k:
                    break
        
        return recommendations
    
    def predict_batch(self, test_events: pd.DataFrame) -> Dict[int, List[int]]:
        """Predict for multiple sessions."""
        predictions = {}
        
        for session_id, session_events in test_events.groupby('session'):
            predictions[session_id] = self.predict(session_events)
        
        return predictions

class SessionBasedRecommender:
    """Session-based collaborative filtering recommender inspired by SAR."""
    
    def __init__(self, top_k: int = 20, similarity_threshold: float = 0.1):
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.item_cooccurrence = None
        self.item_popularity = None
        
    def fit(self, train_events: pd.DataFrame):
        """Fit the session-based model."""
        logger.info("Training session-based recommender...")
        
        # Calculate item co-occurrence in sessions
        cooccurrence = defaultdict(lambda: defaultdict(int))
        
        # Weight interactions by type
        weights = {'clicks': 1, 'carts': 3, 'orders': 5}
        train_events['weight'] = train_events['type'].map(weights)
        
        session_items = {}
        for session_id, group in train_events.groupby('session'):
            # Get weighted item interactions in session
            session_weights = group.groupby('aid')['weight'].sum()
            session_items[session_id] = session_weights.to_dict()
        
        # Calculate co-occurrence matrix
        for session_id, items in session_items.items():
            item_list = list(items.keys())
            for i, item_a in enumerate(item_list):
                for item_b in item_list[i+1:]:
                    # Weight by both items' weights in session
                    weight = min(items[item_a], items[item_b])
                    cooccurrence[item_a][item_b] += weight
                    cooccurrence[item_b][item_a] += weight
        
        self.item_cooccurrence = dict(cooccurrence)
        
        # Calculate item popularity
        self.item_popularity = (
            train_events.groupby('aid')['weight']
            .sum()
            .sort_values(ascending=False)
            .to_dict()
        )
        
        logger.info(f"Session-based model trained with {len(self.item_cooccurrence)} items")
    
    def predict(self, session_events: pd.DataFrame) -> List[int]:
        """Predict next items for a session."""
        if self.item_cooccurrence is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Get items in current session with weights
        weights = {'clicks': 1, 'carts': 3, 'orders': 5}
        session_events['weight'] = session_events['type'].map(weights)
        session_items = session_events.groupby('aid')['weight'].sum().to_dict()
        
        # Calculate scores for candidate items
        candidate_scores = defaultdict(float)
        
        for session_item, session_weight in session_items.items():
            if session_item in self.item_cooccurrence:
                for candidate_item, cooccur_score in self.item_cooccurrence[session_item].items():
                    if candidate_item not in session_items:  # Don't recommend seen items
                        candidate_scores[candidate_item] += session_weight * cooccur_score
        
        # Sort candidates by score
        ranked_candidates = sorted(
            candidate_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Get top recommendations
        recommendations = [item for item, score in ranked_candidates[:self.top_k]]
        
        # Fill with popular items if needed
        popular_items = sorted(
            self.item_popularity.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for item, _ in popular_items:
            if item not in session_items and item not in recommendations:
                recommendations.append(item)
            if len(recommendations) >= self.top_k:
                break
        
        return recommendations
    
    def predict_batch(self, test_events: pd.DataFrame) -> Dict[int, List[int]]:
        """Predict for multiple sessions."""
        predictions = {}
        
        for session_id, session_events in test_events.groupby('session'):
            predictions[session_id] = self.predict(session_events)
        
        return predictions

class HybridRecommender:
    """Hybrid recommender combining popularity and session-based approaches."""
    
    def __init__(self, top_k: int = 20, popularity_weight: float = 0.3):
        self.top_k = top_k
        self.popularity_weight = popularity_weight
        self.session_weight = 1.0 - popularity_weight
        
        self.popularity_model = PopularityRecommender(top_k)
        self.session_model = SessionBasedRecommender(top_k)
        
    def fit(self, train_events: pd.DataFrame):
        """Fit both models."""
        logger.info("Training hybrid recommender...")
        
        self.popularity_model.fit(train_events)
        self.session_model.fit(train_events)
        
        logger.info("Hybrid model training completed")
    
    def predict(self, session_events: pd.DataFrame) -> List[int]:
        """Predict using hybrid approach."""
        # Get predictions from both models
        pop_recs = self.popularity_model.predict(session_events)
        session_recs = self.session_model.predict(session_events)
        
        # Combine with weights
        seen_items = set(session_events['aid'].unique())
        combined_scores = defaultdict(float)
        
        # Add popularity scores
        for i, item in enumerate(pop_recs):
            if item not in seen_items:
                score = self.popularity_weight * (len(pop_recs) - i) / len(pop_recs)
                combined_scores[item] += score
        
        # Add session-based scores
        for i, item in enumerate(session_recs):
            if item not in seen_items:
                score = self.session_weight * (len(session_recs) - i) / len(session_recs)
                combined_scores[item] += score
        
        # Sort by combined score
        ranked_items = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        recommendations = [item for item, score in ranked_items[:self.top_k]]
        
        return recommendations
    
    def predict_batch(self, test_events: pd.DataFrame) -> Dict[int, List[int]]:
        """Predict for multiple sessions."""
        predictions = {}
        
        for session_id, session_events in test_events.groupby('session'):
            predictions[session_id] = self.predict(session_events)
        
        return predictions