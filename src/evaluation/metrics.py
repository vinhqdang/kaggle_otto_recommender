import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional
import logging

logger = logging.getLogger(__name__)

class OTTOEvaluator:
    """Evaluator for OTTO Multi-Objective Recommender System following the competition metric."""
    
    def __init__(self):
        # Weights for different action types as per OTTO competition
        self.type_weights = {
            'clicks': 0.10,
            'carts': 0.30, 
            'orders': 0.60
        }
    
    def recall_at_k(self, predictions: List[int], ground_truth: List[int], k: int = 20) -> float:
        """Calculate Recall@K for a single session."""
        if not ground_truth:
            return 0.0
        
        # Take top k predictions
        pred_k = predictions[:k] if len(predictions) >= k else predictions
        
        # Calculate intersection
        intersect = set(pred_k) & set(ground_truth)
        
        # Recall@K = |intersection| / min(k, |ground_truth|)
        recall = len(intersect) / min(k, len(ground_truth))
        
        return recall
    
    def calculate_session_recall(self, predictions: Dict[str, List[int]], 
                                ground_truth: Dict[str, List[int]], 
                                session_id: int, k: int = 20) -> Dict[str, float]:
        """Calculate recall for different action types for a single session."""
        recalls = {}
        
        for action_type in ['clicks', 'carts', 'orders']:
            pred_aids = predictions.get(action_type, [])
            true_aids = ground_truth.get(action_type, [])
            
            recalls[action_type] = self.recall_at_k(pred_aids, true_aids, k)
        
        return recalls
    
    def calculate_weighted_recall(self, recalls: Dict[str, float]) -> float:
        """Calculate weighted recall score as per OTTO competition."""
        weighted_score = 0.0
        
        for action_type, weight in self.type_weights.items():
            weighted_score += weight * recalls.get(action_type, 0.0)
        
        return weighted_score
    
    def evaluate_predictions(self, predictions: Dict[int, Dict[str, List[int]]], 
                           ground_truth: Dict[int, Dict[str, List[int]]], 
                           k: int = 20) -> Dict[str, float]:
        """
        Evaluate predictions against ground truth.
        
        Args:
            predictions: Dict mapping session_id -> {action_type: [predicted_aids]}
            ground_truth: Dict mapping session_id -> {action_type: [true_aids]}
            k: Number of recommendations to consider
            
        Returns:
            Dictionary with evaluation metrics
        """
        all_recalls = {'clicks': [], 'carts': [], 'orders': []}
        weighted_scores = []
        
        # Evaluate each session
        for session_id in ground_truth.keys():
            if session_id not in predictions:
                # If no predictions for this session, recall is 0
                session_recalls = {'clicks': 0.0, 'carts': 0.0, 'orders': 0.0}
            else:
                session_recalls = self.calculate_session_recall(
                    predictions[session_id], ground_truth[session_id], session_id, k
                )
            
            # Store individual type recalls
            for action_type in ['clicks', 'carts', 'orders']:
                all_recalls[action_type].append(session_recalls[action_type])
            
            # Calculate weighted score for this session
            weighted_score = self.calculate_weighted_recall(session_recalls)
            weighted_scores.append(weighted_score)
        
        # Calculate mean metrics
        results = {
            'recall_clicks': np.mean(all_recalls['clicks']),
            'recall_carts': np.mean(all_recalls['carts']),
            'recall_orders': np.mean(all_recalls['orders']),
            'weighted_recall': np.mean(weighted_scores),
            'num_sessions': len(ground_truth)
        }
        
        return results
    
    def create_ground_truth_from_events(self, events_df: pd.DataFrame, 
                                       split_timestamp: Optional[int] = None) -> Dict[int, Dict[str, List[int]]]:
        """
        Create ground truth dict from events DataFrame.
        
        Args:
            events_df: DataFrame with columns ['session', 'aid', 'type', 'ts']
            split_timestamp: If provided, only consider events after this timestamp as ground truth
            
        Returns:
            Dict mapping session_id -> {action_type: [aids]}
        """
        if split_timestamp is not None:
            # Only consider events after split timestamp as ground truth
            gt_events = events_df[events_df['ts'] > split_timestamp].copy()
        else:
            gt_events = events_df.copy()
        
        ground_truth = {}
        
        for session_id, session_events in gt_events.groupby('session'):
            session_gt = {'clicks': [], 'carts': [], 'orders': []}
            
            for action_type in ['clicks', 'carts', 'orders']:
                type_events = session_events[session_events['type'] == action_type]
                aids = type_events['aid'].unique().tolist()
                session_gt[action_type] = aids
            
            ground_truth[session_id] = session_gt
        
        return ground_truth
    
    def create_prediction_format(self, session_predictions: Dict[int, List[int]]) -> Dict[int, Dict[str, List[int]]]:
        """
        Convert simple session->aids predictions to multi-objective format.
        For simple algorithms that don't distinguish between action types.
        """
        predictions = {}
        
        for session_id, predicted_aids in session_predictions.items():
            # Use same predictions for all action types
            predictions[session_id] = {
                'clicks': predicted_aids[:20],
                'carts': predicted_aids[:20], 
                'orders': predicted_aids[:20]
            }
        
        return predictions
    
    def evaluate_simple_predictions(self, predictions: Dict[int, List[int]], 
                                   test_events: pd.DataFrame, 
                                   split_ratio: float = 0.8) -> Dict[str, float]:
        """
        Evaluate simple predictions (single list per session) against test data.
        
        Args:
            predictions: Dict mapping session_id -> [predicted_aids]
            test_events: Test events DataFrame
            split_ratio: Ratio to split session for prediction vs ground truth
            
        Returns:
            Evaluation metrics
        """
        # Split each test session for evaluation
        ground_truth = {}
        
        for session_id, session_events in test_events.groupby('session'):
            session_events = session_events.sort_values('ts')
            split_idx = int(len(session_events) * split_ratio)
            
            if split_idx >= len(session_events) - 1:
                continue  # Skip sessions too short to split
            
            # Ground truth is events after split
            gt_events = session_events.iloc[split_idx:]
            
            session_gt = {'clicks': [], 'carts': [], 'orders': []}
            for action_type in ['clicks', 'carts', 'orders']:
                type_events = gt_events[gt_events['type'] == action_type]
                aids = type_events['aid'].unique().tolist()
                session_gt[action_type] = aids
            
            ground_truth[session_id] = session_gt
        
        # Convert predictions to multi-objective format
        multi_predictions = self.create_prediction_format(predictions)
        
        # Evaluate
        return self.evaluate_predictions(multi_predictions, ground_truth)
    
    def print_evaluation_results(self, results: Dict[str, float], model_name: str = "Model"):
        """Print formatted evaluation results."""
        print(f"\n{model_name} Evaluation Results:")
        print("=" * 50)
        print(f"Recall@20 - Clicks:  {results['recall_clicks']:.4f}")
        print(f"Recall@20 - Carts:   {results['recall_carts']:.4f}")
        print(f"Recall@20 - Orders:  {results['recall_orders']:.4f}")
        print(f"Weighted Recall:     {results['weighted_recall']:.4f}")
        print(f"Sessions evaluated:  {results['num_sessions']}")
        print("=" * 50)