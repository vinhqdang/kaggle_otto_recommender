from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Any
import pickle
import os

class BaseRecommender(ABC):
    """Abstract base class for all recommender models."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_trained = False
    
    @abstractmethod
    def fit(self, train_events: pd.DataFrame, **kwargs):
        """Train the recommender model."""
        pass
    
    @abstractmethod
    def predict_batch(self, test_events: pd.DataFrame, **kwargs) -> Dict[int, List[int]]:
        """Predict recommendations for multiple sessions."""
        pass
    
    def predict_session(self, session_events: pd.DataFrame, **kwargs) -> List[int]:
        """Predict recommendations for a single session. Override if needed."""
        # Default implementation using predict_batch
        temp_df = session_events.copy()
        predictions = self.predict_batch(temp_df, **kwargs)
        session_id = session_events['session'].iloc[0]
        return predictions.get(session_id, [])
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {'name': self.name}
    
    def __str__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"