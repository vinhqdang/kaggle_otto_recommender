import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OTTODataProcessor:
    """Data processor for OTTO Multi-Objective Recommender System dataset."""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.sessions_df = None
        self.events_df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load JSONL data and convert to DataFrame."""
        logger.info(f"Loading data from {self.data_path}")
        
        sessions = []
        events = []
        
        with open(self.data_path, 'r') as f:
            for line_idx, line in enumerate(f):
                try:
                    session_data = json.loads(line.strip())
                    session_id = session_data['session']
                    
                    for event in session_data['events']:
                        events.append({
                            'session': session_id,
                            'aid': event['aid'],
                            'ts': event['ts'],
                            'type': event['type']
                        })
                    
                    sessions.append({
                        'session': session_id,
                        'num_events': len(session_data['events'])
                    })
                    
                    if line_idx % 100000 == 0:
                        logger.info(f"Processed {line_idx} sessions")
                        
                except Exception as e:
                    logger.error(f"Error parsing line {line_idx}: {e}")
                    continue
        
        self.sessions_df = pd.DataFrame(sessions)
        self.events_df = pd.DataFrame(events)
        
        logger.info(f"Loaded {len(self.sessions_df)} sessions with {len(self.events_df)} events")
        return self.events_df
    
    def preprocess_events(self) -> pd.DataFrame:
        """Preprocess event data for modeling."""
        if self.events_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        logger.info("Preprocessing events...")
        
        # Convert timestamp to datetime
        self.events_df['datetime'] = pd.to_datetime(self.events_df['ts'], unit='ms')
        
        # Create event type encodings
        type_mapping = {'clicks': 0, 'carts': 1, 'orders': 2}
        self.events_df['type_encoded'] = self.events_df['type'].map(type_mapping)
        
        # Sort by session and timestamp
        self.events_df = self.events_df.sort_values(['session', 'ts']).reset_index(drop=True)
        
        # Add sequence position within session
        self.events_df['seq_pos'] = self.events_df.groupby('session').cumcount()
        
        # Add time since session start
        session_start = self.events_df.groupby('session')['ts'].transform('min')
        self.events_df['time_since_start'] = self.events_df['ts'] - session_start
        
        logger.info("Event preprocessing completed")
        return self.events_df
    
    def create_targets(self) -> pd.DataFrame:
        """Create target labels for next item prediction."""
        logger.info("Creating prediction targets...")
        
        targets = []
        
        for session_id, group in self.events_df.groupby('session'):
            events = group.sort_values('ts')
            
            # For each prefix of the session, predict what happens next
            for i in range(1, len(events)):
                # Use events up to position i-1 as input
                input_events = events.iloc[:i]
                target_event = events.iloc[i]
                
                # Create different targets for different action types
                targets.append({
                    'session': session_id,
                    'prefix_length': i,
                    'last_aid': input_events.iloc[-1]['aid'],
                    'last_type': input_events.iloc[-1]['type'],
                    'target_aid': target_event['aid'],
                    'target_type': target_event['type'],
                    'target_type_encoded': target_event['type_encoded'],
                    'time_to_next': target_event['ts'] - input_events.iloc[-1]['ts']
                })
        
        targets_df = pd.DataFrame(targets)
        logger.info(f"Created {len(targets_df)} prediction targets")
        
        return targets_df
    
    def split_data(self, test_size: float = 0.4, val_size: float = 0.5, 
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/val/test (60/20/20)."""
        logger.info("Splitting data into train/val/test...")
        
        if self.sessions_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Split sessions to ensure no data leakage
        session_ids = self.sessions_df['session'].unique()
        
        # First split: 60% train, 40% temp (which will be split into 20% val, 20% test)
        train_sessions, temp_sessions = train_test_split(
            session_ids, test_size=test_size, random_state=random_state
        )
        
        # Second split: 20% val, 20% test from the 40% temp
        val_sessions, test_sessions = train_test_split(
            temp_sessions, test_size=val_size, random_state=random_state
        )
        
        # Filter events by session splits
        train_events = self.events_df[self.events_df['session'].isin(train_sessions)].copy()
        val_events = self.events_df[self.events_df['session'].isin(val_sessions)].copy()
        test_events = self.events_df[self.events_df['session'].isin(test_sessions)].copy()
        
        logger.info(f"Train: {len(train_sessions)} sessions, {len(train_events)} events")
        logger.info(f"Val: {len(val_sessions)} sessions, {len(val_events)} events")
        logger.info(f"Test: {len(test_sessions)} sessions, {len(test_events)} events")
        
        return train_events, val_events, test_events
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        if self.events_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        stats = {
            'num_sessions': self.events_df['session'].nunique(),
            'num_unique_items': self.events_df['aid'].nunique(),
            'num_events': len(self.events_df),
            'avg_session_length': self.events_df.groupby('session').size().mean(),
            'median_session_length': self.events_df.groupby('session').size().median(),
            'type_distribution': self.events_df['type'].value_counts().to_dict(),
            'time_span_days': (self.events_df['ts'].max() - self.events_df['ts'].min()) / (1000 * 60 * 60 * 24)
        }
        
        return stats
    
    def create_interaction_matrix(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Create user-item interaction matrix for collaborative filtering."""
        # Create weighted interactions based on event type
        weights = {'clicks': 1, 'carts': 3, 'orders': 5}
        events_df['weight'] = events_df['type'].map(weights)
        
        # Aggregate by session and item
        interactions = events_df.groupby(['session', 'aid'])['weight'].sum().reset_index()
        
        # Pivot to create interaction matrix
        interaction_matrix = interactions.pivot(
            index='session', columns='aid', values='weight'
        ).fillna(0)
        
        return interaction_matrix
    
    def save_processed_data(self, train_events: pd.DataFrame, val_events: pd.DataFrame, 
                           test_events: pd.DataFrame, output_dir: str = "data/processed"):
        """Save processed data splits."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_events.to_parquet(output_path / "train_events.parquet")
        val_events.to_parquet(output_path / "val_events.parquet") 
        test_events.to_parquet(output_path / "test_events.parquet")
        
        # Save interaction matrices
        train_matrix = self.create_interaction_matrix(train_events)
        val_matrix = self.create_interaction_matrix(val_events)
        test_matrix = self.create_interaction_matrix(test_events)
        
        train_matrix.to_parquet(output_path / "train_matrix.parquet")
        val_matrix.to_parquet(output_path / "val_matrix.parquet")
        test_matrix.to_parquet(output_path / "test_matrix.parquet")
        
        logger.info(f"Processed data saved to {output_path}")

def main():
    """Main preprocessing pipeline."""
    processor = OTTODataProcessor("train.jsonl")
    
    # Load and preprocess data
    processor.load_data()
    processor.preprocess_events()
    
    # Print statistics
    stats = processor.get_statistics()
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Split data
    train_events, val_events, test_events = processor.split_data()
    
    # Save processed data
    processor.save_processed_data(train_events, val_events, test_events)
    
    print("\nData preprocessing completed!")

if __name__ == "__main__":
    main()