import json
import pandas as pd
import numpy as np
from pathlib import Path
import random
import logging
from typing import Dict, List, Tuple, Optional
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamingOTTOProcessor:
    """Memory-efficient streaming processor for OTTO dataset."""
    
    def __init__(self, data_path: str, output_dir: str = "data/processed", 
                 train_sample_ratio: float = 0.1):
        """
        Args:
            data_path: Path to train.jsonl file
            output_dir: Directory to save processed files
            train_sample_ratio: Fraction of training data to actually use (for efficiency)
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_sample_ratio = train_sample_ratio
        
        # Split ratios
        self.train_ratio = 0.6
        self.val_ratio = 0.2
        self.test_ratio = 0.2
        
        # Counters
        self.session_counts = {'train': 0, 'val': 0, 'test': 0}
        self.event_counts = {'train': 0, 'val': 0, 'test': 0}
        
        # File handles
        self.train_file = None
        self.val_file = None
        self.test_file = None
        
    def _get_split(self) -> str:
        """Randomly assign session to train/val/test split."""
        rand = random.random()
        if rand < self.train_ratio:
            return 'train'
        elif rand < self.train_ratio + self.val_ratio:
            return 'val'
        else:
            return 'test'
    
    def _should_keep_train_session(self) -> bool:
        """Decide whether to keep this training session based on sampling ratio."""
        return random.random() < self.train_sample_ratio
    
    def _open_files(self):
        """Open output files for writing."""
        self.train_file = open(self.output_dir / "train_events.jsonl", 'w')
        self.val_file = open(self.output_dir / "val_events.jsonl", 'w')
        self.test_file = open(self.output_dir / "test_events.jsonl", 'w')
        
    def _close_files(self):
        """Close output files."""
        if self.train_file:
            self.train_file.close()
        if self.val_file:
            self.val_file.close()
        if self.test_file:
            self.test_file.close()
    
    def _write_session(self, session_data: dict, split: str):
        """Write session to appropriate file."""
        session_json = json.dumps(session_data) + '\n'
        
        if split == 'train':
            self.train_file.write(session_json)
        elif split == 'val':
            self.val_file.write(session_json)
        elif split == 'test':
            self.test_file.write(session_json)
    
    def process_streaming(self, random_seed: int = 42):
        """Process the data in streaming fashion."""
        random.seed(random_seed)
        logger.info(f"Starting streaming processing of {self.data_path}")
        logger.info(f"Training data sampling ratio: {self.train_sample_ratio}")
        
        self._open_files()
        
        try:
            with open(self.data_path, 'r') as f:
                for line_idx, line in enumerate(f):
                    try:
                        session_data = json.loads(line.strip())
                        
                        # Determine split
                        split = self._get_split()
                        
                        # For training data, apply sampling
                        if split == 'train' and not self._should_keep_train_session():
                            continue
                        
                        # Write session to appropriate file
                        self._write_session(session_data, split)
                        
                        # Update counters
                        self.session_counts[split] += 1
                        self.event_counts[split] += len(session_data['events'])
                        
                        # Progress logging
                        if line_idx % 100000 == 0:
                            logger.info(f"Processed {line_idx} sessions. "
                                      f"Train: {self.session_counts['train']}, "
                                      f"Val: {self.session_counts['val']}, "
                                      f"Test: {self.session_counts['test']}")
                        
                    except Exception as e:
                        logger.error(f"Error parsing line {line_idx}: {e}")
                        continue
                        
        finally:
            self._close_files()
        
        # Print final statistics
        logger.info("Streaming processing completed!")
        logger.info("Final statistics:")
        for split in ['train', 'val', 'test']:
            logger.info(f"{split.capitalize()}: {self.session_counts[split]} sessions, "
                       f"{self.event_counts[split]} events")
        
        # Save metadata
        metadata = {
            'session_counts': self.session_counts,
            'event_counts': self.event_counts,
            'train_sample_ratio': self.train_sample_ratio,
            'split_ratios': {
                'train': self.train_ratio,
                'val': self.val_ratio, 
                'test': self.test_ratio
            }
        }
        
        with open(self.output_dir / "split_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {self.output_dir / 'split_metadata.json'}")
        
        return metadata

class StreamingDataLoader:
    """Memory-efficient data loader for processed splits."""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
    
    def load_events_streaming(self, chunk_size: int = 10000) -> pd.DataFrame:
        """Load events in chunks to avoid memory issues."""
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
                    
                    # Yield chunk when it reaches chunk_size
                    if len(events) >= chunk_size:
                        df = pd.DataFrame(events)
                        events = []  # Clear for next chunk
                        yield df
                        
                except Exception as e:
                    logger.error(f"Error parsing line {line_idx}: {e}")
                    continue
        
        # Yield remaining events
        if events:
            yield pd.DataFrame(events)
    
    def load_events_batch(self, max_sessions: Optional[int] = None) -> pd.DataFrame:
        """Load events from file, optionally limiting number of sessions."""
        events = []
        sessions_loaded = 0
        
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
                    
                    sessions_loaded += 1
                    
                    if max_sessions and sessions_loaded >= max_sessions:
                        break
                    
                    if line_idx % 10000 == 0:
                        logger.info(f"Loaded {sessions_loaded} sessions, {len(events)} events")
                        
                except Exception as e:
                    logger.error(f"Error parsing line {line_idx}: {e}")
                    continue
        
        logger.info(f"Loaded {sessions_loaded} sessions, {len(events)} total events")
        return pd.DataFrame(events)

def main():
    """Main streaming preprocessing pipeline."""
    # Process data in streaming fashion
    processor = StreamingOTTOProcessor(
        data_path="train.jsonl",
        output_dir="data/processed",
        train_sample_ratio=0.05  # Use only 5% of training data for efficiency
    )
    
    metadata = processor.process_streaming()
    
    print("\nStreaming processing completed!")
    print("Generated files:")
    print("- data/processed/train_events.jsonl")
    print("- data/processed/val_events.jsonl") 
    print("- data/processed/test_events.jsonl")
    print("- data/processed/split_metadata.json")
    
    # Test loading
    print("\nTesting data loading...")
    train_loader = StreamingDataLoader("data/processed/train_events.jsonl")
    
    # Load a small sample for testing
    train_sample = train_loader.load_events_batch(max_sessions=1000)
    print(f"Loaded training sample: {len(train_sample)} events from up to 1000 sessions")
    
    # Show sample statistics
    print(f"Sample statistics:")
    print(f"- Unique sessions: {train_sample['session'].nunique()}")
    print(f"- Unique items: {train_sample['aid'].nunique()}")
    print(f"- Event types: {train_sample['type'].value_counts().to_dict()}")

if __name__ == "__main__":
    main()