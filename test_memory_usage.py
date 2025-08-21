#!/usr/bin/env python3
"""
Memory usage test script to verify garbage collection is working properly.
"""

import os
import sys
import psutil
import gc
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.streaming_preprocessor import StreamingDataLoader
from src.models.baseline import PopularityRecommender

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_memory_cleanup():
    """Test memory cleanup during model training and prediction."""
    print("Testing memory cleanup improvements...")
    
    # Check if test data exists
    data_path = "data/processed/train_events.jsonl"
    if not os.path.exists(data_path):
        print(f"Test data not found at {data_path}")
        print("Please run the streaming preprocessor first:")
        print("python src/data/streaming_preprocessor.py")
        return
    
    print(f"Initial memory usage: {get_memory_usage():.1f} MB")
    
    # Load a small sample of data
    print("Loading training data...")
    loader = StreamingDataLoader(data_path)
    train_data = loader.load_events_batch(max_sessions=1000)
    
    mem_after_load = get_memory_usage()
    print(f"Memory after loading data: {mem_after_load:.1f} MB")
    
    # Train model
    print("Training model...")
    model = PopularityRecommender(top_k=20)
    model.fit(train_data)
    
    mem_after_training = get_memory_usage()
    print(f"Memory after training: {mem_after_training:.1f} MB")
    
    # Force garbage collection
    print("Running garbage collection...")
    gc.collect()
    
    mem_after_gc = get_memory_usage()
    print(f"Memory after garbage collection: {mem_after_gc:.1f} MB")
    
    # Test batch prediction with memory monitoring
    print("Testing batch predictions with memory monitoring...")
    test_data = loader.load_events_batch(max_sessions=500)
    
    mem_before_pred = get_memory_usage()
    print(f"Memory before predictions: {mem_before_pred:.1f} MB")
    
    predictions = model.predict_batch(test_data)
    
    mem_after_pred = get_memory_usage()
    print(f"Memory after predictions: {mem_after_pred:.1f} MB")
    
    # Clean up
    del train_data, test_data, predictions, model
    gc.collect()
    
    final_memory = get_memory_usage()
    print(f"Final memory usage: {final_memory:.1f} MB")
    
    # Summary
    print("\n" + "="*50)
    print("MEMORY USAGE SUMMARY")
    print("="*50)
    print(f"Peak memory usage: {max(mem_after_load, mem_after_training, mem_after_pred):.1f} MB")
    print(f"Memory recovered: {mem_after_pred - final_memory:.1f} MB")
    print(f"Memory efficiency: {'GOOD' if (mem_after_pred - final_memory) > 0 else 'POOR'}")

if __name__ == "__main__":
    test_memory_cleanup()