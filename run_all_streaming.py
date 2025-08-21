#!/usr/bin/env python3
"""
OTTO Multi-Objective Recommender System - Memory-Efficient Streaming Pipeline

This script processes the large dataset efficiently without loading everything into RAM:
1. Stream through data, randomly assign sessions to train/val/test splits
2. Use subset of training data for efficiency
3. Use ALL test data for final evaluation
4. Train models and evaluate with proper memory management
"""

import os
import sys
import time
import logging
import json
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.streaming_preprocessor import StreamingOTTOProcessor, StreamingDataLoader
from src.evaluation.metrics import OTTOEvaluator
from src.models.baseline import PopularityRecommender, SessionBasedRecommender, HybridRecommender

def setup_logging():
    """Set up logging with progress tracking."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('otto_streaming_experiments.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def print_progress(step, total_steps, message):
    """Print progress with step counter."""
    print(f"\n{'='*80}")
    print(f"STEP {step}/{total_steps}: {message}")
    print(f"{'='*80}")

def prepare_test_data_streaming(test_loader: StreamingDataLoader, split_ratio: float = 0.8):
    """Prepare test data in streaming fashion."""
    logger = logging.getLogger(__name__)
    logger.info("Preparing test data for evaluation in streaming mode...")
    
    test_inputs = []
    test_ground_truth = {}
    processed_sessions = 0
    
    # Process test data in chunks to manage memory
    for chunk_df in test_loader.load_events_streaming(chunk_size=50000):
        for session_id, session_events in chunk_df.groupby('session'):
            session_events = session_events.sort_values('ts')
            
            if len(session_events) < 2:
                continue
            
            split_idx = max(1, int(len(session_events) * split_ratio))
            
            # Input events (first part of session)
            input_events = session_events.iloc[:split_idx]
            test_inputs.append(input_events)
            
            # Ground truth events (second part of session)
            gt_events = session_events.iloc[split_idx:]
            
            session_gt = {'clicks': [], 'carts': [], 'orders': []}
            for action_type in ['clicks', 'carts', 'orders']:
                type_events = gt_events[gt_events['type'] == action_type]
                aids = type_events['aid'].unique().tolist()
                session_gt[action_type] = aids
            
            test_ground_truth[session_id] = session_gt
            processed_sessions += 1
            
            if processed_sessions % 10000 == 0:
                logger.info(f"Processed {processed_sessions} test sessions")
    
    test_input_df = pd.concat(test_inputs, ignore_index=True) if test_inputs else pd.DataFrame()
    logger.info(f"Prepared {len(test_ground_truth)} sessions for evaluation")
    
    return test_input_df, test_ground_truth

def train_model_with_streaming(model, model_name: str, train_loader: StreamingDataLoader, 
                              max_train_sessions: int = None):
    """Train model using streaming data loader."""
    logger = logging.getLogger(__name__)
    logger.info(f"Training {model_name} with streaming data...")
    
    # Load training data with session limit for efficiency
    train_events = train_loader.load_events_batch(max_sessions=max_train_sessions)
    
    if len(train_events) == 0:
        raise ValueError("No training data loaded!")
    
    logger.info(f"Training {model_name} on {train_events['session'].nunique()} sessions, "
                f"{len(train_events)} events")
    
    # Train the model
    start_time = time.time()
    model.fit(train_events)
    training_time = time.time() - start_time
    
    logger.info(f"{model_name} training completed in {training_time:.2f} seconds")
    
    return training_time

def main():
    """Main streaming pipeline runner."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    total_steps = 6
    current_step = 0
    
    try:
        # Step 1: Check data file and create streaming processor
        current_step += 1
        print_progress(current_step, total_steps, "Setting up streaming data processor")
        
        data_path = "train.jsonl"
        if not os.path.exists(data_path):
            logger.error(f"Data file {data_path} not found!")
            return
        
        # Create streaming processor with small training sample for efficiency
        processor = StreamingOTTOProcessor(
            data_path=data_path,
            output_dir="data/processed",
            train_sample_ratio=0.03  # Use only 3% of training data for speed
        )
        
        # Step 2: Process data in streaming fashion
        current_step += 1
        print_progress(current_step, total_steps, "Processing data in streaming fashion (this may take a while...)")
        
        # Check if processed files already exist
        processed_dir = Path("data/processed")
        if (processed_dir / "train_events.jsonl").exists() and \
           (processed_dir / "val_events.jsonl").exists() and \
           (processed_dir / "test_events.jsonl").exists():
            logger.info("Processed files already exist, skipping streaming processing")
            
            # Load metadata
            with open(processed_dir / "split_metadata.json", 'r') as f:
                metadata = json.load(f)
        else:
            logger.info("Processing raw data file...")
            metadata = processor.process_streaming()
        
        logger.info("Data processing completed!")
        logger.info("Split statistics:")
        for split in ['train', 'val', 'test']:
            logger.info(f"{split.capitalize()}: {metadata['session_counts'][split]} sessions, "
                       f"{metadata['event_counts'][split]} events")
        
        # Step 3: Initialize models
        current_step += 1
        print_progress(current_step, total_steps, "Initializing models")
        
        # Use simpler models for large-scale processing
        models = {
            'Popularity': PopularityRecommender(top_k=20),
            'SessionBased': SessionBasedRecommender(top_k=20, similarity_threshold=0.2),
            'Hybrid': HybridRecommender(top_k=20, popularity_weight=0.4)
        }
        
        logger.info(f"Initialized {len(models)} models: {list(models.keys())}")
        
        # Step 4: Prepare test data (use ALL test data)
        current_step += 1
        print_progress(current_step, total_steps, "Loading ALL test data for evaluation")
        
        test_loader = StreamingDataLoader("data/processed/test_events.jsonl")
        test_input_df, ground_truth = prepare_test_data_streaming(test_loader)
        
        logger.info(f"Test evaluation prepared: {len(ground_truth)} sessions")
        
        # Step 5: Train and evaluate models
        current_step += 1
        print_progress(current_step, total_steps, "Training and evaluating models")
        
        train_loader = StreamingDataLoader("data/processed/train_events.jsonl")
        evaluator = OTTOEvaluator()
        all_results = {}
        
        # Determine training data limit based on available training sessions
        max_train_sessions = min(50000, metadata['session_counts']['train'])  # Cap at 50k for efficiency
        logger.info(f"Will use up to {max_train_sessions} training sessions per model")
        
        for i, (model_name, model) in enumerate(models.items(), 1):
            print(f"\n{'-'*60}")
            print(f"TRAINING AND EVALUATING MODEL {i}/{len(models)}: {model_name}")
            print(f"{'-'*60}")
            
            try:
                # Train model with streaming data
                training_time = train_model_with_streaming(
                    model, model_name, train_loader, max_train_sessions
                )
                
                # Make predictions on test data
                logger.info(f"Making predictions with {model_name} on {len(test_input_df)} test events...")
                start_time = time.time()
                predictions = model.predict_batch(test_input_df)
                prediction_time = time.time() - start_time
                logger.info(f"{model_name} predictions completed in {prediction_time:.2f} seconds")
                
                # Convert to multi-objective format for evaluation
                multi_predictions = evaluator.create_prediction_format(predictions)
                
                # Evaluate on ALL test data
                logger.info(f"Evaluating {model_name} on ALL test data...")
                results = evaluator.evaluate_predictions(multi_predictions, ground_truth)
                
                # Add metadata
                results['training_time'] = training_time
                results['prediction_time'] = prediction_time
                results['model_name'] = model_name
                results['train_sessions_used'] = min(max_train_sessions, metadata['session_counts']['train'])
                results['test_sessions_evaluated'] = len(ground_truth)
                
                all_results[model_name] = results
                
                # Print results immediately
                evaluator.print_evaluation_results(results, model_name)
                logger.info(f"Training sessions used: {results['train_sessions_used']}")
                logger.info(f"Test sessions evaluated: {results['test_sessions_evaluated']}")
                
            except Exception as e:
                logger.error(f"Error with {model_name}: {e}")
                continue
        
        # Step 6: Final results and summary
        current_step += 1
        print_progress(current_step, total_steps, "Generating final results")
        
        # Save results
        results_file = "streaming_experiment_results.json"
        json_results = {}
        for model_name, model_results in all_results.items():
            json_results[model_name] = {
                k: float(v) if isinstance(v, (float, int)) else str(v)
                for k, v in model_results.items()
            }
        
        # Add metadata to results
        json_results['_metadata'] = metadata
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        
        # Print final summary
        print("\n" + "="*100)
        print("FINAL STREAMING EXPERIMENT SUMMARY")
        print("="*100)
        
        if all_results:
            sorted_results = sorted(
                all_results.items(),
                key=lambda x: x[1].get('weighted_recall', 0),
                reverse=True
            )
            
            print(f"{'Model':<15} {'Weighted':<10} {'Clicks':<8} {'Carts':<8} {'Orders':<8} "
                  f"{'Train Time':<12} {'Train Sessions':<15} {'Test Sessions':<15}")
            print(f"{'':15} {'Recall':<10} {'Recall':<8} {'Recall':<8} {'Recall':<8} "
                  f"{'(sec)':<12} {'Used':<15} {'Evaluated':<15}")
            print("-" * 110)
            
            for model_name, results in sorted_results:
                print(f"{model_name:<15} "
                      f"{results.get('weighted_recall', 0):<10.4f} "
                      f"{results.get('recall_clicks', 0):<8.4f} "
                      f"{results.get('recall_carts', 0):<8.4f} "
                      f"{results.get('recall_orders', 0):<8.4f} "
                      f"{results.get('training_time', 0):<12.1f} "
                      f"{results.get('train_sessions_used', 0):<15,} "
                      f"{results.get('test_sessions_evaluated', 0):<15,}")
            
            print("=" * 110)
            
            best_model, best_results = sorted_results[0]
            print(f"\nBEST MODEL: {best_model}")
            print(f"BEST WEIGHTED RECALL: {best_results.get('weighted_recall', 0):.4f}")
            print(f"EVALUATED ON: {best_results.get('test_sessions_evaluated', 0):,} test sessions")
            
        else:
            print("No successful results to display.")
        
        print(f"\nStreaming experiments completed successfully!")
        print(f"Results saved to: {results_file}")
        print(f"Log saved to: otto_streaming_experiments.log")
        
    except Exception as e:
        logger.error(f"Streaming pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()