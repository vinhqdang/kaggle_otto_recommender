#!/usr/bin/env python3
"""
OTTO Multi-Objective Recommender System - Complete Pipeline Runner

This script runs the complete pipeline with progress tracking:
1. Data preprocessing (60/20/20 split)
2. Model training and evaluation
3. Results reporting

Run this script and it will handle the large dataset processing efficiently.
"""

import os
import sys
import time
import logging
from pathlib import Path
import pandas as pd
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.preprocessing import OTTODataProcessor
from src.evaluation.metrics import OTTOEvaluator
from src.models.baseline import PopularityRecommender, SessionBasedRecommender, HybridRecommender
from src.models.deep_learning import SASRecRecommender

def setup_logging():
    """Set up logging with progress tracking."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('otto_experiments.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def print_progress(step, total_steps, message):
    """Print progress with step counter."""
    print(f"\n{'='*80}")
    print(f"STEP {step}/{total_steps}: {message}")
    print(f"{'='*80}")

def main():
    """Main pipeline runner."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    total_steps = 8
    current_step = 0
    
    try:
        # Step 1: Initialize and check data
        current_step += 1
        print_progress(current_step, total_steps, "Initializing and checking data file")
        
        data_path = "train.jsonl"
        if not os.path.exists(data_path):
            logger.error(f"Data file {data_path} not found!")
            return
        
        logger.info(f"Found data file: {data_path}")
        
        # Step 2: Load and preprocess data
        current_step += 1
        print_progress(current_step, total_steps, "Loading and preprocessing data (this may take a while...)")
        
        processor = OTTODataProcessor(data_path)
        
        # Load data with progress tracking
        logger.info("Loading JSONL data...")
        start_time = time.time()
        processor.load_data()
        load_time = time.time() - start_time
        logger.info(f"Data loading completed in {load_time:.2f} seconds")
        
        # Preprocess events
        logger.info("Preprocessing events...")
        start_time = time.time()
        processor.preprocess_events()
        preprocess_time = time.time() - start_time
        logger.info(f"Event preprocessing completed in {preprocess_time:.2f} seconds")
        
        # Step 3: Print dataset statistics
        current_step += 1
        print_progress(current_step, total_steps, "Analyzing dataset statistics")
        
        stats = processor.get_statistics()
        print("\nDATASET STATISTICS:")
        print("-" * 40)
        for key, value in stats.items():
            print(f"{key}: {value}")
        print("-" * 40)
        
        # Step 4: Split data
        current_step += 1
        print_progress(current_step, total_steps, "Splitting data into train/val/test (60/20/20)")
        
        start_time = time.time()
        train_events, val_events, test_events = processor.split_data()
        split_time = time.time() - start_time
        logger.info(f"Data splitting completed in {split_time:.2f} seconds")
        
        # Save processed data
        logger.info("Saving processed data...")
        processor.save_processed_data(train_events, val_events, test_events)
        
        # Step 5: Initialize models and evaluator
        current_step += 1
        print_progress(current_step, total_steps, "Initializing models and evaluator")
        
        evaluator = OTTOEvaluator()
        
        # Define models with optimized parameters for large dataset
        models = {
            'Popularity': PopularityRecommender(top_k=20),
            'SessionBased': SessionBasedRecommender(top_k=20, similarity_threshold=0.1),
            'Hybrid': HybridRecommender(top_k=20, popularity_weight=0.3)
        }
        
        # Add SASRec with reduced parameters for faster training on large dataset
        models['SASRec'] = SASRecRecommender(
            hidden_size=32,      # Reduced from 64
            num_blocks=1,        # Reduced from 2
            num_heads=2,
            max_seq_len=15,      # Reduced from 20
            batch_size=512,      # Increased for efficiency
            num_epochs=2,        # Reduced from 3
            learning_rate=1e-3
        )
        
        logger.info(f"Initialized {len(models)} models: {list(models.keys())}")
        
        # Step 6: Prepare test data for evaluation
        current_step += 1
        print_progress(current_step, total_steps, "Preparing test data for evaluation")
        
        def prepare_test_data(test_events, split_ratio=0.8):
            """Prepare test data by splitting sessions."""
            logger.info("Preparing test data for evaluation...")
            test_inputs = []
            test_ground_truth = {}
            
            processed_sessions = 0
            total_sessions = test_events['session'].nunique()
            
            for session_id, session_events in test_events.groupby('session'):
                processed_sessions += 1
                if processed_sessions % 10000 == 0:
                    logger.info(f"Processed {processed_sessions}/{total_sessions} sessions for evaluation")
                
                session_events = session_events.sort_values('ts')
                
                if len(session_events) < 2:
                    continue
                
                split_idx = max(1, int(len(session_events) * split_ratio))
                
                # Input events
                input_events = session_events.iloc[:split_idx]
                test_inputs.append(input_events)
                
                # Ground truth events
                gt_events = session_events.iloc[split_idx:]
                
                session_gt = {'clicks': [], 'carts': [], 'orders': []}
                for action_type in ['clicks', 'carts', 'orders']:
                    type_events = gt_events[gt_events['type'] == action_type]
                    aids = type_events['aid'].unique().tolist()
                    session_gt[action_type] = aids
                
                test_ground_truth[session_id] = session_gt
            
            test_input_df = pd.concat(test_inputs, ignore_index=True)
            logger.info(f"Prepared {len(test_ground_truth)} sessions for evaluation")
            
            return test_input_df, test_ground_truth
        
        # Use validation set for faster initial testing
        logger.info("Using validation set for evaluation")
        test_input_df, ground_truth = prepare_test_data(val_events)
        
        # Step 7: Run model experiments
        current_step += 1
        print_progress(current_step, total_steps, "Running model experiments")
        
        all_results = {}
        
        for i, (model_name, model) in enumerate(models.items(), 1):
            print(f"\n{'-'*60}")
            print(f"TRAINING MODEL {i}/{len(models)}: {model_name}")
            print(f"{'-'*60}")
            
            try:
                # Train model
                logger.info(f"Training {model_name}...")
                start_time = time.time()
                model.fit(train_events)
                training_time = time.time() - start_time
                logger.info(f"{model_name} training completed in {training_time:.2f} seconds")
                
                # Make predictions
                logger.info(f"Making predictions with {model_name}...")
                start_time = time.time()
                predictions = model.predict_batch(test_input_df)
                prediction_time = time.time() - start_time
                logger.info(f"{model_name} predictions completed in {prediction_time:.2f} seconds")
                
                # Convert to multi-objective format
                multi_predictions = evaluator.create_prediction_format(predictions)
                
                # Evaluate
                logger.info(f"Evaluating {model_name}...")
                results = evaluator.evaluate_predictions(multi_predictions, ground_truth)
                
                # Add timing information
                results['training_time'] = training_time
                results['prediction_time'] = prediction_time
                results['model_name'] = model_name
                
                all_results[model_name] = results
                
                # Print results immediately
                evaluator.print_evaluation_results(results, model_name)
                
            except Exception as e:
                logger.error(f"Error with {model_name}: {e}")
                continue
        
        # Step 8: Final results and summary
        current_step += 1
        print_progress(current_step, total_steps, "Generating final results and summary")
        
        # Save results
        results_file = "experiment_results.json"
        json_results = {}
        for model_name, model_results in all_results.items():
            json_results[model_name] = {
                k: float(v) if isinstance(v, (float, int)) else str(v)
                for k, v in model_results.items()
            }
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        
        # Print final summary
        print("\n" + "="*80)
        print("FINAL EXPERIMENT SUMMARY")
        print("="*80)
        
        if all_results:
            # Sort by weighted recall
            sorted_results = sorted(
                all_results.items(),
                key=lambda x: x[1].get('weighted_recall', 0),
                reverse=True
            )
            
            print(f"{'Model':<15} {'Weighted Recall':<15} {'Clicks':<10} {'Carts':<10} {'Orders':<10} {'Train Time':<12}")
            print("-" * 85)
            
            for model_name, model_results in sorted_results:
                print(f"{model_name:<15} "
                      f"{model_results.get('weighted_recall', 0):<15.4f} "
                      f"{model_results.get('recall_clicks', 0):<10.4f} "
                      f"{model_results.get('recall_carts', 0):<10.4f} "
                      f"{model_results.get('recall_orders', 0):<10.4f} "
                      f"{model_results.get('training_time', 0):<12.2f}")
            
            print("=" * 85)
            
            # Best model
            best_model, best_results = sorted_results[0]
            print(f"\nBEST MODEL: {best_model}")
            print(f"BEST WEIGHTED RECALL: {best_results.get('weighted_recall', 0):.4f}")
            
        else:
            print("No successful results to display.")
        
        print(f"\nAll experiments completed successfully!")
        print(f"Results saved to: {results_file}")
        print(f"Log saved to: otto_experiments.log")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()