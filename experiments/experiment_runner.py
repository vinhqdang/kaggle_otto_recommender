import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Any

from src.data.preprocessing import OTTODataProcessor
from src.evaluation.metrics import OTTOEvaluator
from src.models.baseline import PopularityRecommender, SessionBasedRecommender, HybridRecommender
from src.models.deep_learning import SASRecRecommender

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Runs experiments comparing different recommendation algorithms."""
    
    def __init__(self, data_path: str, output_dir: str = "experiments/results"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.processor = OTTODataProcessor(data_path)
        self.evaluator = OTTOEvaluator()
        
        self.train_events = None
        self.val_events = None
        self.test_events = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess the data."""
        logger.info("Loading and preprocessing data...")
        
        # Load data
        self.processor.load_data()
        self.processor.preprocess_events()
        
        # Print dataset statistics
        stats = self.processor.get_statistics()
        logger.info("Dataset Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        # Split data
        self.train_events, self.val_events, self.test_events = self.processor.split_data()
        
        # Save processed data
        self.processor.save_processed_data(
            self.train_events, self.val_events, self.test_events,
            output_dir="data/processed"
        )
        
        logger.info("Data preprocessing completed")
    
    def prepare_test_data_for_evaluation(self, test_events: pd.DataFrame, split_ratio: float = 0.8):
        """Prepare test data by splitting sessions for evaluation."""
        test_inputs = []
        test_ground_truth = {}
        
        for session_id, session_events in test_events.groupby('session'):
            session_events = session_events.sort_values('ts')
            
            if len(session_events) < 2:
                continue  # Skip single-event sessions
            
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
        
        test_input_df = pd.concat(test_inputs, ignore_index=True)
        
        return test_input_df, test_ground_truth
    
    def run_experiment(self, model, model_name: str, use_validation: bool = False) -> Dict[str, Any]:
        """Run experiment for a single model."""
        logger.info(f"Running experiment for {model_name}...")
        
        # Select test data
        test_data = self.val_events if use_validation else self.test_events
        
        # Train model
        start_time = time.time()
        model.fit(self.train_events)
        training_time = time.time() - start_time
        
        # Prepare test data
        test_input_df, ground_truth = self.prepare_test_data_for_evaluation(test_data)
        
        # Make predictions
        start_time = time.time()
        predictions = model.predict_batch(test_input_df)
        prediction_time = time.time() - start_time
        
        # Convert to multi-objective format for evaluation
        multi_predictions = self.evaluator.create_prediction_format(predictions)
        
        # Evaluate
        results = self.evaluator.evaluate_predictions(multi_predictions, ground_truth)
        
        # Add timing information
        results['training_time'] = training_time
        results['prediction_time'] = prediction_time
        results['model_name'] = model_name
        
        # Print results
        self.evaluator.print_evaluation_results(results, model_name)
        
        return results
    
    def run_all_experiments(self, use_validation: bool = False) -> Dict[str, Dict[str, Any]]:
        """Run experiments for all models."""
        logger.info("Starting comprehensive experiments...")
        
        if self.train_events is None:
            self.load_and_preprocess_data()
        
        # Define models to test
        models = {
            'Popularity': PopularityRecommender(top_k=20),
            'SessionBased': SessionBasedRecommender(top_k=20),
            'Hybrid': HybridRecommender(top_k=20, popularity_weight=0.3),
            'SASRec': SASRecRecommender(
                hidden_size=64,
                num_blocks=1, 
                num_heads=2,
                max_seq_len=20,
                batch_size=128,
                num_epochs=3,  # Reduced for faster training
                learning_rate=1e-3
            )
        }
        
        all_results = {}
        
        # Run experiments
        for model_name, model in models.items():
            try:
                results = self.run_experiment(model, model_name, use_validation)
                all_results[model_name] = results
            except Exception as e:
                logger.error(f"Error running experiment for {model_name}: {e}")
                continue
        
        # Save results
        self._save_results(all_results, use_validation)
        
        # Print summary
        self._print_summary(all_results)
        
        return all_results
    
    def _save_results(self, results: Dict[str, Dict[str, Any]], use_validation: bool):
        """Save experiment results to file."""
        suffix = "validation" if use_validation else "test"
        results_file = self.output_dir / f"experiment_results_{suffix}.json"
        
        # Convert numpy types to Python types for JSON serialization
        json_results = {}
        for model_name, model_results in results.items():
            json_results[model_name] = {
                k: float(v) if isinstance(v, (np.float32, np.float64)) else 
                   int(v) if isinstance(v, (np.int32, np.int64)) else v
                for k, v in model_results.items()
            }
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
    
    def _print_summary(self, results: Dict[str, Dict[str, Any]]):
        """Print experiment summary."""
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        
        # Sort by weighted recall
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].get('weighted_recall', 0),
            reverse=True
        )
        
        print(f"{'Model':<15} {'Weighted Recall':<15} {'Clicks':<10} {'Carts':<10} {'Orders':<10} {'Train Time':<12}")
        print("-" * 80)
        
        for model_name, model_results in sorted_results:
            print(f"{model_name:<15} "
                  f"{model_results.get('weighted_recall', 0):<15.4f} "
                  f"{model_results.get('recall_clicks', 0):<10.4f} "
                  f"{model_results.get('recall_carts', 0):<10.4f} "
                  f"{model_results.get('recall_orders', 0):<10.4f} "
                  f"{model_results.get('training_time', 0):<12.2f}")
        
        print("="*80)
        
        # Best model
        if sorted_results:
            best_model, best_results = sorted_results[0]
            print(f"Best Model: {best_model}")
            print(f"Best Weighted Recall: {best_results.get('weighted_recall', 0):.4f}")

def main():
    """Main experiment runner."""
    # Set up paths
    data_path = "train.jsonl"
    
    if not os.path.exists(data_path):
        logger.error(f"Data file {data_path} not found!")
        return
    
    # Run experiments
    runner = ExperimentRunner(data_path)
    
    # First run on validation set for quick testing
    logger.info("Running validation experiments...")
    val_results = runner.run_all_experiments(use_validation=True)
    
    # Then run on full test set
    logger.info("Running test experiments...")
    test_results = runner.run_all_experiments(use_validation=False)
    
    logger.info("All experiments completed!")

if __name__ == "__main__":
    main()