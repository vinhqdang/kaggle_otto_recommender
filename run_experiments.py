#!/usr/bin/env python3
"""
OTTO Multi-Objective Recommender System - Main Experiment Runner

This script runs the complete pipeline:
1. Data preprocessing (60/20/20 split)
2. Model training and evaluation
3. Results reporting

Usage:
    python run_experiments.py
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))
sys.path.append(str(Path(__file__).parent / 'experiments'))

from experiment_runner import ExperimentRunner

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('experiments/otto_experiments.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Main function to run all experiments."""
    parser = argparse.ArgumentParser(description='Run OTTO Recommender System Experiments')
    parser.add_argument('--data-path', default='train.jsonl', 
                       help='Path to the training data file')
    parser.add_argument('--output-dir', default='experiments/results',
                       help='Directory to save results')
    parser.add_argument('--validation-only', action='store_true',
                       help='Run only on validation set (faster)')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        logger.error(f"Data file {args.data_path} not found!")
        sys.exit(1)
    
    # Initialize experiment runner
    runner = ExperimentRunner(args.data_path, args.output_dir)
    
    try:
        if args.validation_only:
            logger.info("Running validation experiments only...")
            results = runner.run_all_experiments(use_validation=True)
        else:
            logger.info("Running full experiments...")
            # Run validation first, then test
            logger.info("Phase 1: Validation experiments")
            val_results = runner.run_all_experiments(use_validation=True)
            
            logger.info("Phase 2: Test experiments")  
            test_results = runner.run_all_experiments(use_validation=False)
        
        logger.info("All experiments completed successfully!")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise

if __name__ == "__main__":
    main()