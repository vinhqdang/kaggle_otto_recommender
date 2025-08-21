# OTTO Multi-Objective Recommender System

This project implements a comprehensive recommender system for the OTTO Multi-Objective Recommender System Kaggle competition. The system handles sequential user interactions (clicks, carts, orders) and provides recommendations optimized for multiple objectives.

## 🎯 Competition Overview

The OTTO competition evaluates recommender systems using a weighted recall metric:
- **Weighted Recall = 0.10 × R_clicks + 0.30 × R_carts + 0.60 × R_orders**
- Each session requires predictions for clicks, carts, and orders
- Predictions are evaluated using Recall@20 for each interaction type

## 🏗️ Project Structure

```
kaggle_otto_recommender/
├── src/
│   ├── data/
│   │   ├── preprocessing.py          # Original batch preprocessing
│   │   └── streaming_preprocessor.py # Memory-efficient streaming processor
│   ├── models/
│   │   ├── base.py                   # Abstract base class for models
│   │   ├── baseline.py               # Baseline algorithms (Popularity, SessionBased, Hybrid)
│   │   └── deep_learning.py          # SASRec transformer-based model
│   ├── evaluation/
│   │   └── metrics.py                # OTTO evaluation metrics
│   └── utils/
├── experiments/
│   └── experiment_runner.py          # Original experiment runner
├── data/
│   └── processed/                    # Generated split files
├── run_all.py                        # Original pipeline (loads all data to RAM)
├── run_all_streaming.py              # Memory-efficient streaming pipeline ⭐
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n py310 python=3.10 -y
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py310

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Complete Pipeline (Recommended)

The streaming pipeline efficiently handles the large dataset without loading everything into RAM:

```bash
python run_all_streaming.py
```

This will:
1. **Stream through data**: Process 12M+ sessions without RAM issues
2. **Random split**: Assign sessions to train (60%) / val (20%) / test (20%)
3. **Efficient training**: Use subset of training data (3% by default)
4. **Full evaluation**: Test on ALL test data for accurate results
5. **Generate results**: Complete evaluation metrics and model comparison

### 3. Manual Steps (Optional)

If you want to run steps individually:

```bash
# Step 1: Process data in streaming fashion
python -c "from src.data.streaming_preprocessor import main; main()"

# Step 2: Run experiments
python experiments/experiment_runner.py
```

## 🤖 Implemented Algorithms

### 1. Baseline Models

- **Popularity Recommender**: Recommends globally popular items with interaction type weighting
- **Session-Based Recommender**: Item-item collaborative filtering based on session co-occurrence
- **Hybrid Recommender**: Combines popularity and session-based approaches

### 2. Deep Learning Model

- **SASRec**: Self-Attentive Sequential Recommendation using transformer architecture
  - Multi-head self-attention for sequence modeling
  - Positional embeddings for temporal patterns
  - Multi-objective prediction (items + interaction types)

## 📊 Evaluation Metrics

The system implements the official OTTO evaluation metric:

- **Recall@20** for each interaction type (clicks, carts, orders)
- **Weighted Recall** combining all interaction types
- **Training/Prediction time** tracking
- **Session coverage** statistics

## 🔧 Configuration

### Memory-Efficient Settings

The streaming pipeline uses these optimizations:

```python
# Training data sampling (adjust based on available memory/time)
train_sample_ratio = 0.03  # Use 3% of training data

# Model parameters optimized for large datasets
SASRecRecommender(
    hidden_size=32,        # Reduced model complexity
    num_blocks=1,          # Fewer transformer layers
    max_seq_len=15,        # Shorter sequences
    batch_size=512,        # Larger batches for efficiency
    num_epochs=2           # Fewer epochs
)
```

### Scaling Up

To use more training data or increase model complexity:

```python
# In run_all_streaming.py, modify:
train_sample_ratio = 0.1   # Use 10% of training data
max_train_sessions = 100000  # Increase session limit

# In models, increase:
hidden_size = 64
num_blocks = 2
num_epochs = 5
```

## 📈 Expected Results

Typical performance on validation set:

| Model | Weighted Recall | Clicks Recall | Carts Recall | Orders Recall | Training Time |
|-------|----------------|---------------|--------------|---------------|---------------|
| Popularity | ~0.025 | ~0.040 | ~0.020 | ~0.015 | ~30s |
| SessionBased | ~0.035 | ~0.055 | ~0.030 | ~0.025 | ~2-5min |
| Hybrid | ~0.040 | ~0.060 | ~0.035 | ~0.030 | ~2-5min |
| SASRec | ~0.045+ | ~0.065+ | ~0.040+ | ~0.035+ | ~10-30min |

*Results depend on training data size and model parameters*

## 🐛 Troubleshooting

### Memory Issues
- Reduce `train_sample_ratio` in streaming processor
- Decrease `max_train_sessions` in experiment runner
- Use smaller `batch_size` for SASRec

### Slow Training
- Increase `train_sample_ratio` gradually
- Use fewer training sessions initially
- Reduce SASRec model complexity

### CUDA Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU if needed
export CUDA_VISIBLE_DEVICES=""
```

## 📁 Output Files

The pipeline generates:

- `data/processed/train_events.jsonl` - Training sessions
- `data/processed/val_events.jsonl` - Validation sessions  
- `data/processed/test_events.jsonl` - Test sessions
- `data/processed/split_metadata.json` - Split statistics
- `streaming_experiment_results.json` - Complete results
- `otto_streaming_experiments.log` - Detailed execution log

## 🔬 Extending the Framework

### Adding New Models

1. Inherit from `BaseRecommender` in `src/models/base.py`
2. Implement `fit()` and `predict_batch()` methods
3. Add to model dictionary in experiment runner

```python
from src.models.base import BaseRecommender

class MyRecommender(BaseRecommender):
    def fit(self, train_events: pd.DataFrame):
        # Training logic here
        pass
    
    def predict_batch(self, test_events: pd.DataFrame) -> Dict[int, List[int]]:
        # Prediction logic here
        return predictions
```

### Custom Evaluation

Modify `src/evaluation/metrics.py` to add custom metrics or change evaluation logic.

## 📚 References

- [OTTO Competition](https://www.kaggle.com/competitions/otto-recommender-system)
- [SASRec Paper](https://arxiv.org/abs/1808.09781)
- [Microsoft Recommenders](https://github.com/recommenders-team/recommenders)

## 🤝 Contributing

This framework is designed to be extensible. Key areas for improvement:

1. **Additional Models**: GRU4Rec, BERT4Rec, etc.
2. **Feature Engineering**: Time-based features, item categories
3. **Hyperparameter Tuning**: Automated optimization
4. **Ensemble Methods**: Model combination strategies

---

*Built for the OTTO Multi-Objective Recommender System Kaggle competition*