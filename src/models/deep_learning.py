import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

class SequentialDataset(Dataset):
    """Dataset for sequential recommendation."""
    
    def __init__(self, events_df: pd.DataFrame, max_seq_len: int = 50, 
                 item_encoder: Optional[LabelEncoder] = None):
        self.max_seq_len = max_seq_len
        self.events_df = events_df.copy()
        
        # Encode items
        if item_encoder is None:
            self.item_encoder = LabelEncoder()
            # Add special tokens: 0 for padding, 1 for unknown
            all_items = ['<PAD>', '<UNK>'] + list(events_df['aid'].unique())
            self.item_encoder.fit(all_items)
        else:
            self.item_encoder = item_encoder
        
        # Encode interaction types
        self.type_encoder = LabelEncoder()
        self.type_encoder.fit(['clicks', 'carts', 'orders'])
        
        self.sequences = self._create_sequences()
    
    def _create_sequences(self) -> List[Dict]:
        """Create training sequences from events."""
        sequences = []
        
        for session_id, group in self.events_df.groupby('session'):
            events = group.sort_values('ts')
            
            # Encode items and types
            try:
                item_ids = self.item_encoder.transform(events['aid'].astype(str))
            except ValueError:
                # Handle unknown items
                item_ids = []
                for aid in events['aid'].astype(str):
                    try:
                        item_ids.append(self.item_encoder.transform([aid])[0])
                    except ValueError:
                        item_ids.append(1)  # <UNK> token
                item_ids = np.array(item_ids)
            
            type_ids = self.type_encoder.transform(events['type'])
            
            # Create sequences of different lengths
            for i in range(2, len(events) + 1):
                seq_items = item_ids[:i-1]  # Input sequence
                seq_types = type_ids[:i-1]
                target_item = item_ids[i-1]  # Target item
                target_type = type_ids[i-1]  # Target type
                
                # Pad/truncate sequence
                if len(seq_items) > self.max_seq_len:
                    seq_items = seq_items[-self.max_seq_len:]
                    seq_types = seq_types[-self.max_seq_len:]
                else:
                    pad_len = self.max_seq_len - len(seq_items)
                    seq_items = np.pad(seq_items, (pad_len, 0), constant_values=0)
                    seq_types = np.pad(seq_types, (pad_len, 0), constant_values=0)
                
                sequences.append({
                    'session_id': session_id,
                    'seq_items': seq_items,
                    'seq_types': seq_types,
                    'target_item': target_item,
                    'target_type': target_type
                })
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return {
            'seq_items': torch.LongTensor(seq['seq_items']),
            'seq_types': torch.LongTensor(seq['seq_types']),
            'target_item': torch.LongTensor([seq['target_item']]),
            'target_type': torch.LongTensor([seq['target_type']])
        }

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, hidden_size = x.size()
        
        # Linear transformations
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_size)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            scores.masked_fill_(mask == 0, -1e9)
        
        # Apply causal mask (for autoregressive prediction)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        scores.masked_fill_(causal_mask == 0, -1e9)
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        # Residual connection and layer norm
        output = self.layer_norm(x + context)
        
        return output

class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward."""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout_rate)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, mask=None):
        # Self-attention
        attn_output = self.attention(x, mask)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(attn_output)
        output = self.layer_norm(attn_output + self.dropout(ff_output))
        
        return output

class SASRecModel(nn.Module):
    """Self-Attentive Sequential Recommendation model inspired by SASRec."""
    
    def __init__(self, num_items: int, num_types: int, hidden_size: int = 128, 
                 num_blocks: int = 2, num_heads: int = 2, max_seq_len: int = 50,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.num_items = num_items
        self.num_types = num_types
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.item_embedding = nn.Embedding(num_items, hidden_size, padding_idx=0)
        self.type_embedding = nn.Embedding(num_types, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, dropout_rate)
            for _ in range(num_blocks)
        ])
        
        # Output layers
        self.item_predictor = nn.Linear(hidden_size, num_items)
        self.type_predictor = nn.Linear(hidden_size, num_types)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, seq_items, seq_types):
        batch_size, seq_len = seq_items.size()
        
        # Create position indices
        positions = torch.arange(seq_len, device=seq_items.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        item_emb = self.item_embedding(seq_items)
        type_emb = self.type_embedding(seq_types)
        pos_emb = self.position_embedding(positions)
        
        # Combine embeddings
        x = item_emb + type_emb + pos_emb
        x = self.dropout(x)
        
        # Create attention mask (ignore padding tokens)
        mask = (seq_items != 0).float()
        
        # Apply transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x, mask)
        
        # Predictions
        item_logits = self.item_predictor(x)
        type_logits = self.type_predictor(x)
        
        return item_logits, type_logits

class SASRecRecommender:
    """SASRec-based sequential recommender."""
    
    def __init__(self, hidden_size: int = 128, num_blocks: int = 2, 
                 num_heads: int = 2, max_seq_len: int = 50, 
                 learning_rate: float = 1e-3, batch_size: int = 256,
                 num_epochs: int = 10, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        
        self.model = None
        self.item_encoder = None
        self.type_encoder = None
        
    def fit(self, train_events: pd.DataFrame, val_events: Optional[pd.DataFrame] = None):
        """Train the SASRec model."""
        logger.info("Training SASRec model...")
        
        # Create dataset
        train_dataset = SequentialDataset(train_events, self.max_seq_len)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        self.item_encoder = train_dataset.item_encoder
        self.type_encoder = train_dataset.type_encoder
        
        # Initialize model
        num_items = len(self.item_encoder.classes_)
        num_types = len(self.type_encoder.classes_)
        
        self.model = SASRecModel(
            num_items=num_items,
            num_types=num_types,
            hidden_size=self.hidden_size,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            max_seq_len=self.max_seq_len
        ).to(self.device)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        item_criterion = nn.CrossEntropyLoss(ignore_index=0)
        type_criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            num_batches = 0
            
            for batch in train_loader:
                seq_items = batch['seq_items'].to(self.device)
                seq_types = batch['seq_types'].to(self.device)
                target_items = batch['target_item'].squeeze().to(self.device)
                target_types = batch['target_type'].squeeze().to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                item_logits, type_logits = self.model(seq_items, seq_types)
                
                # Use last position for prediction
                last_item_logits = item_logits[:, -1, :]
                last_type_logits = type_logits[:, -1, :]
                
                # Calculate losses
                item_loss = item_criterion(last_item_logits, target_items)
                type_loss = type_criterion(last_type_logits, target_types)
                
                total_loss_batch = item_loss + 0.1 * type_loss  # Weight type loss less
                
                # Backward pass
                total_loss_batch.backward()
                optimizer.step()
                
                total_loss += total_loss_batch.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            logger.info(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}")
        
        logger.info("SASRec training completed")
    
    def predict_session(self, session_events: pd.DataFrame, top_k: int = 20) -> List[int]:
        """Predict next items for a single session."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        self.model.eval()
        
        # Prepare sequence
        events = session_events.sort_values('ts')
        
        try:
            item_ids = self.item_encoder.transform(events['aid'].astype(str))
        except ValueError:
            # Handle unknown items
            item_ids = []
            for aid in events['aid'].astype(str):
                try:
                    item_ids.append(self.item_encoder.transform([aid])[0])
                except ValueError:
                    item_ids.append(1)  # <UNK> token
            item_ids = np.array(item_ids)
        
        type_ids = self.type_encoder.transform(events['type'])
        
        # Pad/truncate sequence
        if len(item_ids) > self.max_seq_len:
            seq_items = item_ids[-self.max_seq_len:]
            seq_types = type_ids[-self.max_seq_len:]
        else:
            pad_len = self.max_seq_len - len(item_ids)
            seq_items = np.pad(item_ids, (pad_len, 0), constant_values=0)
            seq_types = np.pad(type_ids, (pad_len, 0), constant_values=0)
        
        # Convert to tensors
        seq_items = torch.LongTensor(seq_items).unsqueeze(0).to(self.device)
        seq_types = torch.LongTensor(seq_types).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            item_logits, _ = self.model(seq_items, seq_types)
            item_scores = F.softmax(item_logits[0, -1, :], dim=0)
        
        # Get top-k predictions (exclude padding and unknown tokens and seen items)
        seen_items = set(events['aid'].unique())
        
        # Convert scores to numpy and get top items
        item_scores = item_scores.cpu().numpy()
        top_indices = np.argsort(item_scores)[::-1]
        
        recommendations = []
        for idx in top_indices:
            if idx < 2:  # Skip padding and unknown tokens
                continue
            
            try:
                item_id = int(self.item_encoder.inverse_transform([idx])[0])
                if item_id not in seen_items:
                    recommendations.append(item_id)
                if len(recommendations) >= top_k:
                    break
            except (ValueError, IndexError):
                continue
        
        return recommendations
    
    def predict_batch(self, test_events: pd.DataFrame, top_k: int = 20) -> Dict[int, List[int]]:
        """Predict for multiple sessions."""
        predictions = {}
        
        for session_id, session_events in test_events.groupby('session'):
            try:
                predictions[session_id] = self.predict_session(session_events, top_k)
            except Exception as e:
                logger.warning(f"Error predicting for session {session_id}: {e}")
                predictions[session_id] = []
        
        return predictions