"""
Smart Grid AI - Ensemble Forecasting Project
Main module initialization
"""

from .data_loader import generate_synthetic_data, preprocess_data, create_sequences, get_data_stats
from .models.ensemble import StackingEnsemble, LSTMBase, TransformerBase

__version__ = "1.0.0"
__all__ = [
    'generate_synthetic_data',
    'preprocess_data',
    'create_sequences',
    'get_data_stats',
    'StackingEnsemble',
    'LSTMBase',
    'TransformerBase'
]
