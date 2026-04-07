"""Text encoder using PhoBERT for Vietnamese text"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Tuple, Dict


class TextEncoder(nn.Module):
    """
    Text encoder using PhoBERT.
    
    Extracts 768-dimensional embeddings from Vietnamese text using
    the pretrained PhoBERT model.
    """
    
    def __init__(
        self,
        model_name: str = 'vinai/phobert-base',
        max_length: int = 256,
        freeze_embeddings: bool = False
    ):
        """
        Initialize text encoder.
        
        Args:
            model_name: Pretrained model identifier
            max_length: Maximum token length
            freeze_embeddings: Freeze pretrained weights if True
        """
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        if freeze_embeddings:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, text: list, device: torch.device) -> torch.Tensor:
        """
        Encode text to embeddings.
        
        Args:
            text: List of text strings
            device: Device to use
            
        Returns:
            Embeddings (batch_size, 768)
        """
        # Tokenize
        tokens = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**tokens)
        
        # Extract [CLS] token embeddings (pooled representation)
        embeddings = outputs.last_hidden_state[:, 0, :]
        
        return embeddings
    
    @property
    def output_dim(self) -> int:
        return 768
