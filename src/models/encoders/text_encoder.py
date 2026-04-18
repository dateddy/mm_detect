"""Text encoder using PhoBERT for Vietnamese text"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Optional


class TextEncoder(nn.Module):
    """
    Text encoder using PhoBERT (vinai/phobert-base-v2).
    
    Extracts [CLS] token embeddings (768-dimensional) from Vietnamese text.
    Supports freezing for Phase 1 training.
    """
    
    def __init__(
        self,
        model_name: str = "vinai/phobert-base-v2",
        max_length: int = 256,
        freeze_encoder: bool = False
    ):
        """
        Initialize text encoder.
        
        Args:
            model_name: HuggingFace model identifier (PhoBERT)
            max_length: Maximum token sequence length
            freeze_encoder: Freeze all encoder parameters if True
        """
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, text: list[str], device: torch.device) -> torch.Tensor:
        """
        Encode text to embeddings.
        
        Args:
            text: List of text strings (Vietnamese)
            device: Device to use for computation
            
        Returns:
            Embeddings (batch_size, 768) - [CLS] token representations
        """
        # Tokenize
        tokens = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**tokens)
        
        # Extract [CLS] token (first token of last hidden state)
        embeddings = outputs.last_hidden_state[:, 0, :]  # (batch_size, 768)
        
        return embeddings
    
    def freeze(self) -> None:
        """Freeze all parameters"""
        for param in self.model.parameters():
            param.requires_grad = False
    
    def unfreeze(self) -> None:
        """Unfreeze all parameters"""
        for param in self.model.parameters():
            param.requires_grad = True
    
    def unfreeze_top_k_layers(self, k: int = 4) -> None:
        """
        Unfreeze top k transformer blocks.
        
        Args:
            k: Number of top layers to unfreeze from the end
        """
        # Freeze all
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze top k encoder layers
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            num_layers = len(self.model.encoder.layer)
            for layer_idx in range(max(0, num_layers - k), num_layers):
                for param in self.model.encoder.layer[layer_idx].parameters():
                    param.requires_grad = True
        
        # Always unfreeze layer norm and embeddings
        for param in self.model.bert.embeddings.parameters():
            param.requires_grad = True
        if hasattr(self.model.bert, 'encoder') and hasattr(self.model.bert.encoder, 'layer_norm'):
            for param in self.model.bert.encoder.layer_norm.parameters():
                param.requires_grad = True
