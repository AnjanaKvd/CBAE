import torch
import torch.nn as nn
import numpy as np
from typing import Union, List

class CLIPEncoder(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        import open_clip
        
        # Load pre-trained CLIP model
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', device=device)
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        
        # Keep only the text encoder portion (we don't need image encoding for CBAE conditioning)
        # Note: OpenCLIP encapsulates both, we just retain what we need dynamically.
        self.model = model
        
        # Apply INT8 dynamic quantization targeting linear layers
        self.model = torch.quantization.quantize_dynamic(
            self.model, {torch.nn.Linear}, dtype=torch.qint8
        )
        
        # Fix for open_clip expecting weight.dtype
        for m in self.model.modules():
            if hasattr(m, 'mlp') and hasattr(m.mlp, 'c_fc'):
                m.mlp.c_fc.int8_original_dtype = torch.float32
        
        # Freeze all parameters
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
            
    def encode_text(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Encodes a text string or list of strings into normalized CLIP embeddings.
        Output: (Batch, 512) float tensor.
        """
        if isinstance(text, str):
            text = [text]
            
        with torch.no_grad():
            text_tokens = self.tokenizer(text)
            text_features = self.model.encode_text(text_tokens)
            
            # Normalize to unit vector
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
        return text_features


class WhisperEncoder(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        from transformers import WhisperModel, WhisperFeatureExtractor
        
        # We only need the encoder portion of Whisper for feature extraction, 
        # but loading the full model then extracting the encoder is standard.
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
        full_model = WhisperModel.from_pretrained("openai/whisper-tiny").to(device)
        self.encoder = full_model.encoder
        
        # Apply INT8 dynamic quantization targeting linear layers
        self.encoder = torch.quantization.quantize_dynamic(
            self.encoder, {torch.nn.Linear}, dtype=torch.qint8
        )
        
        # Freeze all parameters
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    def encode_audio(self, audio_array: np.ndarray, sr: int = 16000) -> torch.Tensor:
        """
        Encodes raw audio waveform (16kHz) into Whisper encoder representations.
        Returns: (Batch, T, 384) float tensor.
        """
        # Ensure 16kHz sampling rate is explicitly handled via the feature extractor
        with torch.no_grad():
            inputs = self.feature_extractor(
                audio_array, 
                sampling_rate=sr, 
                return_tensors="pt"
            )
            input_features = inputs.input_features
            
            outputs = self.encoder(input_features)
            
            # outputs.last_hidden_state shape: (Batch, T, 384)
            return outputs.last_hidden_state
