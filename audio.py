from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from encodec import EncodecModel
from encodec.utils import convert_audio

def create_audio_tokenizer(bandwidth=1.5):
    """Create audio tokenizer with specified settings."""
    tokenizer = AudioTokenizer()
    tokenizer._ensure_model_loaded(bandwidth)
    return tokenizer

class AudioTokenizer:
    """Handles conversion between raw audio and Encodec tokens."""
    
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        sample_rate: int = 24000,
        chunk_size: int = 24000 * 30,  # 30 seconds
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.model = None  # Lazy load to avoid CUDA fork issues
        
    def _ensure_model_loaded(self, bandwidth=1.5):
        """Lazy load the model to avoid CUDA initialization in main process."""
        if self.model is None:
            self.model = EncodecModel.encodec_model_24khz()
            self.model = self.model.to(self.device)
            self.model.set_target_bandwidth(bandwidth)
            self.model.eval()

    @torch.no_grad()
    def encode(self, audio: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Convert raw audio to Encodec codebook indices.
        
        Args:
            audio: Audio array of shape [channels, samples] or [samples]
                  Values should be in [-1, 1] range
                  
        Returns:
            Array of shape [frames, N] containing codebook indices where N depends on bandwidth
        """
        self._ensure_model_loaded()
        
        # Convert to torch first
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        # Handle stereo to mono conversion
        if audio.ndim == 2 and audio.shape[0] == 2:  # Stereo [2, samples]
            audio = audio.mean(dim=0)  # Convert to mono by averaging channels
        elif audio.ndim == 1:  # Already mono
            pass
        else:
            # Handle other cases - take first channel if multi-channel
            if audio.ndim == 2:
                audio = audio[0]  # Take first channel
            
        # Ensure we have mono audio as 1D tensor
        if audio.ndim != 1:
            raise ValueError(f"Expected 1D audio after conversion, got {audio.ndim}D")
            
        # Add channel dimension for Encodec (expects [1, samples] for mono)
        audio = audio.unsqueeze(0)  # Now [1, samples]
            
        # Ensure correct sample rate
        if hasattr(audio, 'sample_rate') and audio.sample_rate != self.sample_rate:
            audio = convert_audio(audio, audio.sample_rate, self.sample_rate)
        
        # Add batch dimension for Encodec: (batch, channels, samples)
        audio = audio.unsqueeze(0)  # Now [1, 1, samples]
            
        # Move to device
        audio = audio.to(self.device)
        
        # Pad if needed - pad the sample dimension (last dimension)
        original_length = audio.shape[-1]
        if audio.shape[-1] % self.chunk_size != 0:
            pad_length = self.chunk_size - (audio.shape[-1] % self.chunk_size)
            audio = F.pad(audio, (0, pad_length))
            
        # Encode
        encoded_frames = self.model.encode(audio)
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
        
        # Convert to numpy - codes shape is (batch, codebooks, frames)
        # We want (frames, codebooks)
        result = codes.squeeze(0).T.cpu().numpy()
            
        return result

    @torch.no_grad()
    def decode(self, codes: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Convert Encodec codebook indices back to audio.
        
        Args:
            codes: Array of shape [frames, N] containing codebook indices
            
        Returns:
            Audio array of shape [1, samples] with values in [-1, 1] range
        """
        self._ensure_model_loaded()
        
        # Convert to torch
        if isinstance(codes, np.ndarray):
            codes = torch.from_numpy(codes)
        
        # Move to device and unsqueeze batch
        codes = codes.to(self.device)
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)
        
        # Prepare format expected by encodec
        codes_list = [codes[:, :, i] for i in range(codes.shape[-1])]
        
        # Decode
        audio = self.model.decode([(codes_list, None)])[0]
        
        # Convert to numpy
        return audio.cpu().numpy() 