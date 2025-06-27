from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from encodec import EncodecModel  # type: ignore[attr-defined]
from encodec.utils import convert_audio

from config import SAMPLE_RATE, ENCODEC_BANDWIDTH, get_logger


logger = get_logger(__name__)

def create_audio_tokenizer(bandwidth: float = ENCODEC_BANDWIDTH) -> 'AudioTokenizer':
    """Create audio tokenizer with specified settings."""
    tokenizer = AudioTokenizer()
    tokenizer._ensure_model_loaded(bandwidth)
    return tokenizer

class AudioTokenizer:
    """Handles conversion between raw audio and Encodec tokens.
    
    The Encodec model runs on GPU (if available) as per spec requirement.
    Results are returned as CPU numpy arrays for efficient caching."""
    
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        sample_rate: int = SAMPLE_RATE,
        chunk_size: int = SAMPLE_RATE * 30,  # 30 seconds
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.model = None  # Lazy load to avoid CUDA fork issues
        
    def _ensure_model_loaded(self, bandwidth=ENCODEC_BANDWIDTH):
        """Lazy load the model to avoid CUDA initialization in main process."""
        if self.model is None:
            logger.info(f"Loading Encodec model with bandwidth {bandwidth}")
            self.model = EncodecModel.encodec_model_24khz()
            self.model = self.model.to(self.device)
            self.model.set_target_bandwidth(bandwidth)
            self.model.eval()

    @torch.no_grad()
    def encode(self, audio: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Convert raw audio to Encodec codebook indices.
        
        IMPORTANT: Audio must be resampled to 24kHz by caller before encoding!
        This function does NOT handle resampling automatically.
        
        Args:
            audio: Audio array of shape [channels, samples] or [samples]
                  Values should be in [-1, 1] range and sampled at 24kHz
                  
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
            
        # Note: Audio resampling should be handled by caller if needed
        # We expect 24kHz audio as input
        
        # Add batch dimension for Encodec: (batch, channels, samples)
        audio = audio.unsqueeze(0)  # Now [1, 1, samples]
            
        # Move to device
        audio = audio.to(self.device)
        
        # Encode - encodec handles variable length audio
        encoded_frames = self.model.encode(audio)
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
        
        # Convert to numpy - codes shape is (batch, codebooks, frames)
        # We want (frames, codebooks)
        # Note: We return CPU numpy for caching/storage, but encoding happened on GPU
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
            codes = torch.from_numpy(codes).long()  # Ensure integer type
        
        # codes shape is [frames, codebooks]
        # Transpose to [codebooks, frames] then add batch dimension
        codes = codes.T.unsqueeze(0)  # Now [1, codebooks, frames]
        
        # Move to device
        codes = codes.to(self.device)
        
        # Decode - encodec expects list of [(codes, scales)]
        # where codes is [batch, codebooks, frames]
        audio = self.model.decode([(codes, None)])
        
        # Convert to numpy - shape is [batch, channels, samples]
        return audio[0].cpu().numpy() 