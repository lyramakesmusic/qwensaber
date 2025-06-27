"""
Beat Saber map converter.

Map format reference:
- b: beat number (multiply by 12 for our tick format)
- x: horizontal position (0-3)
- y: vertical position (0-2)
- c: color (0=red, 1=blue)
- d: cut direction (0-9)
- a: angle offset (ignored for now)

Note type encoding:
- 9 positions (3x3 grid) * 2 colors * 10 directions = 180 types
- Encoded as: position_idx * 20 + color * 10 + direction
  where position_idx = y * 3 + x

Each .zip archive should contain:
- song.egg: Audio file
- ExpertPlusStandard.dat: Map file (supports both old and new format)
- Info.dat: Metadata including BPM
"""

import json
import numpy as np
from pathlib import Path
import zipfile
import io
from typing import Union, List, Tuple
from torch.utils.data import Dataset
import soundfile as sf
import glob

def parse_note(note: dict) -> tuple[float, int, int, int, int]:
    """
    Parse a note from either old or new format.
    Returns (beat, x, y, color, direction)
    
    Old format may have different color values:
    - 0 = red
    - 1 = blue
    - 2,3 = bombs (ignored)
    """
    if 'b' in note:  # New format
        return (
            float(note['b']),
            int(note['x']),
            int(note['y']),
            int(note['c']),
            int(note['d'])
        )
    else:  # Old format
        color = int(note['_type'])
        # Skip bombs (type 2,3) by returning None
        if color >= 2:
            return None
        return (
            float(note['_time']),
            int(note['_lineIndex']),
            int(note['_lineLayer']),
            color,  # 0=red, 1=blue
            int(note['_cutDirection'])
        )

def load_map(path: Union[str, Path]) -> List[Tuple[float, int]]:
    """
    Load a Beat Saber map and convert to (time, type) pairs.
    Handles both new format (b,x,y,c,d) and old format (_time,_lineIndex,_lineLayer,_type,_cutDirection).
    
    Args:
        path: Path to map JSON file
        
    Returns:
        List of (time, type) pairs where:
            time is in ticks (beat * 12)
            type is encoded as (y * 3 + x) * 20 + color * 10 + direction
    """
    # Load and parse JSON
    with open(path) as f:
        data = json.load(f)
    
    notes = []
    
    # Try both new and old format note arrays
    note_array = data.get('colorNotes', [])  # New format
    if not note_array and '_notes' in data:  # Old format
        note_array = data['_notes']
    
    # Process each note
    for note in note_array:
        # Parse note data from either format
        parsed = parse_note(note)
        if parsed is None:  # Skip bombs
            continue
        beat, x, y, color, direction = parsed
        
        # Validate ranges
        assert 0 <= x <= 3, f"Invalid x position: {x}"
        assert 0 <= y <= 2, f"Invalid y position: {y}"
        assert 0 <= color <= 1, f"Invalid color: {color}"
        assert 0 <= direction <= 9, f"Invalid direction: {direction}"
        
        # Convert beat to ticks
        time = beat * 12
        
        # Encode note type
        position_idx = y * 3 + x
        note_type = position_idx * 20 + color * 10 + direction
        
        notes.append((time, note_type))
    
    # Sort by time
    notes.sort()
    
    return notes

class ZippedBeatSaberDataset(Dataset):
    """Dataset for loading Beat Saber maps from .zip archives."""
    
    def __init__(self, maps_dir: Union[str, Path]):
        """
        Initialize dataset from directory containing .zip files.
        
        Each .zip should contain:
            - song.egg: Audio file
            - ExpertPlusStandard.dat: Map file (supports both old and new format)
            - Info.dat: Metadata including BPM
        """
        self.maps_dir = Path(maps_dir)
        self.zip_files = list(self.maps_dir.glob("*.zip"))
        
        if not self.zip_files:
            raise ValueError(f"No .zip files found in {maps_dir}")
            
        print(f"Found {len(self.zip_files)} map archives")
        
    def __len__(self):
        return len(self.zip_files)
    
    def __getitem__(self, idx: int):
        zip_path = self.zip_files[idx]
        print(f"[ZippedBeatSaberDataset] Loading {zip_path.name}")
        
        with zipfile.ZipFile(zip_path) as zf:
            # Load Info.dat for BPM
            with zf.open('Info.dat') as f:
                info = json.load(f)
                bpm = float(info['_beatsPerMinute'])
            
            # Load map
            with zf.open('ExpertPlusStandard.dat') as f:
                map_data = json.load(f)
                notes = []
                
                # Try both new and old format note arrays
                note_array = map_data.get('colorNotes', [])  # New format
                if not note_array and '_notes' in map_data:  # Old format
                    note_array = map_data['_notes']
                
                for note in note_array:
                    # Parse note data from either format
                    parsed = parse_note(note)
                    if parsed is None:  # Skip bombs
                        continue
                    beat, x, y, color, direction = parsed
                    
                    # Validate ranges
                    assert 0 <= x <= 3, f"Invalid x position: {x}"
                    assert 0 <= y <= 2, f"Invalid y position: {y}"
                    assert 0 <= color <= 1, f"Invalid color: {color}"
                    assert 0 <= direction <= 9, f"Invalid direction: {direction}"
                    
                    # Convert beat to ticks
                    time = beat * 12
                    
                    # Encode note type
                    position_idx = y * 3 + x
                    note_type = position_idx * 20 + color * 10 + direction
                    
                    notes.append((time, note_type))
                
                notes.sort()  # Sort by time
                notes = np.array(notes, dtype=np.float32)
            
            # Load audio
            with zf.open('song.egg') as f:
                # Read audio data from zip stream
                audio_data = io.BytesIO(f.read())
                audio, sr = sf.read(audio_data)
                if audio.ndim == 2:  # Convert stereo to mono
                    audio = audio.mean(axis=1)
                
                # Ensure audio is in the correct format for the tokenizer
                if audio.ndim == 1:
                    audio = audio.reshape(1, -1)  # [channels, samples]
                
                return audio, notes, bpm

def convert_to_v3_format(notes, bpm):
    """Convert simple note format to Beat Saber v3 format."""
    # Decode type integer back to components
    def decode_note_type(type_int):
        position = type_int // 20
        remaining = type_int % 20
        color = remaining // 10
        direction = remaining % 10
        
        # Extract x,y from position
        y = position // 3
        x = position % 3
        
        return x, y, color, direction

    # Convert each note
    color_notes = []
    for note in notes:
        x, y, color, direction = decode_note_type(note["type"])
        
        color_notes.append({
            "b": float(note["time"]),  # Beat time
            "x": int(x),               # Lane (0-2)
            "y": int(y),               # Row (0-2)
            "a": 0,                    # Default angle
            "c": int(color),           # Color (0=red, 1=blue)
            "d": int(direction)        # Cut direction
        })

    # Create v3 format map
    v3_map = {
        "version": "3.3.0",
        "bpmEvents": [],
        "rotationEvents": [],
        "colorNotes": color_notes
    }
    
    return v3_map