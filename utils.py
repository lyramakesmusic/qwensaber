"""
Utility functions for Beat Saber map processing.

This module contains functions for:
- Checking for special v3 notes (arcs, chains)
- Detecting mapping extensions
- Aligning notes to PPQ grid
- Creating playable zip files
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Union, Tuple, Optional
import zipfile


def check_v3_special_notes(map_data: Dict[str, Any]) -> Dict[str, bool]:
    """Check for v3 special note types in map."""
    # map_data = parsed JSON from map file
    # returns = {'has_arcs': bool, 'has_chains': bool, ...}
    
    is_v3 = 'version' in map_data and map_data.get('version', '').startswith('3.')
    
    special_notes = {
        'has_arcs': False,
        'has_chains': False, 
        'has_bombs': False,
        'has_obstacles': False,
        'has_bpm_changes': False,
        'has_rotation_events': False,
        'has_boost_events': False
    }
    
    if not is_v3:
        # Old v2 format
        if '_notes' in map_data:
            for note in map_data.get('_notes', []):
                # note = {'_time': float, '_type': int, ...}
                if note.get('_type', 0) >= 2:
                    special_notes['has_bombs'] = True
                    break
        
        if '_obstacles' in map_data and len(map_data['_obstacles']) > 0:
            special_notes['has_obstacles'] = True
            
        if '_events' in map_data:
            for event in map_data['_events']:
                # event = {'_time': float, '_type': int, '_value': int}
                event_type = event.get('_type', -1)
                if event_type == 100:
                    special_notes['has_bpm_changes'] = True
                elif event_type == 14 or event_type == 15:
                    special_notes['has_rotation_events'] = True
                elif event_type == 5:
                    special_notes['has_boost_events'] = True
    else:
        # New v3 format
        if 'sliders' in map_data and len(map_data.get('sliders', [])) > 0:
            special_notes['has_arcs'] = True
            
        if 'burstSliders' in map_data and len(map_data.get('burstSliders', [])) > 0:
            special_notes['has_chains'] = True
            
        if 'bombNotes' in map_data and len(map_data.get('bombNotes', [])) > 0:
            special_notes['has_bombs'] = True
            
        if 'obstacles' in map_data and len(map_data.get('obstacles', [])) > 0:
            special_notes['has_obstacles'] = True
            
        if 'bpmEvents' in map_data and len(map_data.get('bpmEvents', [])) > 0:
            special_notes['has_bpm_changes'] = True
            
        if 'rotationEvents' in map_data and len(map_data.get('rotationEvents', [])) > 0:
            special_notes['has_rotation_events'] = True
            
        if 'basicBeatmapEvents' in map_data:
            for event in map_data.get('basicBeatmapEvents', []):
                # event = {'b': float, 'et': int, 'i': int, 'f': float}
                if event.get('et') == 5:
                    special_notes['has_boost_events'] = True
                    break
    
    return special_notes


def check_mapping_extensions(map_data: Dict[str, Any], info_data: Dict[str, Any] = None) -> Dict[str, bool]:
    """Check for mapping extension usage (Noodle, Chroma, etc)."""
    # map_data = parsed map JSON
    # info_data = parsed Info.dat JSON (optional)
    # returns = {'has_noodle': bool, 'has_chroma': bool, ...}
    
    extensions = {
        'has_noodle': False,
        'has_chroma': False,
        'has_mapping_extensions': False,
        'has_custom_data': False,
        'has_requirements': False
    }
    
    def has_custom_data_recursive(obj):
        """Recursively check for _customData fields"""
        if isinstance(obj, dict):
            if '_customData' in obj and obj['_customData']:
                return True
            if 'customData' in obj and obj['customData']:
                return True
            for value in obj.values():
                if has_custom_data_recursive(value):
                    return True
        elif isinstance(obj, list):
            for item in obj:
                if has_custom_data_recursive(item):
                    return True
        return False
    
    if has_custom_data_recursive(map_data):
        extensions['has_custom_data'] = True
        # Don't automatically mark as mapping extension!
    
    if info_data:
        # requirements/suggestions = ['Noodle Extensions', 'Chroma', ...]
        requirements = info_data.get('_customData', {}).get('_requirements', [])
        suggestions = info_data.get('_customData', {}).get('_suggestions', [])
        all_mods = requirements + suggestions
        
        if requirements:
            extensions['has_requirements'] = True
        
        for mod in all_mods:
            mod_lower = mod.lower()
            if 'noodle' in mod_lower:
                extensions['has_noodle'] = True
                extensions['has_mapping_extensions'] = True
            elif 'chroma' in mod_lower:
                extensions['has_chroma'] = True
                extensions['has_mapping_extensions'] = True
    
    if map_data.get('version', '').startswith('3.'):
        # V3 format
        notes = map_data.get('colorNotes', [])
        for note in notes:
            # note = {'b': float, 'x': int, 'y': int, 'c': int, 'd': int, 'customData': {...}}
            if 'customData' in note:
                custom = note['customData']
                if any(key in custom for key in ['animation', 'position', 'localRotation', 'noteJumpMovementSpeed']):
                    extensions['has_noodle'] = True
                    extensions['has_mapping_extensions'] = True
                if 'color' in custom:
                    extensions['has_chroma'] = True
                    extensions['has_mapping_extensions'] = True
    else:
        # V2 format
        notes = map_data.get('_notes', [])
        for note in notes:
            # note = {'_time': float, '_lineIndex': int, '_lineLayer': int, '_type': int, '_cutDirection': int, '_customData': {...}}
            if '_customData' in note:
                custom = note['_customData']
                if any(key in custom for key in ['_animation', '_position', '_localRotation', '_noteJumpMovementSpeed']):
                    extensions['has_noodle'] = True
                    extensions['has_mapping_extensions'] = True
                if '_color' in custom:
                    extensions['has_chroma'] = True
                    extensions['has_mapping_extensions'] = True
    
    return extensions


def check_map_features(zip_path: Union[str, Path]) -> Dict[str, Any]:
    """Check a zipped map for all special features and extensions."""
    # zip_path = path to .zip file
    # returns = {'special_notes': {...}, 'extensions': {...}, 'has_360': bool, ...}
    
    features = {
        'special_notes': {},
        'extensions': {},
        'has_360': False,
        'has_90': False,
        'has_lightshow': False,
        'difficulties': []
    }
    
    with zipfile.ZipFile(zip_path) as zf:
        # diff_files = ['ExpertStandard.dat', 'ExpertPlusStandard.dat', ...]
        diff_files = [f for f in zf.namelist() if f.endswith('.dat') and f != 'Info.dat' and f != 'info.dat']
        features['difficulties'] = diff_files
        
        info_data = None
        try:
            info_file = 'Info.dat' if 'Info.dat' in zf.namelist() else 'info.dat'
            with zf.open(info_file) as f:
                info_data = json.load(f)
        except:
            pass
        
        if info_data:
            # beatmap_set = {'_beatmapCharacteristicName': str, '_difficultyBeatmaps': [...]}
            for beatmap_set in info_data.get('_difficultyBeatmapSets', []):
                characteristic = beatmap_set.get('_beatmapCharacteristicName', '')
                if characteristic == '360Degree':
                    features['has_360'] = True
                elif characteristic == '90Degree':
                    features['has_90'] = True
                elif characteristic == 'Lightshow':
                    features['has_lightshow'] = True
        
        if 'ExpertPlusStandard.dat' in zf.namelist():
            with zf.open('ExpertPlusStandard.dat') as f:
                map_data = json.load(f)
                features['special_notes'] = check_v3_special_notes(map_data)
                features['extensions'] = check_mapping_extensions(map_data, info_data)
        
    return features


def get_tail_direction(x1: int, y1: int, x2: int, y2: int, start_direction: int) -> int:
    """Calculate direction for tail note based on movement and start direction."""
    # (x1, y1) = start position
    # (x2, y2) = tail position
    # start_direction = direction of start note (0-9)
    # returns = direction (0-9)
    
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        return 8  # square dot (no movement)
    
    # Check if start direction is diagonal (4,5,6,7)
    if start_direction in [4, 5, 6, 7]:
        # Start is diagonal → use square EXCEPT for perfect diagonal movement
        if abs(dx) == abs(dy):
            return 9  # diamond dot (matching diagonal)
        else:
            return 8  # square dot (contrast)
    
    # Check if start direction is straight (0,1,2,3)
    elif start_direction in [0, 1, 2, 3]:
        # Start is straight → use diamond EXCEPT for perfect straight in same direction
        if (start_direction in [0, 1] and dx == 0) or (start_direction in [2, 3] and dy == 0):
            return 8  # square dot (matching straight)
        else:
            return 9  # diamond dot (contrast)
    
    # Start direction is dot (8,9) - use movement-based logic
    else:
        if abs(dx) == abs(dy):
            return 9  # diamond dot
        else:
            return 8  # square dot


def replace_chains(chains: List[Dict[str, Any]], notes: List[Tuple[float, int]]) -> List[Tuple[float, int]]:
    """Convert burst sliders to regular notes and merge with existing notes."""
    # chains = [{'b': float, 'x': int, 'y': int, 'c': int, 'd': int, 'tb': float, 'tx': int, 'ty': int, ...}, ...]
    # notes = [(beat, note_type), ...]
    # returns = [(beat, note_type), ...] sorted by beat
    
    # Import here to avoid circular dependency
    from maps import encode_note
    
    # Start with existing notes
    result = list(notes)
    
    # Add tail notes for each chain
    for chain in chains:
        # Calculate direction for tail based on movement
        tail_direction = get_tail_direction(
            chain['x'], chain['y'],
            chain['tx'], chain['ty'],
            chain['d']
        )
        
        # Create tail note
        tail_type = encode_note(
            chain['tx'], 
            chain['ty'], 
            chain['c'], 
            tail_direction
        )
        
        # Add tail note at tail beat
        result.append((float(chain['tb']), tail_type))
    
    # Sort by beat time
    result.sort(key=lambda x: x[0])
    
    return result 

def align_notes_to_ppq(notes: List[Tuple[float, int]], ppq: int = 48) -> List[Tuple[float, int]]:
    """Align all notes in map to PPQ grid."""
    aligned = []
    for beat, note_type in notes:
        aligned_beat = round(beat * ppq) / ppq
        aligned.append((aligned_beat, note_type))
    
    aligned.sort(key=lambda x: x[0])
    
    return aligned


def normalize_position(x: Union[int, float], y: Union[int, float]) -> Tuple[Optional[int], Optional[int]]:
    """Convert ME positions to normal Beat Saber grid, skip animation coordinates."""
    # x, y = raw note positions (int thousandths or float)
    if isinstance(x, int) and abs(x) >= 1000:
        x_float = x / 1000.0
    else:
        x_float = float(x)
    
    if isinstance(y, int) and abs(y) >= 1000:
        y_float = y / 1000.0  
    else:
        y_float = float(y)
    
    # x_float, y_float = ME coordinate space (-1.0 to 4.0, 0.0 to 3.0)
    if abs(x_float - round(x_float)) <= 0.1:
        x_float = round(x_float)
    if abs(y_float - round(y_float)) <= 0.1:
        y_float = round(y_float)
    
    # Skip fractional positions (animation keyframes, poodle coordinates)
    if x_float != int(x_float) or y_float != int(y_float):
        return None, None
    
    # x_int, y_int = integer grid positions in ME space
    x_int = int(x_float)
    y_int = int(y_float)
    
    # x_normal = Beat Saber column (0-3)
    if x_int == -1:
        x_normal = 0
    elif x_int in [0, 1, 2, 3]:
        x_normal = x_int
    elif x_int == 4:
        x_normal = 3
    else:
        return None, None
    
    # y_normal = Beat Saber row (0-2)
    if y_int in [0, 1, 2]:
        y_normal = y_int
    elif y_int == 3:
        y_normal = 2
    else:
        return None, None
    
    # (x_normal, y_normal) = standard Beat Saber grid coordinates
    return x_normal, y_normal


def has_unroundable_notes(map_data: Dict[str, Any]) -> bool:
    """Check if map contains fractional ME coordinates that can't be normalized."""
    # map_data = parsed JSON from .dat file
    
    # extensions = {'has_mapping_extensions': bool, ...}
    extensions = check_mapping_extensions(map_data)
    if not extensions.get('has_mapping_extensions', False):
        return False  # Normal maps only have standard coordinates
    
    is_v3 = 'version' in map_data and map_data.get('version', '').startswith('3.')
    
    if is_v3:
        # notes = [{'b': float, 'x': int, 'y': int, 'c': int, 'd': int}, ...]
        notes = map_data.get('colorNotes', [])
        for note in notes:
            # (x, y) = raw note position from map data
            x, y = note.get('x', 0), note.get('y', 0)
            # normalized = (x_normal, y_normal) or (None, None) if unroundable
            normalized = normalize_position(x, y)
            if normalized == (None, None):
                return True
    else:
        # notes = [{'_time': float, '_lineIndex': int, '_lineLayer': int, '_type': int, '_cutDirection': int}, ...]
        notes = map_data.get('_notes', [])
        for note in notes:
            if note.get('_type', 0) >= 2:  # Skip bombs
                continue
            # (x, y) = raw note position from map data
            x, y = note.get('_lineIndex', 0), note.get('_lineLayer', 0)
            # normalized = (x_normal, y_normal) or (None, None) if unroundable
            normalized = normalize_position(x, y)
            if normalized == (None, None):
                return True
    
    return False 