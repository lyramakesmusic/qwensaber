"""
Comprehensive Beat Saber maps analysis tool.
Combines all analysis functionality in one place.
"""

import json
import zipfile
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple
import numpy as np
from tqdm import tqdm
import random
import argparse

from utils import check_v3_special_notes, check_mapping_extensions, check_map_features, normalize_position
from maps import parse_note


def analyze_single_map(zip_path: Path) -> Dict[str, Any]:
    """Analyze a single map file and return statistics."""
    stats = {
        'path': str(zip_path),
        'name': zip_path.name,
        'error': None,
        'features': {},
        'bpm': None,
        'duration': None,
        'note_count': 0,
        'nps': 0,
        'max_nps': 0,
        'density_percentiles': {},
        'position_distribution': Counter(),
        'direction_distribution': Counter(),
        'color_distribution': Counter(),
        'has_off_grid': False,
        'off_grid_count': 0,
        'chain_count': 0,
        'chain_lengths': [],
        'arc_count': 0,
        'bomb_count': 0,
        'obstacle_count': 0,
        'difficulties': [],
        'beat_subdivision_stats': {
            'min_note_gap_beats': None,
            'min_note_gap_subdivision': None,
            'faster_than_64th': False,
            'stacked_notes_count': 0,
            'consecutive_gaps': []
        }
    }
    
    try:
        # Check map features
        features = check_map_features(zip_path)
        stats['features'] = features
        stats['difficulties'] = features.get('difficulties', [])
        
        with zipfile.ZipFile(zip_path) as zf:
            # Get BPM from Info.dat
            info_data = None
            try:
                info_file = 'Info.dat' if 'Info.dat' in zf.namelist() else 'info.dat'
                with zf.open(info_file) as f:
                    info_data = json.load(f)
                    stats['bpm'] = float(info_data.get('_beatsPerMinute', 120))
            except:
                stats['bpm'] = 120  # Default
            
            # Analyze Expert or ExpertPlus difficulty
            map_file = None
            if 'ExpertPlusStandard.dat' in zf.namelist():
                map_file = 'ExpertPlusStandard.dat'
            elif 'ExpertStandard.dat' in zf.namelist():
                map_file = 'ExpertStandard.dat'
            
            if map_file:
                with zf.open(map_file) as f:
                    map_data = json.load(f)
                    
                    # Get notes
                    notes = []
                    is_v3 = 'version' in map_data and map_data.get('version', '').startswith('3.')
                    
                    if is_v3:
                        # V3 format
                        note_array = map_data.get('colorNotes', [])
                        for note in note_array:
                            # note = {'b': beat, 'x': x, 'y': y, 'c': color, 'd': direction}
                            notes.append({
                                'beat': float(note['b']),
                                'x': int(note['x']),
                                'y': int(note['y']),
                                'color': int(note['c']),
                                'direction': int(note['d'])
                            })
                        
                        # Count special notes
                        stats['chain_count'] = len(map_data.get('burstSliders', []))
                        stats['arc_count'] = len(map_data.get('sliders', []))
                        stats['bomb_count'] = len(map_data.get('bombNotes', []))
                        stats['obstacle_count'] = len(map_data.get('obstacles', []))
                        
                        # Analyze chains
                        for chain in map_data.get('burstSliders', []):
                            # Calculate chain length in beats
                            length = abs(float(chain.get('tb', 0)) - float(chain.get('b', 0)))
                            stats['chain_lengths'].append(length)
                    else:
                        # V2 format
                        note_array = map_data.get('_notes', [])
                        for note in note_array:
                            # Skip bombs
                            if note.get('_type', 0) >= 2:
                                stats['bomb_count'] += 1
                                continue
                            notes.append({
                                'beat': float(note['_time']),
                                'x': int(note['_lineIndex']),
                                'y': int(note['_lineLayer']),
                                'color': int(note['_type']),
                                'direction': int(note['_cutDirection'])
                            })
                        
                        stats['obstacle_count'] = len(map_data.get('_obstacles', []))
                    
                    # Analyze notes
                    if notes:
                        stats['note_count'] = len(notes)
                        
                        # Sort by beat
                        notes.sort(key=lambda n: n['beat'])
                        
                        # Duration and NPS
                        first_beat = notes[0]['beat']
                        last_beat = notes[-1]['beat']
                        duration_beats = last_beat - first_beat
                        if stats['bpm'] and duration_beats > 0:
                            duration_seconds = (duration_beats / stats['bpm']) * 60
                            stats['duration'] = duration_seconds
                            stats['nps'] = stats['note_count'] / duration_seconds
                        
                        # Check for off-grid notes (PPQ=48)
                        for note in notes:
                            beat = note['beat']
                            aligned = round(beat * 48) / 48
                            if abs(beat - aligned) > 0.001:
                                stats['has_off_grid'] = True
                                stats['off_grid_count'] += 1
                        
                        # Analyze beat subdivisions and timing gaps
                        if len(notes) > 1 and stats['bpm']:
                            # Calculate gaps between consecutive notes (overall)
                            gaps = []
                            stacked_count = 0
                            
                            for i in range(1, len(notes)):
                                gap = notes[i]['beat'] - notes[i-1]['beat']
                                if gap == 0:
                                    stacked_count += 1
                                elif gap > 0:
                                    gaps.append(gap)
                            
                            # Separate notes by color for per-hand analysis
                            red_notes = [n for n in notes if n['color'] == 0]
                            blue_notes = [n for n in notes if n['color'] == 1]
                            
                            # Calculate gaps for each hand separately
                            red_gaps = []
                            blue_gaps = []
                            
                            if len(red_notes) > 1:
                                red_notes.sort(key=lambda n: n['beat'])
                                for i in range(1, len(red_notes)):
                                    gap = red_notes[i]['beat'] - red_notes[i-1]['beat']
                                    if gap > 0:
                                        red_gaps.append(gap)
                            
                            if len(blue_notes) > 1:
                                blue_notes.sort(key=lambda n: n['beat'])
                                for i in range(1, len(blue_notes)):
                                    gap = blue_notes[i]['beat'] - blue_notes[i-1]['beat']
                                    if gap > 0:
                                        blue_gaps.append(gap)
                            
                            # Calculate minimum gaps
                            min_gap_overall = min(gaps) if gaps else None
                            min_gap_red = min(red_gaps) if red_gaps else None
                            min_gap_blue = min(blue_gaps) if blue_gaps else None
                            min_gap_per_hand = min([g for g in [min_gap_red, min_gap_blue] if g is not None], default=None)
                            
                            def gap_to_subdivision(gap):
                                if gap is None:
                                    return None
                                subdivision = 1.0 / gap
                                return f"1/{subdivision:.0f}"
                            
                            stats['beat_subdivision_stats']['stacked_notes_count'] = stacked_count
                            
                            # Store all subdivision info
                            if min_gap_overall:
                                stats['beat_subdivision_stats']['min_note_gap_beats'] = min_gap_overall
                                stats['beat_subdivision_stats']['min_note_gap_subdivision'] = gap_to_subdivision(min_gap_overall)
                                
                                # 64th note = 1/16 of a quarter note in beats  
                                sixty_fourth_note = 1.0 / 16.0
                                if min_gap_overall < sixty_fourth_note:
                                    stats['beat_subdivision_stats']['faster_than_64th'] = True
                            
                            # Per-hand analysis (more meaningful for Beat Saber)
                            stats['beat_subdivision_stats']['red_hand'] = {
                                'min_gap_beats': min_gap_red,
                                'min_gap_subdivision': gap_to_subdivision(min_gap_red),
                                'note_count': len(red_notes)
                            }
                            
                            stats['beat_subdivision_stats']['blue_hand'] = {
                                'min_gap_beats': min_gap_blue, 
                                'min_gap_subdivision': gap_to_subdivision(min_gap_blue),
                                'note_count': len(blue_notes)
                            }
                            
                            # Most meaningful metric: fastest per-hand subdivision
                            if min_gap_per_hand:
                                stats['beat_subdivision_stats']['min_per_hand_gap_beats'] = min_gap_per_hand
                                stats['beat_subdivision_stats']['min_per_hand_subdivision'] = gap_to_subdivision(min_gap_per_hand)
                                
                                # Check if either hand is faster than 64th notes
                                sixty_fourth_note = 1.0 / 16.0
                                stats['beat_subdivision_stats']['faster_than_64th_per_hand'] = min_gap_per_hand < sixty_fourth_note
                        
                        # Position/direction/color distribution
                        for note in notes:
                            # Convert ME positions to normal grid positions
                            x, y = normalize_position(note['x'], note['y'])
                            if x is not None and y is not None:  # Only count valid positions
                                pos = f"{x},{y}"
                                stats['position_distribution'][pos] += 1
                                stats['direction_distribution'][note['direction']] += 1
                                stats['color_distribution'][note['color']] += 1
                        
                        # Calculate density over time (1-second windows)
                        if stats['bpm'] and duration_seconds > 1:
                            window_size = stats['bpm'] / 60  # 1 second in beats
                            densities = []
                            
                            current_beat = first_beat
                            while current_beat < last_beat:
                                window_end = current_beat + window_size
                                count = sum(1 for n in notes if current_beat <= n['beat'] < window_end)
                                densities.append(count)
                                current_beat += window_size / 2  # 50% overlap
                            
                            if densities:
                                stats['max_nps'] = max(densities)
                                stats['density_percentiles'] = {
                                    'p50': np.percentile(densities, 50),
                                    'p75': np.percentile(densities, 75),
                                    'p90': np.percentile(densities, 90),
                                    'p95': np.percentile(densities, 95),
                                    'p99': np.percentile(densities, 99)
                                }
    
    except Exception as e:
        stats['error'] = f"{type(e).__name__}: {str(e)}"
    
    return stats


def analyze_directory(base_path: str, num_workers: int = 8) -> Dict[str, Any]:
    """Analyze all maps in directory and subdirectories."""
    base_path = Path(base_path)
    
    # Find all .zip files
    zip_files = list(base_path.rglob("*.zip"))
    print(f"Found {len(zip_files)} zip files to analyze")
    
    # Process maps in parallel
    all_stats = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_path = {executor.submit(analyze_single_map, zf): zf for zf in zip_files}
        
        # Process results with progress bar
        for future in tqdm(as_completed(future_to_path), total=len(zip_files), desc="Analyzing maps"):
            path = future_to_path[future]
            try:
                stats = future.result()
                all_stats.append(stats)
            except Exception as e:
                print(f"Error processing {path}: {e}")
    
    # Aggregate statistics
    aggregated = aggregate_stats(all_stats)
    return aggregated


def analyze_by_folders(base_path: str, num_workers: int = 8) -> Dict[str, Any]:
    """Analyze maps grouped by their parent folder."""
    base_path = Path(base_path)
    
    # Find all zip files
    zip_files = list(base_path.rglob("*.zip"))
    
    # Group by folder
    by_folder = defaultdict(list)
    for zf in zip_files:
        # Find which special folder it's in
        folder_name = 'root'
        for parent in zf.parents:
            if parent.name in ['v3_maps', 'noodle_maps', 'problematic_maps', 'low_rated_maps']:
                folder_name = parent.name
                break
        by_folder[folder_name].append(zf)
    
    print(f"Found maps distributed as:")
    for folder, files in sorted(by_folder.items()):
        print(f"  {folder}: {len(files)} maps")
    print()
    
    # Analyze each folder
    folder_stats = {}
    
    for folder_name, files in sorted(by_folder.items()):
        print(f"\nAnalyzing {folder_name} ({len(files)} maps)...")
        
        # Process maps in parallel
        folder_results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(analyze_single_map, zf): zf for zf in files}
            
            for future in tqdm(as_completed(futures), total=len(files), desc=f"Processing {folder_name}"):
                try:
                    stats = future.result()
                    if stats['error'] is None:
                        folder_results.append(stats)
                except:
                    pass
        
        # Aggregate stats for this folder
        if folder_results:
            folder_stats[folder_name] = aggregate_folder_stats(folder_results)
    
    return folder_stats

def analyze_custom_data(base_path: str, sample_size: int = 100) -> Counter:
    """Sample maps to see what's in customData fields."""
    base_path = Path(base_path)
    all_zips = list(base_path.rglob("*.zip"))
    
    if len(all_zips) > sample_size:
        sample = random.sample(all_zips, sample_size)
    else:
        sample = all_zips
    
    print(f"Analyzing customData in {len(sample)} random maps...")
    
    custom_keys = Counter()
    
    for zip_path in tqdm(sample, desc="Sampling"):
        try:
            with zipfile.ZipFile(zip_path) as zf:
                # Check Info.dat
                info_data = None
                try:
                    info_file = 'Info.dat' if 'Info.dat' in zf.namelist() else 'info.dat'
                    with zf.open(info_file) as f:
                        info_data = json.load(f)
                        if '_customData' in info_data:
                            custom_keys['INFO:_customData'] += 1
                            for key in info_data['_customData'].keys():
                                custom_keys[f'INFO:{key}'] += 1
                except:
                    pass
                
                # Check map file
                map_file = None
                if 'ExpertPlusStandard.dat' in zf.namelist():
                    map_file = 'ExpertPlusStandard.dat'
                elif 'ExpertStandard.dat' in zf.namelist():
                    map_file = 'ExpertStandard.dat'
                    
                if map_file:
                    with zf.open(map_file) as f:
                        map_data = json.load(f)
                        
                        # Check root customData
                        if 'customData' in map_data:
                            custom_keys['MAP:customData'] += 1
                            for key in map_data['customData'].keys():
                                custom_keys[f'MAP:{key}'] += 1
                        
                        # Check notes (sample first 10)
                        is_v3 = 'version' in map_data
                        notes = map_data.get('colorNotes' if is_v3 else '_notes', [])
                        
                        for note in notes[:10]:
                            if 'customData' in note or '_customData' in note:
                                cd = note.get('customData', note.get('_customData', {}))
                                for key in cd.keys():
                                    custom_keys[f'NOTE:{key}'] += 1
        except:
            pass
    
    return custom_keys


# Aggregation functions
def aggregate_stats(all_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate statistics from multiple maps."""
    aggregated = {
        'total_maps': len(all_stats),
        'successful_analyses': sum(1 for s in all_stats if s['error'] is None),
        'errors': sum(1 for s in all_stats if s['error'] is not None),
        'by_folder': defaultdict(int),
        'features': defaultdict(int),
        'extensions': defaultdict(int),
        'special_notes': defaultdict(int),
        'bpm_values': [],  # Temporary for calculation
        'duration_values': [],  # Temporary for calculation  
        'note_count_values': [],  # Temporary for calculation
        'nps_values': [],  # Temporary for calculation
        'max_nps_values': [],  # Temporary for calculation
        'off_grid_maps': 0,
        'off_grid_notes_total': 0,
        'chain_stats': {
            'maps_with_chains': 0,
            'total_chains': 0,
            'chain_lengths': []
        },
        'arc_stats': {
            'maps_with_arcs': 0,
            'total_arcs': 0
        },
        'position_usage': Counter(),
        'direction_usage': Counter(),
        'color_usage': Counter(),
        'density_thresholds': {
            'over_10_nps': 0,
            'over_15_nps': 0,
            'over_20_nps': 0,
            'over_25_nps': 0,
            'over_30_nps': 0,
            'over_35_nps': 0
        },
        'subdivision_stats': {
            'maps_with_stacked_notes': 0,
            'total_stacked_notes': 0,
            'maps_faster_than_64th': 0,
            'maps_faster_than_64th_per_hand': 0,
            'subdivision_counter': Counter(),  # Count frequencies only
            'per_hand_subdivision_counter': Counter(),  # Per-hand subdivision frequencies
            'min_gap_stats': {'count': 0, 'sum': 0, 'min': float('inf')},  # Running stats
            'min_per_hand_gap_stats': {'count': 0, 'sum': 0, 'min': float('inf')}  # Per-hand running stats
        }
    }
    
    # Process individual stats
    for stats in all_stats:
        if stats['error']:
            continue
            
        # Count by folder
        path_parts = Path(stats['path']).parts
        for i, part in enumerate(path_parts):
            if part in ['v3_maps', 'noodle_maps', 'problematic_maps', 'low_rated_maps']:
                aggregated['by_folder'][part] += 1
                break
        else:
            aggregated['by_folder']['root'] += 1
        
        # Features
        features = stats['features']
        if features.get('has_360'):
            aggregated['features']['360_degree'] += 1
        if features.get('has_90'):
            aggregated['features']['90_degree'] += 1
        if features.get('has_lightshow'):
            aggregated['features']['lightshow'] += 1
            
        # Special notes
        special = features.get('special_notes', {})
        for key, value in special.items():
            if value:
                aggregated['special_notes'][key] += 1
                
        # Extensions
        ext = features.get('extensions', {})
        for key, value in ext.items():
            if value:
                aggregated['extensions'][key] += 1
        
        # Basic stats - temporary storage for calculation
        if stats['bpm']:
            aggregated['bpm_values'].append(stats['bpm'])
        if stats['duration']:
            aggregated['duration_values'].append(stats['duration'])
        if stats['note_count']:
            aggregated['note_count_values'].append(stats['note_count'])
        if stats['nps']:
            aggregated['nps_values'].append(stats['nps'])
        if stats['max_nps']:
            aggregated['max_nps_values'].append(stats['max_nps'])
            
            # Density thresholds
            if stats['max_nps'] > 10:
                aggregated['density_thresholds']['over_10_nps'] += 1
            if stats['max_nps'] > 15:
                aggregated['density_thresholds']['over_15_nps'] += 1
            if stats['max_nps'] > 20:
                aggregated['density_thresholds']['over_20_nps'] += 1
            if stats['max_nps'] > 25:
                aggregated['density_thresholds']['over_25_nps'] += 1
            if stats['max_nps'] > 30:
                aggregated['density_thresholds']['over_30_nps'] += 1
            if stats['max_nps'] > 35:
                aggregated['density_thresholds']['over_35_nps'] += 1
        
        # Off-grid
        if stats['has_off_grid']:
            aggregated['off_grid_maps'] += 1
            aggregated['off_grid_notes_total'] += stats['off_grid_count']
        
        # Chains and arcs
        if stats['chain_count'] > 0:
            aggregated['chain_stats']['maps_with_chains'] += 1
            aggregated['chain_stats']['total_chains'] += stats['chain_count']
            aggregated['chain_stats']['chain_lengths'].extend(stats['chain_lengths'])
            
        if stats['arc_count'] > 0:
            aggregated['arc_stats']['maps_with_arcs'] += 1
            aggregated['arc_stats']['total_arcs'] += stats['arc_count']
        
        # Position/direction/color usage
        aggregated['position_usage'].update(stats['position_distribution'])
        aggregated['direction_usage'].update(stats['direction_distribution'])
        aggregated['color_usage'].update(stats['color_distribution'])
        
        # Beat subdivision analysis
        sub_stats = stats.get('beat_subdivision_stats', {})
        if sub_stats.get('stacked_notes_count', 0) > 0:
            aggregated['subdivision_stats']['maps_with_stacked_notes'] += 1
            aggregated['subdivision_stats']['total_stacked_notes'] += sub_stats['stacked_notes_count']
        
        if sub_stats.get('faster_than_64th', False):
            aggregated['subdivision_stats']['maps_faster_than_64th'] += 1
            
        if sub_stats.get('faster_than_64th_per_hand', False):
            aggregated['subdivision_stats']['maps_faster_than_64th_per_hand'] += 1
            
        if sub_stats.get('min_note_gap_subdivision'):
            aggregated['subdivision_stats']['subdivision_counter'][sub_stats['min_note_gap_subdivision']] += 1
            
        if sub_stats.get('min_per_hand_subdivision'):
            aggregated['subdivision_stats']['per_hand_subdivision_counter'][sub_stats['min_per_hand_subdivision']] += 1
            
        if sub_stats.get('min_note_gap_beats'):
            gap = sub_stats['min_note_gap_beats']
            gap_stats = aggregated['subdivision_stats']['min_gap_stats']
            gap_stats['count'] += 1
            gap_stats['sum'] += gap
            gap_stats['min'] = min(gap_stats['min'], gap)
            
        if sub_stats.get('min_per_hand_gap_beats'):
            gap = sub_stats['min_per_hand_gap_beats']
            gap_stats = aggregated['subdivision_stats']['min_per_hand_gap_stats']
            gap_stats['count'] += 1
            gap_stats['sum'] += gap
            gap_stats['min'] = min(gap_stats['min'], gap)
    
    # Calculate summary statistics
    def calc_stats(data):
        if not data:
            return {}
        return {
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'p25': np.percentile(data, 25),
            'p75': np.percentile(data, 75),
            'p90': np.percentile(data, 90),
            'p95': np.percentile(data, 95)
        }
    
    # Calculate final stats and remove temporary arrays
    aggregated['bpm_stats'] = calc_stats(aggregated['bpm_values'])
    aggregated['duration_stats'] = calc_stats(aggregated['duration_values'])
    aggregated['note_count_stats'] = calc_stats(aggregated['note_count_values'])
    aggregated['nps_stats'] = calc_stats(aggregated['nps_values'])
    aggregated['max_nps_stats'] = calc_stats(aggregated['max_nps_values'])
    
    # Remove temporary arrays to keep output clean
    del aggregated['bpm_values']
    del aggregated['duration_values'] 
    del aggregated['note_count_values']
    del aggregated['nps_values']
    del aggregated['max_nps_values']
    
    if aggregated['chain_stats']['chain_lengths']:
        aggregated['chain_stats']['length_stats'] = calc_stats(aggregated['chain_stats']['chain_lengths'])
        # Remove the raw chain lengths to keep output clean
        del aggregated['chain_stats']['chain_lengths']
    
    return aggregated


def aggregate_folder_stats(stats_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate statistics for a single folder."""
    agg = {
        'total': len(stats_list),
        'mapping_extensions': 0,
        'has_bpm_changes': 0,
        'has_chains': 0,
        'has_arcs': 0,
        'has_bombs': 0,
        'has_obstacles': 0,
        'off_grid_maps': 0,
        'bpm_values': [],
        'duration_values': [],
        'note_count_values': [],
        'nps_values': [],
        'max_nps_values': []
    }
    
    for stats in stats_list:
        # Count features
        if stats['features'].get('extensions', {}).get('has_mapping_extensions'):
            agg['mapping_extensions'] += 1
        if stats['features'].get('special_notes', {}).get('has_bpm_changes'):
            agg['has_bpm_changes'] += 1
        if stats['features'].get('special_notes', {}).get('has_chains'):
            agg['has_chains'] += 1
        if stats['features'].get('special_notes', {}).get('has_arcs'):
            agg['has_arcs'] += 1
        if stats['features'].get('special_notes', {}).get('has_bombs'):
            agg['has_bombs'] += 1
        if stats['features'].get('special_notes', {}).get('has_obstacles'):
            agg['has_obstacles'] += 1
        if stats['has_off_grid']:
            agg['off_grid_maps'] += 1
        
        # Collect values
        if stats['bpm']:
            agg['bpm_values'].append(stats['bpm'])
        if stats['duration']:
            agg['duration_values'].append(stats['duration'])
        if stats['note_count']:
            agg['note_count_values'].append(stats['note_count'])
        if stats['nps']:
            agg['nps_values'].append(stats['nps'])
        if stats['max_nps']:
            agg['max_nps_values'].append(stats['max_nps'])
    
    # Calculate percentages
    total = agg['total']
    agg['mapping_extensions_pct'] = agg['mapping_extensions'] / total * 100
    agg['has_bpm_changes_pct'] = agg['has_bpm_changes'] / total * 100
    agg['has_chains_pct'] = agg['has_chains'] / total * 100
    agg['has_arcs_pct'] = agg['has_arcs'] / total * 100
    agg['has_bombs_pct'] = agg['has_bombs'] / total * 100
    agg['has_obstacles_pct'] = agg['has_obstacles'] / total * 100
    agg['off_grid_maps_pct'] = agg['off_grid_maps'] / total * 100
    
    # Calculate statistics
    if agg['bpm_values']:
        agg['bpm_median'] = np.median(agg['bpm_values'])
        agg['bpm_mean'] = np.mean(agg['bpm_values'])
    if agg['duration_values']:
        agg['duration_median'] = np.median(agg['duration_values'])
        agg['duration_max'] = np.max(agg['duration_values'])
    if agg['note_count_values']:
        agg['note_count_median'] = np.median(agg['note_count_values'])
        agg['note_count_mean'] = np.mean(agg['note_count_values'])
    if agg['nps_values']:
        agg['nps_median'] = np.median(agg['nps_values'])
        agg['nps_p90'] = np.percentile(agg['nps_values'], 90)
    if agg['max_nps_values']:
        agg['max_nps_median'] = np.median(agg['max_nps_values'])
        agg['max_nps_p90'] = np.percentile(agg['max_nps_values'], 90)
        agg['max_nps_max'] = np.max(agg['max_nps_values'])
    
    return agg


# Printing functions
def print_report(stats: Dict[str, Any]):
    """Print formatted statistics report."""
    print("\n" + "="*80)
    print("BEAT SABER MAP COLLECTION ANALYSIS REPORT")
    print("="*80)
    
    print(f"\nTotal maps analyzed: {stats['total_maps']}")
    print(f"Successful: {stats['successful_analyses']}")
    print(f"Errors: {stats['errors']}")
    
    print("\n--- MAPS BY FOLDER ---")
    for folder, count in sorted(stats['by_folder'].items()):
        print(f"{folder}: {count}")
    
    print("\n--- SPECIAL FEATURES ---")
    for feature, count in sorted(stats['features'].items()):
        print(f"{feature}: {count} ({count/stats['successful_analyses']*100:.1f}%)")
    
    print("\n--- SPECIAL NOTES ---")
    for note_type, count in sorted(stats['special_notes'].items()):
        print(f"{note_type}: {count} ({count/stats['successful_analyses']*100:.1f}%)")
    
    print("\n--- MAPPING EXTENSIONS ---")
    for ext, count in sorted(stats['extensions'].items()):
        print(f"{ext}: {count} ({count/stats['successful_analyses']*100:.1f}%)")
    
    print("\n--- BPM STATISTICS ---")
    bpm = stats['bpm_stats']
    if bpm:
        print(f"Mean: {bpm['mean']:.1f}, Median: {bpm['median']:.1f}, Std: {bpm['std']:.1f}")
        print(f"Range: {bpm['min']:.0f} - {bpm['max']:.0f}")
    
    print("\n--- DURATION STATISTICS (seconds) ---")
    dur = stats['duration_stats']
    if dur:
        print(f"Mean: {dur['mean']:.1f}, Median: {dur['median']:.1f}")
        print(f"Range: {dur['min']:.1f} - {dur['max']:.1f}")
    
    print("\n--- NOTE COUNT STATISTICS ---")
    nc = stats['note_count_stats']
    if nc:
        print(f"Mean: {nc['mean']:.0f}, Median: {nc['median']:.0f}")
        print(f"Range: {nc['min']:.0f} - {nc['max']:.0f}")
    
    print("\n--- NPS (Notes Per Second) STATISTICS ---")
    nps = stats['nps_stats']
    if nps:
        print(f"Mean: {nps['mean']:.2f}, Median: {nps['median']:.2f}")
        print(f"P90: {nps['p90']:.2f}, P95: {nps['p95']:.2f}")
    
    print("\n--- MAX NPS (Peak Density) STATISTICS ---")
    max_nps = stats['max_nps_stats']
    if max_nps:
        print(f"Mean: {max_nps['mean']:.2f}, Median: {max_nps['median']:.2f}")
        print(f"P90: {max_nps['p90']:.2f}, P95: {max_nps['p95']:.2f}")
        print(f"Max: {max_nps['max']:.2f}")
    
    print("\n--- DENSITY THRESHOLDS ---")
    for threshold, count in sorted(stats['density_thresholds'].items()):
        nps_val = int(threshold.split('_')[1])
        print(f"Maps with peak >{nps_val} NPS: {count} ({count/stats['successful_analyses']*100:.1f}%)")
    
    print("\n--- TIMING ALIGNMENT ---")
    print(f"Maps with off-grid notes: {stats['off_grid_maps']} ({stats['off_grid_maps']/stats['successful_analyses']*100:.1f}%)")
    print(f"Total off-grid notes: {stats['off_grid_notes_total']}")
    
    print("\n--- CHAINS (BURST SLIDERS) ---")
    cs = stats['chain_stats']
    print(f"Maps with chains: {cs['maps_with_chains']} ({cs['maps_with_chains']/stats['successful_analyses']*100:.1f}%)")
    print(f"Total chains: {cs['total_chains']}")
    if 'length_stats' in cs:
        print(f"Chain length - Mean: {cs['length_stats']['mean']:.2f} beats, Max: {cs['length_stats']['max']:.2f} beats")
    
    print("\n--- ARCS (SLIDERS) ---")
    arc = stats['arc_stats']
    print(f"Maps with arcs: {arc['maps_with_arcs']} ({arc['maps_with_arcs']/stats['successful_analyses']*100:.1f}%)")
    print(f"Total arcs: {arc['total_arcs']}")
    
    print("\n--- BEAT SUBDIVISION ANALYSIS ---")
    sub = stats['subdivision_stats']
    print(f"Maps with stacked notes: {sub['maps_with_stacked_notes']} ({sub['maps_with_stacked_notes']/stats['successful_analyses']*100:.1f}%)")
    print(f"Total stacked notes: {sub['total_stacked_notes']}")
    print(f"Maps faster than 64th notes (overall): {sub['maps_faster_than_64th']} ({sub['maps_faster_than_64th']/stats['successful_analyses']*100:.1f}%)")
    print(f"Maps faster than 64th notes (per hand): {sub.get('maps_faster_than_64th_per_hand', 0)} ({sub.get('maps_faster_than_64th_per_hand', 0)/stats['successful_analyses']*100:.1f}%)")
    
    if sub['subdivision_counter']:
        print(f"Most common fastest subdivisions (overall timing):")
        for subdivision, count in sub['subdivision_counter'].most_common(10):
            print(f"  {subdivision}: {count} maps")
    
    if sub.get('per_hand_subdivision_counter'):
        print(f"Most common fastest subdivisions (per hand - more meaningful):")
        for subdivision, count in sub['per_hand_subdivision_counter'].most_common(10):
            print(f"  {subdivision}: {count} maps")
    
    gap_stats = sub['min_gap_stats']
    if gap_stats['count'] > 0:
        mean_gap = gap_stats['sum'] / gap_stats['count']
        min_gap = gap_stats['min']
        print(f"Min gap stats (overall) - Mean: {mean_gap:.4f} beats, Min: {min_gap:.4f} beats")
        print(f"Sample size: {gap_stats['count']} maps")
        
    per_hand_gap_stats = sub.get('min_per_hand_gap_stats', {})
    if per_hand_gap_stats.get('count', 0) > 0:
        mean_gap = per_hand_gap_stats['sum'] / per_hand_gap_stats['count']
        min_gap = per_hand_gap_stats['min']
        print(f"Min gap stats (per hand) - Mean: {mean_gap:.4f} beats, Min: {min_gap:.4f} beats")
        print(f"Sample size: {per_hand_gap_stats['count']} maps")
    
    print("\n" + "="*80)


def print_folder_comparison(folder_stats: Dict[str, Any]):
    """Print comparison table of folders."""
    folders = sorted(folder_stats.keys())
    
    print("\n" + "="*100)
    print("FOLDER COMPARISON")
    print("="*100)
    
    # Header
    print(f"{'Metric':<30}", end='')
    for folder in folders:
        print(f"{folder:>18}", end='')
    print()
    print("-"*100)
    
    # Metrics to compare
    metrics = [
        ('Total maps', 'total', '{:,}'),
        ('Mapping Extensions %', 'mapping_extensions_pct', '{:.1f}%'),
        ('BPM Changes %', 'has_bpm_changes_pct', '{:.1f}%'),
        ('Has Chains %', 'has_chains_pct', '{:.1f}%'),
        ('Has Arcs %', 'has_arcs_pct', '{:.1f}%'),
        ('Has Bombs %', 'has_bombs_pct', '{:.1f}%'),
        ('Has Obstacles %', 'has_obstacles_pct', '{:.1f}%'),
        ('Off-grid Notes %', 'off_grid_maps_pct', '{:.1f}%'),
        ('Median BPM', 'bpm_median', '{:.0f}'),
        ('Median Duration (s)', 'duration_median', '{:.1f}'),
        ('Max Duration (s)', 'duration_max', '{:.0f}'),
        ('Median Note Count', 'note_count_median', '{:.0f}'),
        ('Median NPS', 'nps_median', '{:.1f}'),
        ('90th %ile NPS', 'nps_p90', '{:.1f}'),
        ('Median Max NPS', 'max_nps_median', '{:.1f}'),
        ('90th %ile Max NPS', 'max_nps_p90', '{:.1f}'),
        ('Highest Max NPS', 'max_nps_max', '{:.0f}')
    ]
    
    for label, key, fmt in metrics:
        print(f"{label:<30}", end='')
        for folder in folders:
            value = folder_stats[folder].get(key, 'N/A')
            if value != 'N/A':
                print(f"{fmt.format(value):>18}", end='')
            else:
                print(f"{'N/A':>18}", end='')
        print()
    
    print("\n" + "="*100)


def print_extreme_maps(extreme_maps: List[Dict[str, Any]], limit: int = 10):
    """Print extreme duration maps."""
    print(f"\nFound {len(extreme_maps)} extreme duration maps:")
    print("="*80)
    
    for i, m in enumerate(extreme_maps[:limit]):
        print(f"{i+1}. {m['name']}")
        print(f"   Duration: {m['duration_minutes']:.1f} minutes ({m['duration_seconds']:.0f} seconds)")
        print(f"   BPM: {m['bpm']}")
        print(f"   Notes: {m['note_count']}")
        print(f"   NPS: {m['nps']:.2f}, Max NPS: {m['max_nps']:.2f}")
        print()


def print_custom_data_analysis(custom_keys: Counter, limit: int = 50):
    """Print customData field analysis."""
    print("\nMost common customData fields:")
    print("="*80)
    
    for key, count in custom_keys.most_common(limit):
        print(f"{key}: {count}")
    
    # Check for actual ME fields
    me_fields = ['animation', '_animation', 'position', '_position', 'localRotation', 
                 '_localRotation', 'color', '_color', 'noteJumpMovementSpeed', '_noteJumpMovementSpeed']
    
    print("\n\nMapping Extension specific fields found:")
    print("-"*80)
    found_any = False
    for field in me_fields:
        for prefix in ['NOTE:', 'EVENT:']:
            key = f"{prefix}{field}"
            if key in custom_keys:
                print(f"{key}: {custom_keys[key]}")
                found_any = True
    
    if not found_any:
        print("No mapping extension fields found in sample")


def save_results(data: Dict[str, Any], filename: str):
    """Save results to JSON file with numpy type conversion."""
    def convert_for_json(obj):
        """Convert numpy types to native Python types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        elif isinstance(obj, Counter):
            return dict(obj)
        else:
            return obj
    
    converted = convert_for_json(data)
    
    with open(filename, 'w') as f:
        json.dump(converted, f, indent=2)
    
    print(f"\nResults saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description='Analyze Beat Saber maps')
    parser.add_argument('maps_dir', help='Directory containing map files')
    parser.add_argument('--mode', choices=['full', 'folders', 'extreme', 'custom'], 
                       default='full', help='Analysis mode')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers')
    parser.add_argument('--min-duration', type=float, default=600, 
                       help='Minimum duration for extreme maps (seconds)')
    parser.add_argument('--sample-size', type=int, default=100, 
                       help='Sample size for custom data analysis')
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        print(f"Running full analysis on {args.maps_dir}")
        stats = analyze_directory(args.maps_dir, args.workers)
        print_report(stats)
        save_results(stats, 'map_analysis_results.json')
    
    elif args.mode == 'folders':
        print(f"Running folder comparison on {args.maps_dir}")
        folder_stats = analyze_by_folders(args.maps_dir, args.workers)
        print_folder_comparison(folder_stats)
        save_results(folder_stats, 'folder_analysis_results.json')
    
    elif args.mode == 'extreme':
        print(f"Finding extreme duration maps in {args.maps_dir}")
        extreme = find_extreme_maps(args.maps_dir, args.min_duration)
        print_extreme_maps(extreme)
        save_results({'extreme_maps': extreme}, 'extreme_maps.json')
    
    elif args.mode == 'custom':
        print(f"Analyzing customData fields in {args.maps_dir}")
        custom_keys = analyze_custom_data(args.maps_dir, args.sample_size)
        print_custom_data_analysis(custom_keys)
        save_results({'custom_data_fields': dict(custom_keys)}, 'custom_data_analysis.json')


if __name__ == "__main__":
    main() 