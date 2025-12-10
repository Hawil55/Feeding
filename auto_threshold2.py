#final autothreshold
import os
import cv2
from ultralytics import YOLO
from pathlib import Path
from collections import deque
import itertools
import pandas as pd

# --- Configuration ---
# REQUIRED: Update these paths to your specific files and directories
PELLET_MODEL_PATH = '/home/ahawil/pellet_detector_model/results4/train/weights/best.pt'
OPEN_MOUTH_MODEL_PATH = '/home/ahawil/open_mouth_model/data/Results2/train/weights/best.pt'
GROUND_TRUTH_CSV = '/home/ahawil/SEPIAA/DATA/June_sepiaa_recording_2025/auto_threshold/auto_threshold2.csv'# <--- REPLACE WITH YOUR CSV PATH
VIDEO_TO_VALIDATE = '/home/ahawil/SEPIAA/DATA/June_sepiaa_recording_2025/auto_threshold/auto_threshold2.mp4'# <--- REPLACE WITH THE FULL VIDEO PATH

# Output CSV file for the results
OUTPUT_CSV_FILE = 'parameter_optimization_results_threshold_NewMod2_5.csv' 

# NOTE: PRIORITY_AREA_HEIGHT_RATIO is now defined in PARAM_RANGES below to be optimized.
# If a pellet is FIRST detected in the top area, it bypasses MIN_PELLET_DETECTION_FRAMES.

# --- Define Parameter Ranges to Test ---
# The script will iterate through every combination of these values.
# Adjust the ranges based on your initial testing and desired granularity.
PARAM_RANGES = {
    'pellet_disappearance_frames': [5],
    'open_mouth_detection_window': [50, 100],
    'proximity_threshold': [100, 200, 300],
    'pellet_conf_threshold': [0.1, 0.5],
    'open_mouth_conf_threshold': [0.1, 0.5],
    'min_pellet_detection_frames': [1, 5],
    'priority_area_height_ratio': [0, 0.01], # NOW INCLUDED for optimization
}

# Define a tolerance for matching detected events to ground truth events.
# A detected event is considered a "match" if it's within this many frames of a ground truth event.
FRAME_TOLERANCE = 600

# --- Helper Functions ---
def calculate_distance(box1_center, box2_center):
    """Calculates Euclidean distance between two points."""
    return ((box1_center[0] - box2_center[0])**2 + (box1_center[1] - box2_center[1])**2)**0.5

def load_ground_truth(csv_path):
    """Loads manually validated feeding events from a CSV file."""
    try:
        df = pd.read_csv(csv_path)
        # Assuming the CSV has a 'Frame_Number' column
        ground_truth_frames = set(df['Frame_Number'].values)
        return ground_truth_frames
    except FileNotFoundError:
        print(f"Error: Ground truth CSV file not found at {csv_path}. Please check the path.")
        return None

def run_detection(video_path, params):
    """
    Runs the feeding event detection for a single video with a specific set of parameters,
    including ByteTrack and Priority Area logic.
    Returns a set of detected frame numbers.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Could not open video {video_path}. Skipping detection for this video.")
        return set()

    # Get video dimensions for Priority Area calculation
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Get the priority area ratio from the parameters being tested
    priority_area_height_ratio = params['priority_area_height_ratio']
    
    # Calculate priority area height
    priority_area_pixel_height = int(height * priority_area_height_ratio) 

    pellet_model = YOLO(PELLET_MODEL_PATH)
    open_mouth_model = YOLO(OPEN_MOUTH_MODEL_PATH)
    
    # pellet_tracks: {track_id: {'last_seen_frame': int, 'last_coords': (x, y), 'disappearance_start_frame': int or None, 'consecutive_frames': int, 'is_valid': bool}}
    pellet_tracks = {}
    open_mouth_history = deque()
    detected_feeding_events = set()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # --- Run Pellet Tracking with ByteTrack ---
        pellet_results = pellet_model.track(
            frame,
            persist=True,
            conf=params['pellet_conf_threshold'],
            tracker='bytetrack.yaml',
            verbose=False
        )

        current_pellet_track_ids = set()
        if pellet_results[0].boxes.id is not None:
            for det in pellet_results[0].boxes.data:
                # Data is [x1, y1, x2, y2, track_id, conf, cls]
                x1, y1, x2, y2, track_id, conf, cls = det.tolist()
                track_id = int(track_id)
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                
                current_pellet_track_ids.add(track_id)
                
                # Update or create the pellet track
                if track_id in pellet_tracks:
                    pellet_tracks[track_id]['last_seen_frame'] = frame_count
                    pellet_tracks[track_id]['last_coords'] = (center_x, center_y)
                    pellet_tracks[track_id]['disappearance_start_frame'] = None
                    pellet_tracks[track_id]['consecutive_frames'] += 1
                else:
                    # New pellet detected
                    
                    # --- Priority Area Check ---
                    # If the pellet is first detected in the top zone, it is instantly considered 'valid'.
                    is_priority = center_y < priority_area_pixel_height
                    # ---------------------------
                    
                    pellet_tracks[track_id] = {
                        'last_seen_frame': frame_count,
                        'last_coords': (center_x, center_y),
                        'disappearance_start_frame': None,
                        'consecutive_frames': 1,
                        'is_valid': is_priority # Use priority status for initial validation
                    }
                
                # Check for validity based on consecutive frames (only needed if not already valid)
                if not pellet_tracks[track_id]['is_valid'] and pellet_tracks[track_id]['consecutive_frames'] >= params['min_pellet_detection_frames']:
                    pellet_tracks[track_id]['is_valid'] = True

        # --- Run Open Mouth Detection ---
        open_mouth_results = open_mouth_model(frame, verbose=False)[0]
        current_open_mouths = []
        if hasattr(open_mouth_results, 'obb') and open_mouth_results.obb is not None:
            for i in range(len(open_mouth_results.obb.data)):
                conf = open_mouth_results.obb.conf[i].item()
                if conf >= params['open_mouth_conf_threshold']:
                    center_x, center_y, _, _, _ = open_mouth_results.obb.xywhr[i].tolist()
                    current_open_mouths.append({'center': (center_x, center_y), 'conf': conf})
        elif open_mouth_results.boxes is not None:
            for det in open_mouth_results.boxes.data:
                x1, y1, x2, y2, conf, cls = det.tolist()
                if conf >= params['open_mouth_conf_threshold']:
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                    current_open_mouths.append({'center': (center_x, center_y), 'conf': conf})

        open_mouth_history.append((frame_count, current_open_mouths))
        while open_mouth_history and open_mouth_history[0][0] < frame_count - params['open_mouth_detection_window']:
            open_mouth_history.popleft()

        # --- Pellet Tracking & Event Detection ---
        for pellet_id, track_info in list(pellet_tracks.items()):
            if pellet_id not in current_pellet_track_ids:
                # Pellet not found in current frame, start or continue disappearance timer
                if track_info['disappearance_start_frame'] is None:
                    pellet_tracks[pellet_id]['disappearance_start_frame'] = frame_count
                
                # Check for feeding event only if it's a valid pellet and has disappeared for enough frames
                if track_info['is_valid'] and (frame_count - track_info['disappearance_start_frame']) >= params['pellet_disappearance_frames']:
                    # Pellet disappeared, now check for open mouth in the window
                    pellet_disappearance_coords = track_info['last_coords']
                    search_start_frame = max(0, track_info['disappearance_start_frame'] - params['open_mouth_detection_window'])
                    
                    found_open_mouth = False
                    for hist_frame_num, hist_open_mouths in open_mouth_history:
                        if search_start_frame <= hist_frame_num <= frame_count:
                            for open_mouth_det in hist_open_mouths:
                                open_mouth_coords = open_mouth_det['center']
                                if calculate_distance(pellet_disappearance_coords, open_mouth_coords) < params['proximity_threshold']:
                                    detected_feeding_events.add(hist_frame_num) # Add the frame where the open mouth was detected
                                    found_open_mouth = True
                                    break
                            if found_open_mouth:
                                break
                    del pellet_tracks[pellet_id]
            elif not track_info['is_valid'] and track_info['disappearance_start_frame'] is not None and (frame_count - track_info['disappearance_start_frame']) > 100:
                del pellet_tracks[pellet_id]


    cap.release()
    return detected_feeding_events

def evaluate_performance(detected_events, ground_truth_events, frame_tolerance):
    """
    Compares detected events to ground truth events and calculates a score.
    Returns (true_positives, false_positives, false_negatives).
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Create a copy to track matched ground truth events
    unmatched_ground_truth = set(ground_truth_events)
    
    for detected_frame in detected_events:
        matched = False
        for gt_frame in list(unmatched_ground_truth):
            if abs(detected_frame - gt_frame) <= frame_tolerance:
                true_positives += 1
                unmatched_ground_truth.remove(gt_frame)
                matched = True
                break
        if not matched:
            false_positives += 1
            
    false_negatives = len(unmatched_ground_truth)
    
    return true_positives, false_positives, false_negatives

def main():
    """Main function to run the optimization process."""
    if not os.path.exists(PELLET_MODEL_PATH) or not os.path.exists(OPEN_MOUTH_MODEL_PATH):
        print("Error: One or both of the YOLO model paths are incorrect. Please verify.")
        return

    full_video_path = VIDEO_TO_VALIDATE
    if not os.path.exists(full_video_path):
        print(f"Error: The video file '{full_video_path}' was not found. Please check the path and filename.")
        return

    ground_truth_frames = load_ground_truth(GROUND_TRUTH_CSV)
    if ground_truth_frames is None:
        return
    
    # The total number of combinations is now increased due to the added parameter.
    num_combinations = len(list(itertools.product(*PARAM_RANGES.values())))
    print(f"Loaded {len(ground_truth_frames)} ground truth feeding events.")
    print(f"Testing a total of {num_combinations} parameter combinations.")
    
    all_results = []
    
    # Iterate through all parameter combinations
    keys, values = zip(*PARAM_RANGES.items())
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        print(f"Testing combination: {params}")
        
        detected_events = run_detection(full_video_path, params)
        tp, fp, fn = evaluate_performance(detected_events, ground_truth_frames, FRAME_TOLERANCE)
        
        all_results.append({
            'pellet_disappearance_frames': params['pellet_disappearance_frames'],
            'open_mouth_detection_window': params['open_mouth_detection_window'],
            'proximity_threshold': params['proximity_threshold'],
            'pellet_conf_threshold': params['pellet_conf_threshold'],
            'open_mouth_conf_threshold': params['open_mouth_conf_threshold'],
            'min_pellet_detection_frames': params['min_pellet_detection_frames'],
            'priority_area_ratio': params['priority_area_height_ratio'], # Updated to use the tested parameter
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'total_detected': len(detected_events)
        })

    # Find and print the best-performing set
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Define a scoring metric. A good one is to prioritize high TP and low FP/FN.
        # A simple score: TP - FP - FN
        results_df['score'] = results_df['true_positives'] - results_df['false_positives'] - results_df['false_negatives']
        best_run = results_df.sort_values(by='score', ascending=False).iloc[0]
        
        print("\n--- Optimization Complete ---")
        print("Top Recommended Parameters:")
        print(best_run[['pellet_disappearance_frames', 'open_mouth_detection_window', 'proximity_threshold',
                         'pellet_conf_threshold', 'open_mouth_conf_threshold', 'min_pellet_detection_frames',
                         'priority_area_ratio']])
        print("\nPerformance on the validation video:")
        print(f"  True Positives: {best_run['true_positives']}")
        print(f"  False Positives: {best_run['false_positives']}")
        print(f"  False Negatives: {best_run['false_negatives']}")
        
        # Save all results to a CSV file for detailed analysis
        results_df.to_csv(OUTPUT_CSV_FILE, index=False)
        print(f"\nAll results saved to {OUTPUT_CSV_FILE}")
    else:
        print("No results to display. Please check your configuration and data.")

if __name__ == "__main__":
    main()
