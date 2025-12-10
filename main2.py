# Final one
import os
import cv2
from ultralytics import YOLO
from pathlib import Path
from collections import deque
import csv

# --- Configuration ---
# NOTE: These paths are specific to the original environment and will need adjustment
# if running elsewhere.
INPUT_DIR = '/home/ahawil/SEPIAA/DATA/June_sepiaa_recording_2025/2_camera_videos/test'
OUTPUT_DIR = '/home/ahawil/SEPIAA/DATA/June_sepiaa_recording_2025/finalAQ2feedwaste_newmodel_bestthreshold'
PELLET_MODEL_PATH = '/home/ahawil/pellet_detector_model/results4/train/weights/best.pt'
OPEN_MOUTH_MODEL_PATH = '/home/ahawil/open_mouth_model/data/Results2/train/weights/best.pt'

# Parameters for feeding event detection
PELLET_DISAPPEARANCE_FRAMES = 5 # Number of frames a pellet must be absent to be considered "disappeared"
OPEN_MOUTH_DETECTION_WINDOW = 50 # Number of frames to look for an an open mouth (before and during disappearance)
PROXIMITY_THRESHOLD = 200 # Maximum pixel distance between pellet disappearance and open mouth detection (adjust as needed)

# Confidence thresholds for detections
PELLET_CONF_THRESHOLD = 0.5
OPEN_MOUTH_CONF_THRESHOLD = 0.5

# Threshold for continuous pellet detection
MIN_PELLET_DETECTION_FRAMES = 5 # A pellet must be detected for this many consecutive frames to be considered valid

# NEW FEATURE: Priority Area Configuration
# If a pellet is FIRST detected in the top area of the frame, it bypasses MIN_PELLET_DETECTION_FRAMES.
PRIORITY_AREA_HEIGHT_RATIO = 0.0 

# NEW CONFIG: Waste Threshold Line
# Pellets must disappear below this vertical position (as a ratio of height) AND have no mouth detection to be counted as "Wasted".
WASTE_THRESHOLD_Y_RATIO = 0.95 

# --- Load Models ---
try:
    # Initialize the YOLO models using the specified local paths
    pellet_model = YOLO(PELLET_MODEL_PATH)
    open_mouth_model = YOLO(OPEN_MOUTH_MODEL_PATH)
    print("YOLO models loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO models: {e}")
    # Since the script cannot run without the models, we exit here.
    exit()

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Supported video extensions
VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv')

# --- Helper Function for Distance Calculation ---
def calculate_distance(box1_center, box2_center):
    """Calculates Euclidean distance between two points (Euclidean distance)."""
    return ((box1_center[0] - box2_center[0])**2 + (box1_center[1] - box2_center[1])**2)**0.5

# --- Main Processing Loop ---
for filename in os.listdir(INPUT_DIR):
    # Check if the file is a supported video format
    if not filename.lower().endswith(VIDEO_EXTS):
        continue

    video_path = os.path.join(INPUT_DIR, filename)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {filename}. Skipping.")
        continue

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Use 'mp4v' codec for compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    output_video_path = os.path.join(OUTPUT_DIR, f'{Path(filename).stem}_feeding_events.mp4')
    # Setup VideoWriter for the output video with overlaid detections
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Calculate priority area height based on the ratio
    priority_area_pixel_height = int(height * PRIORITY_AREA_HEIGHT_RATIO)
    
    # Calculate the pixel position of the new waste threshold line
    waste_threshold_y = int(height * WASTE_THRESHOLD_Y_RATIO)


    # Initialize list and header for CSV data for the current video
    csv_data = []
    csv_header = [
        'Feeding_Event_ID', 'Frame_Number',
        'Pellet_Disappearance_X', 'Pellet_Disappearance_Y',
        'Open_Mouth_X', 'Open_Mouth_Y',
        'Open_Mouth_Confidence', 'Distance_Pellet_Mouth',
        'Video_Filename',
        'Pellet_Conf_Threshold', 'Open_Mouth_Conf_Threshold',
        'Proximity_Threshold', 'Pellet_Disappearance_Frames',
        'Open_Mouth_Detection_Window', 'Min_Pellet_Detection_Frames',
        'Priority_Area_Ratio', 'Waste_Threshold_Y_Ratio' # Added new parameter
    ]
    feeding_event_id_counter = 0

    print(f'Processing: {filename}')

    # Dictionaries to store tracking information for pellets
    # track_id: {'last_seen_frame': int, 'last_coords': (x, y), 'disappearance_start_frame': int or None, 'consecutive_frames': int, 'is_valid': bool}
    pellet_tracks = {}

    # History of open mouth detections (deque acts as a sliding window)
    # Stores (frame_number, list_of_mouth_detections)
    open_mouth_history = deque()

    frame_count = 0
    feeding_event_count = 0
    feed_wasted_count = 0 # Counter for wasted feed
    total_pellets_detected = 0
    total_open_mouths_detected = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # --- Run Pellet Tracking with ByteTrack ---
        # The tracker is specified here. 'bytetrack.yaml' is an internal configuration for the YOLO tracking logic.
        pellet_results = pellet_model.track(
            frame,
            persist=True,
            conf=PELLET_CONF_THRESHOLD,
            tracker='bytetrack.yaml',
            verbose=False
        )

        current_pellet_track_ids = set()
        if pellet_results[0].boxes.id is not None:
            for det in pellet_results[0].boxes.data:
                # Det format: [x1, y1, x2, y2, track_id, conf, cls]
                x1, y1, x2, y2, track_id, conf, cls = det.tolist()
                track_id = int(track_id)
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                
                total_pellets_detected += 1
                current_pellet_track_ids.add(track_id)

                # Update or create the pellet track information
                if track_id in pellet_tracks:
                    pellet_tracks[track_id]['last_seen_frame'] = frame_count
                    pellet_tracks[track_id]['last_coords'] = (center_x, center_y)
                    pellet_tracks[track_id]['last_conf'] = conf
                    # Reset disappearance tracking since it was seen in this frame
                    pellet_tracks[track_id]['disappearance_start_frame'] = None
                    pellet_tracks[track_id]['consecutive_frames'] += 1
                else:
                    # New pellet detected - check priority area for instant validity
                    is_priority = center_y < priority_area_pixel_height
                    
                    pellet_tracks[track_id] = {
                        'last_seen_frame': frame_count,
                        'last_coords': (center_x, center_y),
                        'last_conf': conf,
                        'disappearance_start_frame': None,
                        'consecutive_frames': 1,
                        'is_valid': is_priority # Initialize validity based on priority area
                    }
                
                # Check for validity based on consecutive frames if not already valid
                if not pellet_tracks[track_id]['is_valid'] and pellet_tracks[track_id]['consecutive_frames'] >= MIN_PELLET_DETECTION_FRAMES:
                    pellet_tracks[track_id]['is_valid'] = True

        # --- Run Open Mouth Detection ---
        open_mouth_results = open_mouth_model(frame, verbose=False)[0]
        current_open_mouths = []
        
        # Handling both standard bounding boxes (boxes) and rotated bounding boxes (obb)
        if hasattr(open_mouth_results, 'obb') and open_mouth_results.obb is not None and len(open_mouth_results.obb.data) > 0:
            for i in range(len(open_mouth_results.obb.data)):
                conf = open_mouth_results.obb.conf[i].item()
                if conf >= OPEN_MOUTH_CONF_THRESHOLD:
                    # xywhr data for rotated boxes
                    center_x, center_y, width_obb, height_obb, angle = open_mouth_results.obb.xywhr[i].tolist()
                    # Approximate standard bbox for visualization (using a simple square)
                    x1 = center_x - width_obb / 2
                    y1 = center_y - height_obb / 2
                    x2 = center_x + width_obb / 2
                    y2 = center_y + height_obb / 2
                    current_open_mouths.append({'bbox': [x1, y1, x2, y2], 'center': (center_x, center_y), 'conf': conf})
        elif open_mouth_results.boxes is not None and len(open_mouth_results.boxes.data) > 0:
            for det in open_mouth_results.boxes.data:
                # x1, y1, x2, y2, conf, cls
                x1, y1, x2, y2, conf, cls = det.tolist()
                if conf >= OPEN_MOUTH_CONF_THRESHOLD:
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                    current_open_mouths.append({'bbox': [x1, y1, x2, y2], 'center': (center_x, center_y), 'conf': conf})
        
        total_open_mouths_detected_in_frame = len(current_open_mouths)
        total_open_mouths_detected += total_open_mouths_detected_in_frame
        
        # Update open mouth history (sliding window)
        open_mouth_history.append((frame_count, current_open_mouths))
        # Remove detections older than the detection window
        while open_mouth_history and open_mouth_history[0][0] < frame_count - OPEN_MOUTH_DETECTION_WINDOW:
            open_mouth_history.popleft()

        # --- Feeding Event Detection Logic ---
        # Check for pellets that have disappeared (i.e., not in current_pellet_track_ids)
        for pellet_id, track_info in list(pellet_tracks.items()):
            if pellet_id not in current_pellet_track_ids:
                # Pellet not seen in current frame
                if track_info['disappearance_start_frame'] is None:
                    pellet_tracks[pellet_id]['disappearance_start_frame'] = frame_count
                
                # Check if the pellet is valid and has been absent for the required number of frames
                if track_info['is_valid'] and (frame_count - track_info['disappearance_start_frame']) >= PELLET_DISAPPEARANCE_FRAMES:
                    # Disappearance confirmed, now check for open mouth in the detection window
                    feeding_detected = False
                    pellet_disappearance_coords = track_info['last_coords']
                    
                    # Define the search range for open mouth events:
                    # The window extends from (Disappearance start frame - Window size) up to the current frame.
                    search_start_frame = max(0, track_info['disappearance_start_frame'] - OPEN_MOUTH_DETECTION_WINDOW)
                    search_end_frame = frame_count

                    # Draw proximity circle (for visualization)
                    cv2.circle(frame, (int(pellet_disappearance_coords[0]), int(pellet_disappearance_coords[1])), PROXIMITY_THRESHOLD, (255, 0, 255), 2)
                    cv2.putText(frame, "Search Area", (int(pellet_disappearance_coords[0]) + 30, int(pellet_disappearance_coords[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

                    
                    # Iterate through the history of open mouth detections
                    for hist_frame_num, hist_open_mouths in open_mouth_history:
                        # Check if the detection falls within the time window
                        if search_start_frame <= hist_frame_num <= search_end_frame:
                            for open_mouth_det in hist_open_mouths:
                                open_mouth_coords = open_mouth_det['center']
                                # Check for spatial proximity
                                if calculate_distance(pellet_disappearance_coords, open_mouth_coords) < PROXIMITY_THRESHOLD:
                                    
                                    # --- COUNT IS PERFORMED ONCE HERE ---
                                    feeding_event_id_counter += 1
                                    feeding_event_count += 1
                                    print(f"Feeding event detected at frame {frame_count}! Total events: {feeding_event_count}")
                                    # Visualization of the confirmed event
                                    cv2.circle(frame, (int(pellet_disappearance_coords[0]), int(pellet_disappearance_coords[1])), 20, (0, 255, 255), -1)
                                    cv2.putText(frame, "FEEDING EVENT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)
                                    feeding_detected = True

                                    # Record data for CSV
                                    csv_data.append([
                                        feeding_event_id_counter,
                                        frame_count,
                                        round(pellet_disappearance_coords[0], 2), round(pellet_disappearance_coords[1], 2),
                                        round(open_mouth_coords[0], 2), round(open_mouth_coords[1], 2),
                                        round(open_mouth_det['conf'], 4),
                                        round(calculate_distance(pellet_disappearance_coords, open_mouth_coords), 2),
                                        filename,
                                        PELLET_CONF_THRESHOLD, OPEN_MOUTH_CONF_THRESHOLD,
                                        PROXIMITY_THRESHOLD, PELLET_DISAPPEARANCE_FRAMES,
                                        OPEN_MOUTH_DETECTION_WINDOW, MIN_PELLET_DETECTION_FRAMES,
                                        PRIORITY_AREA_HEIGHT_RATIO, WASTE_THRESHOLD_Y_RATIO
                                    ])
                                    break # Stop checking other open mouths for this pellet in the current history frame
                            
                            if feeding_detected:
                                break # Stop checking other history frames for this pellet (Event confirmed)
                    
                    # --- FIX IMPLEMENTED HERE: Check if feeding was detected and resolve the track ---
                    if feeding_detected:
                        # FIX: Remove the track immediately upon confirmed feeding event 
                        # to prevent re-counting in subsequent frames.
                        del pellet_tracks[pellet_id]
                    
                    else: # if not feeding_detected
                        # This block handles the case where the pellet disappeared, but no mouth was detected in the window.
                        
                        # --- NEW: Apply spatial condition for Wasted Feed ---
                        if pellet_disappearance_coords[1] >= waste_threshold_y:
                            print(f"INFO: Pellet {pellet_id} disappeared below the waste line at frame {pellet_tracks[pellet_id]['disappearance_start_frame']}, and no feeding event detected by frame {frame_count}. COUNTED AS WASTED.")
                            feed_wasted_count += 1
                            cv2.putText(frame, "FEED WASTED!", (width - 400, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3) # Red visualization
                        else:
                            print(f"INFO: Pellet {pellet_id} disappeared *above* the waste line at frame {pellet_tracks[pellet_id]['disappearance_start_frame']}. Not counted as wasted based on position.")
                        # ---------------------------------

                        # Remove the pellet track after the full check failed (whether successful or not)
                        del pellet_tracks[pellet_id]
            
            # Cleanup non-valid tracks (those that didn't meet MIN_PELLET_DETECTION_FRAMES) if they are lost for too long
            elif not track_info['is_valid'] and track_info['disappearance_start_frame'] is not None and (frame_count - track_info['disappearance_start_frame']) > 100:
                print(f"INFO: Removing non-valid pellet {pellet_id} track after long disappearance.")
                del pellet_tracks[pellet_id]

        # --- Drawing Detections and Info on Frame ---
        
        # Draw Priority Area boundary
        cv2.rectangle(frame, (0, 0), (int(width), int(priority_area_pixel_height)), (255, 100, 0), 2) # Blue/Orange outline
        cv2.putText(frame, "Priority Drop Zone (Instant Valid)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

        # Draw Waste Threshold Line (New Visualization)
        cv2.line(frame, (0, waste_threshold_y), (width, waste_threshold_y), (0, 0, 200), 2, cv2.LINE_AA) # Dark red line
        cv2.putText(frame, f'Waste Threshold (Y={WASTE_THRESHOLD_Y_RATIO})', (width - 350, waste_threshold_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)

        # Draw pellet detections from ByteTrack results
        if pellet_results[0].boxes.id is not None:
            for det in pellet_results[0].boxes.data:
                x1, y1, x2, y2, track_id, conf, cls = det.tolist()
                track_id = int(track_id)
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                # Change color based on whether the pellet is 'valid'
                is_valid = pellet_tracks.get(track_id, {}).get('is_valid', False)
                # Red for valid, orange for pending
                color = (0, 0, 255) if is_valid else (0, 165, 255)
                # Draw a filled circle at the pellet center
                cv2.circle(frame, (int(center_x), int(center_y)), 5, color, -1)
                cv2.putText(frame, f'P{track_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw open mouth detections
        for mouth_det in current_open_mouths:
            x1, y1, x2, y2 = [int(val) for val in mouth_det['bbox']]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green for open mouths
            cv2.putText(frame, 'Open Mouth', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display counts - Adjusted Y positions to include Feed Wasted
        cv2.putText(frame, f'Feeding Events: {feeding_event_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3) # Blue
        cv2.putText(frame, f'Feed Wasted: {feed_wasted_count}', (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3) # Red

        cv2.putText(frame, f'Pellets (current): {len(current_pellet_track_ids)}', (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f'Open Mouths (current): {total_open_mouths_detected_in_frame}', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Frame: {frame_count}', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display parameters/thresholds - shifted down slightly
        y_offset = 280 # Shifted from 270
        cv2.putText(frame, f'Pellet Conf Threshold: {PELLET_CONF_THRESHOLD}', (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(frame, f'Open Mouth Conf Threshold: {OPEN_MOUTH_CONF_THRESHOLD}', (50, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(frame, f'Proximity Threshold: {PROXIMITY_THRESHOLD}', (50, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(frame, f'Pellet Disappearance Frames: {PELLET_DISAPPEARANCE_FRAMES}', (50, y_offset + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(frame, f'Open Mouth Detection Window: {OPEN_MOUTH_DETECTION_WINDOW}', (50, y_offset + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(frame, f'Min Pellet Detection Frames: {MIN_PELLET_DETECTION_FRAMES}', (50, y_offset + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(frame, f'Priority Area Ratio: {PRIORITY_AREA_HEIGHT_RATIO}', (50, y_offset + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 1)


        out.write(frame)

    cap.release()
    out.release()
    print(f'Finished processing: {filename}. Total feeding events: {feeding_event_count}. Total wasted feed: {feed_wasted_count}') # Added to console output
    print(f'Saved output video to: {output_video_path}')

    # Save CSV file for the current video
    csv_output_path = os.path.join(OUTPUT_DIR, f'{Path(filename).stem}_feeding_events.csv')
    with open(csv_output_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(csv_header)
        csv_writer.writerows(csv_data)
    print(f'Saved feeding event data to CSV: {csv_output_path}')

print("All videos processed.")
