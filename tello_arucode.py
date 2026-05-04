import time
import csv
import threading
import math
import cv2
import numpy as np
import signal
import sys
from djitellopy import Tello
from datetime import datetime

# --- 1. CONFIGURATION ---
MARKER_SIZE_M = 0.20       # 200mm
MISSION_SPEED = 10        
Kp = 0.65                  
CENTER_TOLERANCE = 6.0     # 6cm
LOGGING_INTERVAL = 0.1
BLIND_MOVE_CM = 20.0      

ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
CAMERA_MATRIX = np.array([[252.0, 0, 160.0], [0, 252.0, 120.0], [0, 0, 1]], dtype=float)
DIST_COEFFS = np.zeros((5, 1))

# --- 2. INITIALIZE ---
drone = Tello()
drone.connect()
drone.streamoff()

drone.streamon()
time.sleep(2)  # Wait for stream to initialize
# Use downward camera (1) for detecting markers on the ground
drone.set_video_direction(1)  # Downward camera for ground markers
print("Camera set to DOWNWARD - make sure ArUco markers are on the ground below drone")


try:
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    aruco_params = cv2.aruco.DetectorParameters()
except AttributeError:
    aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT_ID)
    aruco_params = cv2.aruco.DetectorParameters_create()

logging_active = True
CURRENT_TAG = "idle"
target_lock_id = -1  # Which ID we are currently hunting
rel_pos_data = None  # (rx, ry, id)

# --- 3. VIDEO & TARGETED DETECTION THREAD ---
def video_worker():
    global rel_pos_data, logging_active
    container = drone.get_frame_read()
   
    detector = None
    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    # Give time for video stream to stabilize
    time.sleep(1)
    print("ArUco detection thread starting (no window - macOS compatibility)...")

    while logging_active:
        frame = container.frame
        if frame is None or frame.size == 0: 
            time.sleep(0.01)
            continue
       
        img = frame.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       
        if detector:
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
       
        found_target = False
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(img, corners, ids)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE_M, CAMERA_MATRIX, DIST_COEFFS)
           
            ids_list = ids.flatten().tolist()
           
            # --- FIX 1: Search specifically for the TARGET ID ---
            # Show ALL detected markers for debugging
            all_markers_text = f"Detected IDs: {ids_list}"
            cv2.putText(img, all_markers_text, (10, 60), 0, 0.6, (255, 255, 0), 2)
            
            if target_lock_id in ids_list:
                idx = ids_list.index(target_lock_id)
                # Convert to cm for display and math
                rx = tvecs[idx][0][0] * 100
                ry = tvecs[idx][0][1] * 100
                rel_pos_data = (rx, ry, target_lock_id)
                found_target = True
               
                cv2.putText(img, f"TARGET:{target_lock_id} FOUND! X:{round(rx,1)} Y:{round(ry,1)} cm",
                            (10,30), 0, 0.7, (0, 255, 0), 2)
                print(f"✓ Found target {target_lock_id} at X:{round(rx,1)} Y:{round(ry,1)} cm")
            else:
                rel_pos_data = None
                cv2.putText(img, f"SEARCHING FOR ID:{target_lock_id}", (10,30), 0, 0.7, (0, 0, 255), 2)
        else:
            rel_pos_data = None
            cv2.putText(img, "NO MARKERS DETECTED", (10,30), 0, 0.7, (0, 0, 255), 2)

        # No window display (macOS compatibility)
        # Just sleep briefly to avoid CPU overload
        time.sleep(0.03)

# --- 4. ADJUSTMENT FUNCTION (AXIS SWAPPED) ---
def adjustment_function(target_id, timeout=15):
    """
    Centers the drone over a specific target_id.
    Returns True if successful, False if timeout
    """
    global CURRENT_TAG, target_lock_id
    target_lock_id = target_id
    CURRENT_TAG = f"aligning_id_{target_id}"
    
    start_time = time.time()
    last_detection_time = None
   
    while logging_active:
        # Check timeout
        if time.time() - start_time > timeout:
            drone.send_rc_control(0, 0, 0, 0)
            print(f"  ⚠️  Timeout searching for marker {target_id}")
            return False
        
        if rel_pos_data is not None:
            rx, ry, found_id = rel_pos_data
            last_detection_time = time.time()
           
            # --- AXIS MAPPING FOR DOWNWARD CAMERA ---
            error_lr = rx   # Positive rx = marker is right, move right
            error_fb = ry   # Positive ry = marker is forward, move forward

            distance = math.sqrt(rx**2 + ry**2)
            if distance < CENTER_TOLERANCE:
                drone.send_rc_control(0, 0, 0, 0)
                print(f"  ✓ Locked on Marker {target_id}! (distance: {round(distance,1)}cm)")
                return True
           
            v_lr = int(np.clip(error_lr * Kp * 2, -MISSION_SPEED, MISSION_SPEED))
            v_fb = int(np.clip(error_fb * Kp * 2, -MISSION_SPEED, MISSION_SPEED))
           
            print(f"  Adjusting: rx={round(rx,1)} ry={round(ry,1)} dist={round(distance,1)}cm -> LR={v_lr} FB={v_fb}")
            drone.send_rc_control(v_lr, v_fb, 0, 0)
        else:
            # No marker detected - hover in place
            drone.send_rc_control(0, 0, 0, 0)
            if last_detection_time is None:
                print(f"  Searching for marker {target_id}... ({round(time.time()-start_time,1)}s)")
       
        time.sleep(0.05)
    
    return False

# --- 5. LOGGING ---
def log_data():
    filename = f"Corrected_Axis_Log_{datetime.now().strftime('%H%M%S')}.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Tag", "Time(s)", "Rel_X_cm", "Rel_Y_cm", "Target_ID", "Bat"])
        start_t = time.time()
        while logging_active:
            if rel_pos_data:
                state = drone.get_current_state()
                writer.writerow([CURRENT_TAG, round(time.time()-start_t, 2),
                                 round(rel_pos_data[0],2), round(rel_pos_data[1],2),
                                 target_lock_id, state.get('bat')])
            f.flush()
            time.sleep(LOGGING_INTERVAL)

# --- 6. MISSION ---
threading.Thread(target=log_data, daemon=True).start()
threading.Thread(target=video_worker, daemon=True).start()

try:
    print("\n=== STARTING MISSION ===")
    print("Make sure ArUco markers (IDs 0-7) are placed on the GROUND below the drone")
    print("Markers should be 20cm x 20cm, printed clearly\n")
    
    drone.takeoff()
    print("Takeoff complete, hovering...")
    #time.sleep(3)  # Allow IMU to stabilize and video to start
    
    drone.move_up(50)
    print("Moved up 50cm, waiting for video stream to stabilize...")
    #time.sleep(2)
    
    print("\nStarting marker detection...")
    print("Check the video window to see what the camera sees!\n")
    
    # Wait a bit to see if any markers are visible
    #time.sleep(2)
    
    # Marker Flow: Start at 0, make a square visiting corners (0,1,2,3) and midpoints (4,5,6,7)
    # Path: 0 → 4 → 1 → 5 → 2 → 6 → 3 → 7 → 0
    full_path = [0, 4, 1, 5, 2, 6, 3, 7, 0]
    corners = [0, 1, 2, 3]
    
    print("\n⏳ Waiting 3 seconds for video stream to stabilize...")
    time.sleep(3)

    for lap in range(1, 11):
        print(f"\n{'='*60}")
        print(f"LAP {lap}")
        print(f"{'='*60}")
        
        for i, target_id in enumerate(full_path):
            print(f"\n--- Step {i+1}/{len(full_path)}: Marker {target_id} ---")
            
            # First marker (0) - drone should already be over it after takeoff
            if i == 0:
                print(f"  Starting position - centering on marker {target_id}...")
                success = adjustment_function(target_id, timeout=10)
                if not success:
                    print(f"  ⚠️  Could not find starting marker {target_id}! Aborting mission.")
                    break
            else:
                # Move forward to find next marker (30cm = distance between markers)
                CURRENT_TAG = "forward"
                print(f"  Moving forward 30cm to search for marker {target_id}...")
                drone.move_forward(10)
                time.sleep(1.5)
               
                # Precision align to target
                print(f"  Attempting to lock onto marker {target_id}...")
                success = adjustment_function(target_id, timeout=15)
                if not success:
                    print(f"  ⚠️  Could not lock onto marker {target_id}, continuing...")
           
            # Rotate at corners
            if target_id in corners:
                CURRENT_TAG = "anticlockwise_90"
                print(f"  🔄 Corner marker - rotating 90° counter-clockwise")
                drone.rotate_counter_clockwise(90)
                time.sleep(1.5)

    drone.land()
except Exception as e:
    print(f"Error: {e}"); drone.land()
finally:
    logging_active = False
    drone.streamoff()
    cv2.destroyAllWindows()