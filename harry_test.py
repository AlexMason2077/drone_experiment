"""
DJI Tello EDU - ArUco Marker Path Following
Path: 0 → 4 → 1 → 5 → 2 → 6 → 3 → 7
Drone starts on top of marker 0
"""

import time
import cv2
import numpy as np
from djitellopy import Tello

# ============ CONFIGURATION ============
MARKER_SIZE_M = 0.20  # 20cm markers
MARKER_SPACING_CM = 30  # Distance between markers
HOVER_HEIGHT_CM = 60  # Height above markers

# ArUco dictionary (adjust if using different dictionary)
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50

# Camera calibration (approximate for Tello)
CAMERA_MATRIX = np.array([[252.0, 0, 160.0], 
                          [0, 252.0, 120.0], 
                          [0, 0, 1]], dtype=float)
DIST_COEFFS = np.zeros((5, 1))

# Control parameters
Kp = 0.8  # Proportional gain for position control
MAX_SPEED = 15  # Max RC velocity
CENTER_TOLERANCE_CM = 8.0  # How close to center before "locked"
SEARCH_TIMEOUT_SEC = 10  # Max time to search for a marker

# Mission path
PATH = [0, 4, 1, 5, 2, 6, 3, 7]

# ============ INITIALIZATION ============
print("="*60)
print("🚁 TELLO ARUCO PATH FOLLOWING")
print("="*60)

drone = Tello()
print("\n📡 Connecting to Tello...")
drone.connect()
battery = drone.get_battery()
print(f"✓ Connected! Battery: {battery}%")

if battery < 20:
    print("⚠️  WARNING: Battery is low!")
    input("Press Enter to continue anyway, or Ctrl+C to abort...")

# Start video stream
print("\n📹 Starting video stream...")
drone.streamoff()
time.sleep(1)
drone.streamon()
time.sleep(2)

# Set to downward camera
drone.set_video_direction(1)
print("✓ Camera: DOWNWARD")

# Setup ArUco detector
try:
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    use_new_api = True
except:
    aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT_ID)
    aruco_params = cv2.aruco.DetectorParameters_create()
    detector = None
    use_new_api = False

frame_read = drone.get_frame_read()
time.sleep(1)

# ============ HELPER FUNCTIONS ============

def detect_markers(frame):
    """Detect ArUco markers in frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if use_new_api:
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    
    return corners, ids

def get_marker_position(corners, ids, target_id):
    """
    Get position of target marker relative to drone
    Returns: (x_cm, y_cm, distance_cm) or None if not found
    """
    if ids is None or target_id not in ids.flatten():
        return None
    
    idx = ids.flatten().tolist().index(target_id)
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, MARKER_SIZE_M, CAMERA_MATRIX, DIST_COEFFS
    )
    
    # Convert to cm
    x_cm = tvecs[idx][0][0] * 100  # Left/Right
    y_cm = tvecs[idx][0][1] * 100  # Forward/Back
    z_cm = tvecs[idx][0][2] * 100  # Distance (height)
    
    distance = np.sqrt(x_cm**2 + y_cm**2)
    
    return x_cm, y_cm, distance

def center_on_marker(target_id, timeout=SEARCH_TIMEOUT_SEC):
    """
    Center drone over target marker
    Returns: True if successful, False if timeout
    """
    print(f"  🎯 Centering on marker {target_id}...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        frame = frame_read.frame
        if frame is None or frame.size == 0:
            time.sleep(0.05)
            continue
        
        corners, ids = detect_markers(frame)
        
        # Check if we found the target
        pos = get_marker_position(corners, ids, target_id)
        
        if pos is None:
            # Marker not found - hover and wait
            drone.send_rc_control(0, 0, 0, 0)
            elapsed = time.time() - start_time
            if int(elapsed) % 2 == 0:  # Print every 2 seconds
                print(f"    Searching... ({round(elapsed,1)}s)")
            time.sleep(0.1)
            continue
        
        x_cm, y_cm, distance = pos
        
        # Check if centered
        if distance < CENTER_TOLERANCE_CM:
            drone.send_rc_control(0, 0, 0, 0)
            print(f"    ✓ Locked! (offset: {round(distance,1)}cm)")
            return True
        
        # Calculate control velocities
        # TESTING AXIS SWAP: For downward camera, axes might be rotated
        # Try: y -> left/right, x -> forward/back
        v_lr = int(np.clip(-y_cm * Kp, -MAX_SPEED, MAX_SPEED))
        v_fb = int(np.clip(-x_cm * Kp, -MAX_SPEED, MAX_SPEED))
        
        print(f"    Adjusting: x={round(x_cm,1)} y={round(y_cm,1)} dist={round(distance,1)}cm → LR={v_lr} FB={v_fb}")
        drone.send_rc_control(v_lr, v_fb, 0, 0)
        
        time.sleep(0.05)
    
    # Timeout
    drone.send_rc_control(0, 0, 0, 0)
    print(f"    ⚠️  Timeout after {timeout}s")
    return False

# ============ MAIN MISSION ============

try:
    print("\n" + "="*60)
    print("🚀 STARTING MISSION")
    print("="*60)
    print(f"Path: {' → '.join(map(str, PATH))}")
    print(f"Marker spacing: {MARKER_SPACING_CM}cm")
    print("="*60 + "\n")
    
    # Takeoff
    print("🛫 Taking off...")
    drone.takeoff()
    time.sleep(2)
    
    # Move to hover height
    print(f"⬆️  Moving up to {HOVER_HEIGHT_CM}cm...")
    drone.move_up(HOVER_HEIGHT_CM)
    time.sleep(2)
    
    print("\n⏳ Waiting for video stream to stabilize...")
    time.sleep(3)
    
    # Follow the path
    for i, target_id in enumerate(PATH):
        print(f"\n{'='*60}")
        print(f"STEP {i+1}/{len(PATH)}: Marker {target_id}")
        print(f"{'='*60}")
        
        if i == 0:
            # First marker - drone should already be over it
            print("  Starting position - verifying location...")
            success = center_on_marker(target_id, timeout=10)
            if not success:
                print(f"\n⚠️  ERROR: Cannot find starting marker {target_id}!")
                print("Make sure drone is positioned over marker 0")
                break
        else:
            # Move toward next marker
            print(f"  ➡️  Moving forward {MARKER_SPACING_CM}cm...")
            drone.move_forward(MARKER_SPACING_CM)
            time.sleep(1.5)
            
            # Center on marker
            success = center_on_marker(target_id, timeout=15)
            if not success:
                print(f"  ⚠️  Could not lock onto marker {target_id}")
                user_input = input("    Continue anyway? (y/n): ")
                if user_input.lower() != 'y':
                    break
        
        # Brief pause at each marker
        time.sleep(0.5)
    
    print("\n" + "="*60)
    print("✅ MISSION COMPLETE!")
    print("="*60)
    
    # Land
    print("\n🛬 Landing...")
    drone.land()
    
except KeyboardInterrupt:
    print("\n\n⚠️  Mission interrupted by user")
    drone.send_rc_control(0, 0, 0, 0)
    time.sleep(0.5)
    drone.land()
    
except Exception as e:
    print(f"\n\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    drone.send_rc_control(0, 0, 0, 0)
    time.sleep(0.5)
    drone.land()
    
finally:
    print("\n🔌 Cleaning up...")
    drone.streamoff()
    print("Done!")
