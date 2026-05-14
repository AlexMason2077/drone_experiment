"""
Tello Camera Viewer with Web Interface
This displays the Tello camera feed in your web browser instead of OpenCV window.
Works better on macOS where OpenCV windows don't always appear.
"""

import cv2
import time
import numpy as np
from djitellopy import Tello
from flask import Flask, Response
import threading

app = Flask(__name__)

# Global variables
drone = None
frame_read = None
current_frame = None
is_running = True

# ArUco setup
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
MARKER_SIZE_M = 0.20
CAMERA_MATRIX = np.array([[252.0, 0, 160.0], [0, 252.0, 120.0], [0, 0, 1]], dtype=float)
DIST_COEFFS = np.zeros((5, 1))

try:
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
except:
    aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT_ID)
    aruco_params = cv2.aruco.DetectorParameters_create()
    detector = None

detected_markers = set()

def process_frame():
    """Process frames from drone and detect ArUco markers"""
    global current_frame, detected_markers
    
    while is_running:
        if frame_read is None:
            time.sleep(0.1)
            continue
            
        frame = frame_read.frame
        if frame is None or frame.size == 0:
            time.sleep(0.01)
            continue
        
        img = frame.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers
        if detector:
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        
        # Draw markers and info
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(img, corners, ids)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, MARKER_SIZE_M, CAMERA_MATRIX, DIST_COEFFS
            )
            
            ids_list = ids.flatten().tolist()
            detected_markers.update(ids_list)
            
            # Draw info for each marker
            for i, marker_id in enumerate(ids_list):
                rx = tvecs[i][0][0] * 100  # cm
                ry = tvecs[i][0][1] * 100  # cm
                rz = tvecs[i][0][2] * 100  # cm (distance)
                
                cv2.drawFrameAxes(img, CAMERA_MATRIX, DIST_COEFFS, 
                                 rvecs[i], tvecs[i], MARKER_SIZE_M * 0.5)
                
                cv2.putText(img, f"ID:{marker_id} X:{round(rx,1)} Y:{round(ry,1)} Z:{round(rz,1)}cm",
                            (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (0, 255, 0), 2)
            
            status = f"DETECTED: {ids_list}"
            color = (0, 255, 0)
        else:
            status = "NO MARKERS DETECTED"
            color = (0, 0, 255)
        
        # Add status overlay
        cv2.putText(img, status, (10, img.shape[0] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(img, f"All detected: {sorted(list(detected_markers))}",
                    (10, img.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(img, "Camera: DOWNWARD", (10, img.shape[0] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        current_frame = img
        time.sleep(0.03)  # ~30 FPS

def generate_frames():
    """Generate frames for web streaming"""
    while is_running:
        if current_frame is None:
            time.sleep(0.1)
            continue
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', current_frame)
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Web page with video feed"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tello Camera Feed</title>
        <style>
            body {
                margin: 0;
                padding: 20px;
                background-color: #1a1a1a;
                color: white;
                font-family: Arial, sans-serif;
                text-align: center;
            }
            h1 {
                color: #4CAF50;
            }
            img {
                max-width: 95%;
                border: 3px solid #4CAF50;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.5);
            }
            .info {
                margin-top: 20px;
                padding: 15px;
                background-color: #2a2a2a;
                border-radius: 5px;
                display: inline-block;
            }
        </style>
    </head>
    <body>
        <h1>🚁 Tello Camera Feed - ArUco Detection</h1>
        <img src="/video_feed" alt="Tello Camera">
        <div class="info">
            <p><strong>Instructions:</strong></p>
            <p>Place ArUco markers (4x4, IDs 0-7) on the ground below the drone</p>
            <p>Markers should be 20cm x 20cm, printed clearly</p>
            <p>This feed shows the DOWNWARD camera view</p>
        </div>
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def main():
    global drone, frame_read, is_running
    
    print("=== Tello Camera Web Viewer ===\n")
    
    # Initialize drone
    drone = Tello()
    print("Connecting to Tello...")
    drone.connect()
    
    battery = drone.get_battery()
    print(f"Connected! Battery: {battery}%\n")
    
    if battery < 20:
        print("WARNING: Battery is low!")

        
    
    # drone.takeoff()
    # print("Takeoff successful! Hovering for 3 seconds...")
    # time.sleep(3)  # Give the drone time to stabilize
    
    

    # Start video stream
    print("Starting video stream...")
    drone.streamoff()
    time.sleep(1)
    drone.streamon()
    time.sleep(2)
    
    # Set to downward camera
    drone.set_video_direction(1)
    print("Camera: DOWNWARD\n")
    
    # Get frame reader
    frame_read = drone.get_frame_read()
    time.sleep(1)
    
    # Start frame processing thread
    processing_thread = threading.Thread(target=process_frame, daemon=True)
    processing_thread.start()
    
    print("=" * 60)
    print("✓ Camera feed is ready!")
    print("=" * 60)
    print("\nOpen your web browser and go to:")
    print("\n    http://localhost:5001\n")
    print("You will see the Tello camera feed in your browser!")
    print("=" * 60)
    print("\nPress Ctrl+C to stop\n")
    
    try:
        # Run Flask web server
        app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        # print("Attempting to land...")
        # drone.land()
        # print("Landing successful!")
        is_running = False
        drone.streamoff()
        drone.end()
        print("Done!")

if __name__ == "__main__":
    main()
