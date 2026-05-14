from djitellopy import Tello
import cv2
import cv2.aruco as aruco
import numpy as np
import time

# Tello setup
tello = Tello()
tello.connect()
print(tello.get_battery())
tello.streamon()
tello.takeoff()
time.sleep(2)
tello.send_rc_control(0, 0, 0, 0)

# ArUco + camera (calibrate Tello camera for accuracy!)
marker_length = 0.12  # meters, measure your printed size
camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])  # approximate; calibrate!
dist_coeffs = np.zeros((4, 1))

aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)  # assuming 5x5 from image
parameters = aruco.DetectorParameters_create()

# Route: clockwise square around the 8 markers
route = [0, 1, 2, 3, 4, 5, 6, 7]
route_idx = 0

# Target pose per ID: [x, y, z] in marker frame (hover centered 60cm up)
targets = {
    0: np.array([0.0, 0.0, 0.60]),
    1: np.array([0.0, 0.0, 0.60]),
    2: np.array([0.0, 0.0, 0.60]),
    3: np.array([0.0, 0.0, 0.60]),
    4: np.array([0.0, 0.0, 0.60]),
    5: np.array([0.0, 0.0, 0.60]),
    6: np.array([0.0, 0.0, 0.60]),
    7: np.array([0.0, 0.0, 0.60]),
}

state = "SEARCH"
frames_at_target = 0
Kp_lin = 35  # tune gains
max_vel = 50

def velocity_from_error(err):
    vx = np.clip(Kp_lin * err[0] * 100, -max_vel, max_vel)  # right/left
    vy = np.clip(Kp_lin * err[2] * 100, -max_vel, max_vel)  # fwd/bkwd (z in cam)
    vz = np.clip(-Kp_lin * err[1] * 100, -max_vel, max_vel) # up/down
    return int(vx), int(vy), int(vz)

def yaw_from_center(corners, img_shape):
    cx = corners[:, 0].mean()
    img_w = img_shape[1]
    yaw_err = (cx - img_w / 2) / (img_w / 2)
    return int(np.clip(-40 * yaw_err, -40, 40))

try:
    while True:
        frame = tello.get_frame_read().frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        current_id = route[route_idx]

        if state == "SEARCH":
            if ids is not None and current_id in ids.flatten():
                state = "ALIGN"
                frames_at_target = 0
                print(f"Found {current_id}, aligning...")
            else:
                # Gentle clockwise search yaw + slight forward drift
                tello.send_rc_control(0, 5, 0, 20)
                continue

        elif state == "ALIGN":
            if ids is None or current_id not in ids.flatten():
                state = "SEARCH"
                print(f"Lost {current_id}, searching...")
                continue

            # Get this marker's pose
            idx = np.where(ids.flatten() == current_id)[0][0]
            mk_corners = corners[idx]
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers([mk_corners], marker_length, camera_matrix, dist_coeffs)
            rvec, tvec = rvecs[0], tvecs[0]
            drone_pos = tvec.reshape(3)  # position in marker frame

            target = targets[current_id]
            err = target - drone_pos

            vx, vy, vz = velocity_from_error(err)
            yaw = yaw_from_center(mk_corners, frame.shape)
            tello.send_rc_control(vx, vy, vz, yaw)

            if np.linalg.norm(err) < 0.06:  # tolerance
                frames_at_target += 1
                if frames_at_target > 30:  # ~1 sec at 30fps
                    print(f"Done with {current_id}, next {route[(route_idx+1)%8]}")
                    route_idx = (route_idx + 1) % len(route)
                    state = "SEARCH"
                    time.sleep(1)  # brief pause
            else:
                frames_at_target = 0

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(2)
    tello.land()
    tello.streamoff()
