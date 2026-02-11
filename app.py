# app.py
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import roslibpy
import base64
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

# ROS connection setup
client = roslibpy.Ros(host='localhost', port=9091)
client.run()

# Create ROS topic publishers
talker = roslibpy.Topic(client, '/camera/image_raw/compressed', 'sensor_msgs/CompressedImage')
camera_info_pub = roslibpy.Topic(client, '/camera/camera_info', 'sensor_msgs/CameraInfo')

# Store camera info to publish alongside each frame
camera_info_msg = None

# Calibration state
CHECKERBOARD = (7, 10)  # internal corners: (rows-1, cols-1) for 8x11 board
SQUARE_SIZE = 0.015     # 15mm in meters
calib_img_points = []   # 2D points in image
calib_obj_points = []   # 3D points in world
calib_image_size = None

# Prepare object points for one checkerboard view
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

@app.route('/')
def index():
    """Serve the camera web interface"""
    return render_template('camera.html')

@app.route('/camera_info', methods=['POST'])
def receive_camera_info():
    """Receive camera intrinsics from iPhone"""
    global camera_info_msg
    data = request.json
    w = data['width']
    h = data['height']
    focal_mm = data.get('focalLength')  # May be None if browser doesn't expose it

    # Estimate focal length in pixels from focal length in mm
    # iPhone sensor width ~4.8mm (approximate for most models)
    if focal_mm:
        fx = focal_mm * w / 4.8
        fy = fx
    else:
        # Default estimate: ~70 degree horizontal FOV
        fx = w / (2 * 0.7)  # rough estimate
        fy = fx

    cx = w / 2.0
    cy = h / 2.0

    camera_info_msg = roslibpy.Message({
        'width': w,
        'height': h,
        'distortion_model': 'plumb_bob',
        'D': [0.0, 0.0, 0.0, 0.0, 0.0],
        'K': [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        'R': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        'P': [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    })

    focal_source = f"{focal_mm}mm (from browser)" if focal_mm else "estimated"
    print(f"Camera info: {w}x{h}, focal={focal_source}, fx={fx:.1f}px")
    return {'status': 'success', 'fx': fx, 'fy': fy}

@app.route('/calibrate_frame', methods=['POST'])
def calibrate_frame():
    """Capture a frame for calibration, detect checkerboard corners"""
    global calib_image_size
    data = request.json
    image_data = data['image'].split(',')[1]

    img_bytes = base64.b64decode(image_data)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    calib_image_size = gray.shape[::-1]  # (w, h)

    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)

    if found:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        calib_img_points.append(corners)
        calib_obj_points.append(objp)
        return {'status': 'success', 'count': len(calib_img_points)}
    else:
        return {'status': 'not_found', 'count': len(calib_img_points)}

@app.route('/calibrate_finish', methods=['POST'])
def calibrate_finish():
    """Run camera calibration with collected frames"""
    global camera_info_msg, calib_img_points, calib_obj_points

    if len(calib_img_points) < 5:
        return {'status': 'error', 'message': f'Need at least 5 frames, have {len(calib_img_points)}'}

    # Pure pinhole model: only estimate fx, fy, cx, cy (no distortion)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        calib_obj_points, calib_img_points, calib_image_size, None, None,
        flags=cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
            | cv2.CALIB_ZERO_TANGENT_DIST)

    fx, fy = mtx[0, 0], mtx[1, 1]
    cx, cy = mtx[0, 2], mtx[1, 2]
    w, h = calib_image_size

    camera_info_msg = roslibpy.Message({
        'width': w,
        'height': h,
        'distortion_model': 'plumb_bob',
        'D': [0.0, 0.0, 0.0, 0.0, 0.0],
        'K': [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        'R': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        'P': [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    })

    # Reset calibration data
    calib_img_points.clear()
    calib_obj_points.clear()

    print(f"Calibration done: reprojection error={ret:.4f}, fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    return {'status': 'success', 'error': ret, 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}

@app.route('/calibrate_reset', methods=['POST'])
def calibrate_reset():
    """Reset collected calibration frames"""
    calib_img_points.clear()
    calib_obj_points.clear()
    return {'status': 'success'}

@app.route('/upload', methods=['POST'])
def upload_frame():
    """Receive image frame from iPhone and publish to ROS"""
    data = request.json
    # Extract base64 image data (remove data:image/jpeg;base64, prefix)
    image_data = data['image'].split(',')[1]

    # Publish JPEG directly as CompressedImage (no decode/encode overhead)
    talker.publish(roslibpy.Message({
        'format': 'jpeg',
        'data': image_data
    }))

    # Publish camera info alongside each frame
    if camera_info_msg:
        camera_info_pub.publish(camera_info_msg)

    return {'status': 'success'}

if __name__ == '__main__':
    # Run with HTTPS (required for iPhone camera access)
    app.run(host='0.0.0.0', port=5000, ssl_context='adhoc')