# app.py
from flask import Flask, render_template, request
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

# Create ROS topic publisher
talker = roslibpy.Topic(client, '/camera/image_raw', 'sensor_msgs/Image')

@app.route('/')
def index():
    """Serve the camera web interface"""
    return render_template('camera.html')

@app.route('/upload', methods=['POST'])
def upload_frame():
    """Receive image frame from iPhone and publish to ROS"""
    data = request.json
    # Extract base64 image data (remove data:image/jpeg;base64, prefix)
    image_data = data['image'].split(',')[1]
    
    # Convert base64 to OpenCV image
    img_bytes = base64.b64decode(image_data)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Resize for lower bandwidth ROS publishing
    img = cv2.resize(img, (320, 240))

    # Publish as ROS message
    height, width, channels = img.shape
    talker.publish(roslibpy.Message({
        'height': height,
        'width': width,
        'encoding': 'bgr8',
        'step': width * channels,
        'data': img.flatten().tolist()
    }))
    
    return {'status': 'success'}

if __name__ == '__main__':
    # Run with HTTPS (required for iPhone camera access)
    app.run(host='0.0.0.0', port=5000, ssl_context='adhoc')