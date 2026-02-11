#!/usr/bin/env python
"""
ROS node to receive iPhone camera stream from rosbridge
Subscribes to /camera/image_raw topic and displays using OpenCV
"""

import rospy
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np

class CameraListener:
    def __init__(self):
        # Create subscriber for compressed camera topic
        self.sub = rospy.Subscriber('/camera/image_raw/compressed', CompressedImage, self.callback)

        rospy.loginfo("Camera listener started. Waiting for images...")

    def callback(self, msg):
        """Callback function when new compressed image is received"""
        try:
            # Decode JPEG directly
            img_array = np.frombuffer(msg.data, dtype=np.uint8)
            cv_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # Display the image
            cv2.imshow("iPhone Camera Stream", cv_image)
            cv2.waitKey(1)

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

if __name__ == '__main__':
    # Initialize ROS node
    rospy.init_node('camera_listener', anonymous=True)
    
    # Create listener instance
    listener = CameraListener()
    
    try:
        # Keep node running
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down camera listener")
    
    # Clean up OpenCV windows
    cv2.destroyAllWindows()