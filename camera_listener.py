#!/usr/bin/env python
"""
ROS node to receive iPhone camera stream from rosbridge
Subscribes to /camera/image_raw topic and displays using OpenCV
"""

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraListener:
    def __init__(self):
        # Initialize CV Bridge for ROS-OpenCV conversion
        self.bridge = CvBridge()
        
        # Create subscriber for camera topic
        self.sub = rospy.Subscriber('/camera/image_raw', Image, self.callback)
        
        rospy.loginfo("Camera listener started. Waiting for images...")
    
    def callback(self, msg):
        """Callback function when new image is received"""
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
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