#!/usr/bin/env python
"""
Computes relative pose between two cameras using shared ArUco marker observations.
Subscribes to two compressed image topics and their camera_info topics.
Publishes the 4x4 transform (target_cam -> phone_cam) as Float64MultiArray.

Usage:
    python camera_pose_publisher.py /cam_1/rgb/compressed --marker-size 0.15
"""

import argparse
import rospy
import numpy as np
import cv2
from cv2 import aruco
from sensor_msgs.msg import CompressedImage, CameraInfo
from std_msgs.msg import Float64MultiArray, MultiArrayDimension


class CameraPosePublisher:
    def __init__(self, target_topic, phone_topic, marker_size, aruco_dict_id):
        self.marker_size = marker_size

        # ArUco setup
        self.aruco_dict = aruco.getPredefinedDictionary(aruco_dict_id)
        self.aruco_params = aruco.DetectorParameters()

        # Camera intrinsics
        self.target_K = None
        self.target_D = np.zeros(5)
        self.phone_K = None
        self.phone_D = np.zeros(5)

        # Latest frames (always kept up to date)
        self.target_img = None
        self.phone_img = None

        # Visualization copies (updated each timer tick)
        self.target_vis = None
        self.phone_vis = None

        # Last computed relative transform (persists across frames)
        self.last_T = None

        # 3D marker corners (centered at origin, on XY plane)
        half = self.marker_size / 2
        self.marker_obj_points = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0]
        ], dtype=np.float32)

        # Axis length for visualization (in meters)
        self.axis_length = 0.1

        # Publisher
        self.transform_pub = rospy.Publisher(
            '/camera_relative_pose', Float64MultiArray, queue_size=1)

        # Derive camera_info topic from image topic
        # e.g. /cam_1/rgb/compressed -> /cam_1/rgb/camera_info
        target_base = target_topic.rsplit('/compressed', 1)[0]
        phone_base = phone_topic.rsplit('/compressed', 1)[0]

        # Subscribers
        rospy.Subscriber(target_topic, CompressedImage, self.target_image_cb)
        rospy.Subscriber(target_base + '/camera_info', CameraInfo, self.target_info_cb)
        rospy.Subscriber(phone_topic, CompressedImage, self.phone_image_cb)
        rospy.Subscriber(phone_base.rsplit('/', 1)[0] + '/camera_info',
                         CameraInfo, self.phone_info_cb)

        rospy.loginfo(f"Subscribed to target: {target_topic}")
        rospy.loginfo(f"Subscribed to phone:  {phone_topic}")
        rospy.loginfo(f"Marker size: {marker_size}m, ArUco dict: DICT_4X4_50")

        # Timer for compute + visualization at 10Hz
        rospy.Timer(rospy.Duration(0.1), self._timer_cb)

    def target_info_cb(self, msg):
        self.target_K = np.array(msg.K).reshape(3, 3)
        if len(msg.D) > 0:
            self.target_D = np.array(msg.D)

    def phone_info_cb(self, msg):
        self.phone_K = np.array(msg.K).reshape(3, 3)
        if len(msg.D) > 0:
            self.phone_D = np.array(msg.D)

    def target_image_cb(self, msg):
        self.target_img = self._decode(msg)

    def phone_image_cb(self, msg):
        self.phone_img = self._decode(msg)

    def _timer_cb(self, event):
        """Timer callback: always visualize, compute when possible."""
        # Update vis copies from latest frames
        if self.target_img is not None:
            self.target_vis = self.target_img.copy()
        if self.phone_img is not None:
            self.phone_vis = self.phone_img.copy()

        # Always show whatever we have
        if self.target_vis is None and self.phone_vis is None:
            return

        # Run detection + publish if we have everything
        if (self.target_vis is not None and self.phone_vis is not None
                and self.target_K is not None and self.phone_K is not None):
            self._compute()
        else:
            # Show status on available images
            if self.target_vis is not None and self.target_K is None:
                cv2.putText(self.target_vis, "Waiting for CameraInfo...", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            if self.phone_vis is not None and self.phone_K is None:
                cv2.putText(self.phone_vis, "Waiting for CameraInfo...", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        if self.target_vis is not None:
            cv2.imshow("Target Camera", self.target_vis)
        if self.phone_vis is not None:
            cv2.imshow("Phone Camera", self.phone_vis)
        cv2.waitKey(1)

    def _decode(self, msg):
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    def _detect_markers(self, img, K, D):
        """Detect ArUco markers. Returns (poses dict, corners, ids, rvecs, tvecs)."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is None:
            return {}, [], None, [], []

        poses = {}
        rvecs, tvecs = [], []
        for i, mid in enumerate(ids.flatten()):
            _, rvec, tvec = cv2.solvePnP(
                self.marker_obj_points,
                corners[i].reshape(-1, 2),
                K, D)
            R, _ = cv2.Rodrigues(rvec)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tvec.flatten()
            poses[int(mid)] = T
            rvecs.append(rvec)
            tvecs.append(tvec)
        return poses, corners, ids, rvecs, tvecs

    def _draw_marker_axes(self, vis, K, D, corners, ids, rvecs, tvecs):
        """Draw detected marker outlines and per-marker coordinate axes."""
        if ids is None:
            return
        aruco.drawDetectedMarkers(vis, corners, ids)
        L = self.axis_length
        for rvec, tvec in zip(rvecs, tvecs):
            cv2.drawFrameAxes(vis, K, D, rvec, tvec, L)

    def _compute(self):
        """Detect markers, draw axes, compute and publish relative transform."""
        # Detect markers in both
        target_poses, t_corners, t_ids, t_rvecs, t_tvecs = \
            self._detect_markers(self.target_img, self.target_K, self.target_D)
        phone_poses, p_corners, p_ids, p_rvecs, p_tvecs = \
            self._detect_markers(self.phone_img, self.phone_K, self.phone_D)

        # Draw marker axes on both views
        self._draw_marker_axes(self.target_vis, self.target_K, self.target_D,
                               t_corners, t_ids, t_rvecs, t_tvecs)
        # self._draw_marker_axes(self.phone_vis, self.phone_K, self.phone_D,
        #                        p_corners, p_ids, p_rvecs, p_tvecs)

        # Status text
        t_count = 0 if t_ids is None else len(t_ids)
        p_count = 0 if p_ids is None else len(p_ids)
        cv2.putText(self.target_vis, f"Markers: {t_count}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(self.phone_vis, f"Markers: {p_count}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Compute and publish relative transform if common markers found
        common = set(target_poses.keys()) & set(phone_poses.keys())
        if common:
            mid = min(common)
            T_target_marker = target_poses[mid]
            T_phone_marker = phone_poses[mid]
            T = T_phone_marker @ np.linalg.inv(T_target_marker)
            self.last_T = T

            # Publish
            msg = Float64MultiArray()
            msg.layout.dim = [
                MultiArrayDimension(label='rows', size=4, stride=16),
                MultiArrayDimension(label='cols', size=4, stride=4)
            ]
            msg.data = T.flatten().tolist()
            self.transform_pub.publish(msg)

            cv2.putText(self.phone_vis, f"Common marker: {mid}", (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            rospy.loginfo_throttle(2.0,
                f"Marker {mid} | t=[{T[0,3]:.3f}, {T[1,3]:.3f}, {T[2,3]:.3f}]")
        else:
            cv2.putText(self.phone_vis, "No common markers", (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw target marker axes on phone view using the relative transform
        if self.last_T is not None and target_poses:
            for mid, T_marker_in_target in target_poses.items():
                # Transform marker pose from target camera frame to phone camera frame
                T_marker_in_phone = self.last_T @ T_marker_in_target
                rvec, _ = cv2.Rodrigues(T_marker_in_phone[:3, :3])
                tvec = T_marker_in_phone[:3, 3]
                cv2.drawFrameAxes(self.phone_vis, self.phone_K, self.phone_D,
                                  rvec, tvec, self.axis_length)
                # Label the projected marker
                pts_2d, _ = cv2.projectPoints(
                    np.zeros((1, 3), dtype=np.float32), rvec, tvec,
                    self.phone_K, self.phone_D)
                pt = tuple(pts_2d.reshape(-1, 2)[0].astype(int))
                cv2.putText(self.phone_vis, f"M{mid}", (pt[0]+5, pt[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)


def main():
    parser = argparse.ArgumentParser(
        description='Compute relative pose between two cameras via ArUco markers')
    parser.add_argument('target_topic',
        help='Compressed image topic for target camera (e.g. /cam_1/rgb/compressed)')
    parser.add_argument('--phone-topic', default='/camera/image_raw/compressed',
        help='Compressed image topic for phone camera (default: /camera/image_raw/compressed)')
    parser.add_argument('--marker-size', type=float, default=0.15,
        help='ArUco marker size in meters (default: 0.15)')

    # Strip ROS remapping args before parsing
    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node('camera_pose_publisher', anonymous=True)
    CameraPosePublisher(
        args.target_topic, args.phone_topic,
        args.marker_size, aruco.DICT_4X4_50)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down camera pose publisher")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
