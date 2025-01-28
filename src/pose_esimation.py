import cv2
import numpy as np
import json


def load_hand_landmarks(json_path):
    """ Load hand landmark data from a JSON file. """
    with open(json_path, 'r') as f:
        landmarks = json.load(f)
    return landmarks


def get_3d_hand_points(landmarks, depth_image, camera_intrinsics):
    """
    Convert 2D hand landmarks into 3D points using depth information.

    Args:
        landmarks: List of 2D normalized hand keypoints [(x, y, z), ...].
        depth_image: Depth map corresponding to the RGB image.
        camera_intrinsics: Camera intrinsic parameters [fx, fy, cx, cy].

    Returns:
        3D hand keypoints in real-world coordinates.
    """
    fx, fy, cx, cy = camera_intrinsics
    h, w = depth_image.shape

    hand_3d_points = []
    for lm in landmarks[0]:
        x_px, y_px = int(lm[0] * w), int(lm[1] * h)
        depth = depth_image[y_px, x_px]

        X = (x_px - cx) * depth / fx
        Y = (y_px - cy) * depth / fy
        Z = depth

        hand_3d_points.append((X, Y, Z))

    return np.array(hand_3d_points, dtype=np.float32)


def estimate_camera_pose(hand_3d_points, camera_intrinsics, landmarks):
    """
    Estimate the camera pose relative to the hand using PnP.

    Args:
        hand_3d_points: 3D hand landmarks in world space.
        camera_intrinsics: Camera intrinsic matrix.
        landmarks: 2D hand landmarks.

    Returns:
        Rotation and translation vectors (rvec, tvec).
    """
    model_points = np.array([
        hand_3d_points[0],  # Wrist
        hand_3d_points[5],  # Base of index finger
        hand_3d_points[9],  # Base of middle finger
        hand_3d_points[13],  # Base of ring finger
        hand_3d_points[17]  # Base of pinky finger
    ], dtype=np.float32)

    image_points = np.array([
        [lm[0], lm[1]] for lm in landmarks[0]
    ], dtype=np.float32)[:5]

    success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_intrinsics, None)

    if success:
        return rvec, tvec
    else:
        return None, None


def process_hand_pose(json_path, depth_image_path, camera_intrinsics):
    """
    Full pipeline to estimate hand pose using depth data and landmarks.
    """
    landmarks = load_hand_landmarks(json_path)

    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

    hand_3d_points = get_3d_hand_points(landmarks, depth_image, camera_intrinsics)

    rvec, tvec = estimate_camera_pose(hand_3d_points, camera_intrinsics, landmarks)

    if rvec is not None and tvec is not None:
        print("Estimated Rotation Vector (rvec):", rvec.flatten())
        print("Estimated Translation Vector (tvec):", tvec.flatten())
    else:
        print("Failed to estimate camera pose.")

    return rvec, tvec