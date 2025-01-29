import cv2
import numpy as np
import json


def load_hand_landmarks(json_path):
    """ Load hand landmark data from a JSON file. """
    with open(json_path, 'r') as f:
        landmarks = json.load(f)
    return landmarks


def load_depth_from_log(log_path):
    """ Reads a depth log file and converts it into a NumPy array. """
    with open(log_path, 'r') as f:
        depth_lines = f.readlines()

    depth_values = [list(map(float, line.strip().split(','))) for line in depth_lines]
    depth_matrix = np.array(depth_values, dtype=np.float32)

    print(f"Loaded depth matrix from {log_path}, shape: {depth_matrix.shape}")
    return depth_matrix


def get_3d_hand_points(landmarks, depth_matrix, camera_intrinsics):
    """
    Convert 2D hand landmarks into 3D points using depth information.

    Args:
        landmarks: List of 2D normalized hand keypoints [(x, y, z), ...].
        depth_matrix: Depth map loaded from log file.
        camera_intrinsics: Camera intrinsic parameters [fx, fy, cx, cy].

    Returns:
        3D hand keypoints in real-world coordinates.
    """
    fx = camera_intrinsics[0, 0]  # Focal length in x
    fy = camera_intrinsics[1, 1]  # Focal length in y
    cx = camera_intrinsics[0, 2]  # Principal point x
    cy = camera_intrinsics[1, 2]  # Principal point y
    h, w = depth_matrix.shape

    hand_3d_points = []
    for lm in landmarks[0]:
        x_px, y_px = int(lm[0] * w), int(lm[1] * h)

        if x_px < 0 or x_px >= w or y_px < 0 or y_px >= h:
            print(f"Warning: Skipping out-of-bounds landmark at ({x_px}, {y_px})")
            continue

        depth = depth_matrix[y_px, x_px]

        if depth <= 0:
            print(f"Warning: Invalid depth at ({x_px}, {y_px}), skipping point.")
            continue

        depth_mm = depth * 1000  # Convert meters to millimeters
        print(f"Depth at ({x_px}, {y_px}): {depth_mm:.2f} mm")
        X = (x_px - cx) * depth_mm / fx
        Y = (y_px - cy) * depth_mm / fy
        Z = depth_mm

        hand_3d_points.append((X, Y, Z))

    if len(hand_3d_points) < 5:
        print("Error: Not enough valid 3D points from depth map.")
        return None

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

    selected_indices = [0, 5, 9, 13, 17, 8]

    model_points = hand_3d_points[selected_indices]
    image_points = np.array(
        [[landmarks[0][i][0], landmarks[0][i][1]] for i in selected_indices],
        dtype=np.float32
    )

    camera_intrinsics = np.array(camera_intrinsics, dtype=np.float32)

    success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_intrinsics, None)

    if success:
        return rvec, tvec
    else:
        print("Error: PnP estimation failed.")
        return None, None


def process_hand_pose(json_path, depth_log_path, camera_intrinsics):
    """
    Full pipeline to estimate hand pose using depth logs and landmarks.
    """
    landmarks = load_hand_landmarks(json_path)

    depth_matrix = load_depth_from_log(depth_log_path)

    hand_3d_points = get_3d_hand_points(landmarks, depth_matrix, camera_intrinsics)

    if hand_3d_points is None:
        print("Failed to generate 3D hand points.")
        return None, None

    rvec, tvec = estimate_camera_pose(hand_3d_points, camera_intrinsics, landmarks)

    if rvec is not None and tvec is not None:
        print("Estimated Rotation Vector (rvec):", rvec.flatten())
        print("Estimated Translation Vector (tvec):", tvec.flatten())
    else:
        print("Pose estimation failed.")

    return rvec, tvec


if __name__ == "__main__":
    json_path = "/Users/denys.koval/University/VirtualRingTryOn/data/results/original_0_landmarks.json"
    depth_log_path = "/Users/denys.koval/University/VirtualRingTryOn/data/images/depth_logs_0.txt"

    camera_intrinsics = np.array([[1464, 0, 960],  # fx, 0, cx
                                  [0, 1464, 720],  # 0, fy, cy
                                  [0, 0, 1]], dtype=np.float32)

    rvec, tvec = process_hand_pose(json_path, depth_log_path, camera_intrinsics)