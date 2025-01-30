import cv2
import numpy as np
import open3d as o3d
import json

def load_depth_from_log(log_path, target_size):
    """ Reads a depth log file and converts it into a NumPy array. """
    with open(log_path, 'r') as f:
        depth_lines = f.readlines()

    depth_values = [list(map(float, line.strip().split(','))) for line in depth_lines]
    depth_matrix = np.array(depth_values, dtype=np.float32)

    depth_matrix = cv2.resize(depth_matrix, target_size, interpolation=cv2.INTER_NEAREST)

    return depth_matrix * 1000.0  # Convert mm to meters

def load_rgbd_data(rgb_path, depth_log_path, landmarks_path):
    rgb_image = cv2.imread(rgb_path)
    if rgb_image is None:
        raise FileNotFoundError(f"Error: Could not load RGB image at {rgb_path}")

    original_height, original_width, _ = rgb_image.shape
    depth_map = load_depth_from_log(depth_log_path, (original_width, original_height))

    with open(landmarks_path, 'r') as f:
        hand_landmarks = json.load(f)

    return rgb_image, depth_map, hand_landmarks

def convert_landmarks_to_3d(landmarks, depth_map, intrinsics):
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    height, width = depth_map.shape
    hand_3d_points = []

    for lm in landmarks[0]:
        x_px, y_px = int(lm[0] * width), int(lm[1] * height)
        if 0 <= x_px < width and 0 <= y_px < height:
            depth = depth_map[y_px, x_px]
            if depth > 0:
                X = (x_px - cx) * depth / fx
                Y = (y_px - cy) * depth / fy
                Z = depth
                hand_3d_points.append((X, Y, Z))
            else:
                hand_3d_points.append(None)
        else:
            hand_3d_points.append(None)

    return np.array([p for p in hand_3d_points if p is not None], dtype=np.float32)

def compute_ring_transformation(hand_3d_points):
    if len(hand_3d_points) < 10:
        return None

    keypoint_5 = np.array(hand_3d_points[5])
    keypoint_6 = np.array(hand_3d_points[6])
    keypoint_9 = np.array(hand_3d_points[9])

    ring_center = (keypoint_5 + keypoint_6) / 2

    x_axis = keypoint_6 - keypoint_5
    x_axis /= np.linalg.norm(x_axis)

    y_axis = keypoint_9 - keypoint_5
    y_axis /= np.linalg.norm(y_axis)

    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)

    y_axis = np.cross(z_axis, x_axis)

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, 0] = y_axis
    transformation_matrix[:3, 1] = z_axis
    transformation_matrix[:3, 2] = x_axis
    transformation_matrix[:3, 3] = ring_center

    return transformation_matrix

def get_transformation_matrix(rgb_path, depth_log_path, landmarks_path, camera_intrinsics):
    rgb_image, depth_map, hand_landmarks = load_rgbd_data(rgb_path, depth_log_path, landmarks_path)
    hand_3d_points = convert_landmarks_to_3d(hand_landmarks, depth_map, camera_intrinsics)
    transformation_matrix = compute_ring_transformation(hand_3d_points)
    return transformation_matrix

if __name__ == "__main__":
    rgb_path = "/Users/denys.koval/University/VirtualRingTryOn/data/images/original_0.png"
    depth_log_path = "/Users/denys.koval/University/VirtualRingTryOn/data/images/depth_logs_0.txt"
    landmarks_path = "/Users/denys.koval/University/VirtualRingTryOn/data/results/original_0_landmarks.json"
    ring_model_path = "/Users/denys.koval/University/VirtualRingTryOn/models/ring.obj"

    camera_intrinsics = np.array([[1464, 0, 960],
                                  [0, 1464, 720],
                                  [0, 0, 1]], dtype=np.float32)

    transformation_matrix = get_transformation_matrix(rgb_path, depth_log_path, landmarks_path, camera_intrinsics)
    print(transformation_matrix)

    def visualize_transformed_ring(ring_model_path, transformation_matrix):
        ring_model = o3d.io.read_triangle_mesh(ring_model_path)
        ring_model.transform(transformation_matrix)
        o3d.visualization.draw_geometries([ring_model])

    visualize_transformed_ring("/Users/denys.koval/University/VirtualRingTryOn/data/models/ring/ring.obj", transformation_matrix)