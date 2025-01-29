import open3d as o3d
import numpy as np
import cv2
from pose_estimation import get_transformation_matrix

def project_3d_to_2d(point_3d, camera_intrinsics):
    """Projects a 3D point onto a 2D image using intrinsic parameters."""
    fx, fy, cx, cy = camera_intrinsics[0, 0], camera_intrinsics[1, 1], camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    x, y, z = point_3d
    x_2d = int((x * fx / z) + cx)
    y_2d = int((y * fy / z) + cy)
    return x_2d, y_2d

def render_ring_on_image(ring_model_path, transformation_matrix, rgb_path, camera_intrinsics):
    rgb_image = cv2.imread(rgb_path)
    if rgb_image is None:
        raise FileNotFoundError(f"Error: Could not load RGB image at {rgb_path}")

    ring_model = o3d.io.read_triangle_mesh(ring_model_path)

    ring_model.scale(1, center=ring_model.get_center())

    R_correction = o3d.geometry.get_rotation_matrix_from_axis_angle([np.pi / 2, 0, 0])
    ring_model.rotate(R_correction, center=ring_model.get_center())

    ring_model.transform(transformation_matrix)

    vertices = np.asarray(ring_model.vertices)
    projected_points = [project_3d_to_2d(v, camera_intrinsics) for v in vertices]

    ring_mask = np.zeros_like(rgb_image, dtype=np.uint8)
    for point in projected_points:
        if 0 <= point[0] < rgb_image.shape[1] and 0 <= point[1] < rgb_image.shape[0]:
            cv2.circle(ring_mask, point, radius=2, color=(0, 255, 0), thickness=-1)

    blended_image = cv2.addWeighted(rgb_image, 1.0, ring_mask, 0.8, 0)

    cv2.imshow("Rendered Ring", blended_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    rgb_path = "/Users/denys.koval/University/VirtualRingTryOn/data/images/original_2.png"
    depth_log_path = "/Users/denys.koval/University/VirtualRingTryOn/data/images/depth_logs_2.txt"
    landmarks_path = "/Users/denys.koval/University/VirtualRingTryOn/data/results/original_2_landmarks.json"
    ring_model_path = "/Users/denys.koval/University/VirtualRingTryOn/data/models/ring/ring.glb"

    camera_intrinsics = np.array([[1464, 0, 960],
                                  [0, 1464, 720],
                                  [0, 0, 1]], dtype=np.float32)

    transformation_matrix = get_transformation_matrix(rgb_path, depth_log_path, landmarks_path, camera_intrinsics)

    if transformation_matrix is not None:
        render_ring_on_image(ring_model_path, transformation_matrix, rgb_path, camera_intrinsics)