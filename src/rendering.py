import open3d as o3d
import numpy as np
import cv2
from pose_estimation import get_transformation_matrix, load_depth_from_log


def project_3d_to_2d(point_3d, camera_intrinsics):
    """Projects a 3D point onto a 2D image using intrinsic parameters."""
    fx, fy, cx, cy = camera_intrinsics[0, 0], camera_intrinsics[1, 1], camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    x, y, z = point_3d
    x_2d = int((x * fx / z) + cx)
    y_2d = int((y * fy / z) + cy)
    return x_2d, y_2d


def compute_depth_offset(vertices, camera_intrinsics, depth_map):
    errors = []

    for point_3d in vertices:
        x_2d, y_2d = project_3d_to_2d(point_3d, camera_intrinsics)

        if 0 <= x_2d < depth_map.shape[1] and 0 <= y_2d < depth_map.shape[0]:
            Z_ring = point_3d[2]
            Z_depthmap = depth_map[y_2d, x_2d]
            error = Z_ring - Z_depthmap
            errors.append(error)

    if errors:
        return np.mean(errors)
    return 0


def render_ring_on_image(ring_model_path, transformation_matrix, rgb_path, camera_intrinsics, depth_map):
    img = cv2.imread(rgb_path)
    ring = o3d.io.read_triangle_mesh(ring_model_path)
    ring.scale(1, center=ring.get_center())
    ring.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle([np.pi / 2, 0, 0]), center=ring.get_center())
    ring.transform(transformation_matrix)

    vertices = np.asarray(ring.vertices)
    triangles = np.asarray(ring.triangles)

    overlay = np.zeros_like(img, dtype=np.uint8)
    depth_offset = compute_depth_offset(vertices, camera_intrinsics, depth_map)

    for tri in triangles:
        pts_2d = []
        for idx in tri:
            point_3d = vertices[idx]
            x_2d, y_2d = project_3d_to_2d(point_3d, camera_intrinsics)

            if point_3d[2] > depth_map[y_2d, x_2d] + depth_offset:
                pts_2d = []
                break

            pts_2d.append((x_2d, y_2d))

        if len(pts_2d) == 3:
            cv2.fillConvexPoly(overlay, np.array(pts_2d, dtype=np.int32), (0, 255, 255))

    blended = cv2.addWeighted(img, 1.0, overlay, 1, 0)
    cv2.imshow("Rendered Ring", blended)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    rgb_path = "/Users/denys.koval/University/VirtualRingTryOn/data/images/original_1.png"
    depth_log_path = "/Users/denys.koval/University/VirtualRingTryOn/data/images/depth_logs_1.txt"
    landmarks_path = "/Users/denys.koval/University/VirtualRingTryOn/data/results/original_1_landmarks.json"
    ring_model_path = "/Users/denys.koval/University/VirtualRingTryOn/data/models/ring/ring.glb"

    intrinsics = np.array([[1464, 0, 960],
                           [0, 1464, 720],
                           [0, 0, 1]], dtype=np.float32)

    image = cv2.imread(rgb_path)
    depth_map = load_depth_from_log(depth_log_path, (image.shape[0], image.shape[1]))
    matrix = get_transformation_matrix(rgb_path, depth_log_path, landmarks_path, intrinsics)
    if matrix is not None:
        render_ring_on_image(ring_model_path, matrix, rgb_path, intrinsics, depth_map)
