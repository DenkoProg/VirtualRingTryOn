import open3d as o3d
import numpy as np
import cv2
from pose_estimation import get_transformation_matrix, load_depth_from_log


def project_3d_to_2d(point_3d, camera_intrinsics):
    """Projects a 3D point onto a 2D image using intrinsic parameters."""
    fx, fy, cx, cy = (
        camera_intrinsics[0, 0],
        camera_intrinsics[1, 1],
        camera_intrinsics[0, 2],
        camera_intrinsics[1, 2],
    )
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


def render_ring_on_image(
    ring_model_path,
    transformation_matrix,
    rgb_path,
    camera_intrinsics,
    depth_map,
    output_path=None
):
    """
    Renders a ring onto an image and either displays it or saves to a file.

    Args:
        ring_model_path: Path to the ring model file
        transformation_matrix: 4x4 transformation matrix
        rgb_path: Path to the input RGB image
        camera_intrinsics: Camera intrinsic parameters
        depth_map: Depth map of the scene
        output_path: Optional path to save the result. If None, displays the result.
    """
    img = cv2.imread(rgb_path)
    if img is None:
        raise FileNotFoundError(f"Error: Could not load image at {rgb_path}")

    ring = o3d.io.read_triangle_mesh(ring_model_path)
    ring.scale(1, center=ring.get_center())
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

            if 0 <= x_2d < img.shape[1] and 0 <= y_2d < img.shape[0]:
                if point_3d[2] > depth_map[y_2d, x_2d] + depth_offset:
                    pts_2d = []
                    break
                pts_2d.append((x_2d, y_2d))

        if len(pts_2d) == 3:
            cv2.fillConvexPoly(overlay, np.array(pts_2d, dtype=np.int32), (0, 255, 255))

    blended = cv2.addWeighted(img, 1.0, overlay, 1, 0)

    if output_path:
        cv2.imwrite(output_path, blended)
    else:
        cv2.imshow("Rendered Ring", blended)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return blended
