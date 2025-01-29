import numpy as np
import cv2
import json
import os
import trimesh
import pyrender

def load_ring_model(model_path):
    """ Load the 3D ring model using Open3D or Trimesh """
    if model_path.endswith('.obj') or model_path.endswith('.glb'):
        mesh = trimesh.load(model_path)
        return mesh
    else:
        raise ValueError("Unsupported model format. Use .obj or .glb")

def load_hand_pose(json_path):
    """ Load hand pose landmarks from a JSON file """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return np.array(data[0], dtype=np.float32)

def transform_ring_to_hand(ring_model, rvec, tvec):
    """ Apply transformation to align ring model with estimated hand pose """
    R, _ = cv2.Rodrigues(rvec)

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = tvec.flatten()

    ring_model.apply_transform(transformation_matrix)
    return ring_model

def render_scene(image_path, ring_model, camera_intrinsics):
    """ Render the scene with the transformed ring model over the hand """
    img = cv2.imread(image_path)
    h, w, _ = img.shape

    scene = pyrender.Scene()
    mesh = pyrender.Mesh.from_trimesh(ring_model)
    scene.add(mesh)

    fx, fy, cx, cy = camera_intrinsics
    camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
    scene.add(camera)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light)

    renderer = pyrender.OffscreenRenderer(w, h)
    rendered_color, _ = renderer.render(scene)

    alpha = 0.5
    blended = cv2.addWeighted(img, 1 - alpha, rendered_color, alpha, 0)

    output_path = os.path.join("data/results", "final_rendered.png")
    cv2.imwrite(output_path, blended)
    cv2.imshow("Final Render", blended)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Final rendered image saved to: {output_path}")

def process_ring_render(image_path, json_path, model_path, camera_intrinsics):
    """ Full pipeline to load ring model, transform it, and render it on the hand """
    landmarks = load_hand_pose(json_path)

    ring_model = load_ring_model(model_path)

    rvec = np.array([[0.1], [-0.2], [0.3]])
    tvec = np.array([[0], [0], [50]])

    transformed_ring = transform_ring_to_hand(ring_model, rvec, tvec)

    render_scene(image_path, transformed_ring, camera_intrinsics)


if __name__ == "__main__":
    image_path = "data/images/original_0.png"
    json_path = "data/results/original_0_landmarks.json"
    model_path = "ring/ring.obj"
    camera_intrinsics = [1464, 1464, 960, 720]  # fx, fy, cx, cy

    process_ring_render(image_path, json_path, model_path, camera_intrinsics)