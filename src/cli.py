import typer
import cv2
import numpy as np
from hand_localization import process_images_in_folder
from pose_estimation import get_transformation_matrix, load_depth_from_log
from rendering import render_ring_on_image
import os

app = typer.Typer()


@app.command(name="render-ring-on-image")
def render_ring_on_image_command(images_folder: str = typer.Option(..., help="Path to the images folder."),
                         landmarks_folder: str = typer.Option(..., help="Path to the landmarks output folder."),
                         rgb_path: str = typer.Option(..., help="Path to the RGB image."),
                         depth_log_path: str = typer.Option(..., help="Path to the depth log file."),
                         ring_model_path: str = typer.Option(..., help="Path to the ring model file.")):
    """Render a ring on an image using the given transformation matrix."""

    intrinsics = np.array([[1464, 0, 960],
                           [0, 1464, 720],
                           [0, 0, 1]], dtype=np.float32)

    process_images_in_folder(images_folder, landmarks_folder)
    image = cv2.imread(rgb_path)
    depth_map = load_depth_from_log(depth_log_path, (image.shape[0], image.shape[1]))
    matrix = get_transformation_matrix(rgb_path, depth_log_path, os.path.join(landmarks_folder, os.path.splitext(os.path.basename(rgb_path))[0] + "_landmarks.json"), intrinsics)
    if matrix is not None:
        render_ring_on_image(ring_model_path, matrix, rgb_path, intrinsics, depth_map)
    typer.echo(f"Rendering ring on image: {rgb_path}")


if __name__ == "__main__":
    app()