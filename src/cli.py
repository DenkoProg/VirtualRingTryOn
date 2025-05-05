import typer
import cv2
import numpy as np
from hand_localization import process_image
from pose_estimation import get_transformation_matrix, load_depth
from rendering import render_ring_on_image
from video_processing import (
    save_frames,
    landmarks_on_original_frames,
    save_results,
    create_video_from_images,
)
import os
from pathlib import Path

app = typer.Typer()


def get_intrinsics():
    """Get camera intrinsics matrix."""
    return np.array([[800, 0, 600], [0, 1000, 720], [0, 0, 1]], dtype=np.float32)


@app.command(name="render-ring-on-image")
def render_ring_on_image_command(
    rgb_path: str = typer.Option(..., help="Path to the RGB image."),
    depth_path: str = typer.Option(..., help="Path to the depth data (either image or log file)."),
    ring_model_path: str = typer.Option(..., help="Path to the ring model file."),
    output_dir: str = typer.Option(..., help="Directory to save the output image.")
):
    """Render a ring on a single image using the given transformation matrix."""
    try:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        landmarks_path = output_dir_path / f"{Path(rgb_path).stem}_landmarks.json"
        process_image(rgb_path, str(output_dir_path))

        image = cv2.imread(rgb_path)
        depth_map = load_depth(depth_path, (image.shape[0], image.shape[1]))

        intrinsics = get_intrinsics()
        matrix = get_transformation_matrix(
            rgb_path,
            depth_path,
            str(landmarks_path),
            intrinsics
        )

        if matrix is not None:
            output_path = output_dir_path / f"{Path(rgb_path).stem}_with_ring.png"
            render_ring_on_image(ring_model_path, matrix, rgb_path, intrinsics, depth_map, str(output_path))
            typer.echo(f"Successfully rendered ring on image. Output saved to: {output_path}")
        else:
            typer.echo("Failed to get transformation matrix. Ring was not rendered.", err=True)
            raise typer.Exit(1)

    except Exception as e:
        typer.echo(f"Error rendering ring on image: {str(e)}", err=True)
        raise typer.Exit(1)


@app.command(name="render-ring-on-video")
def render_ring_on_video_command(
    video_path: str = typer.Option(..., help="Path to the input video file."),
    output_path: str = typer.Option(..., help="Path to save the output video."),
    ring_model_path: str = typer.Option(..., help="Path to the ring model file."),
    num_frames: int = typer.Option(
        30, help="Number of frames to process from the video."
    ),
    fps: float = typer.Option(30.0, help="Frames per second for the output video."),
):
    """Render a ring on a video by processing frames and creating a new video."""
    try:
        temp_dir = Path("temp_processing")
        frames_dir = temp_dir / "frames"
        landmarks_dir = temp_dir / "landmarks"
        results_dir = temp_dir / "results"

        for dir_path in [frames_dir, landmarks_dir, results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        typer.echo("Extracting frames from video...")
        save_frames(video_path, num_frames, str(frames_dir))

        typer.echo("Processing landmarks...")
        landmarks_on_original_frames(str(frames_dir), str(landmarks_dir))

        typer.echo("Rendering rings on frames...")
        save_results(str(frames_dir), str(results_dir), num_frames)

        typer.echo("Creating output video...")
        create_video_from_images(str(results_dir), output_path, fps)

        for dir_path in [frames_dir, landmarks_dir, results_dir]:
            for file in dir_path.glob("*"):
                file.unlink()
            dir_path.rmdir()
        temp_dir.rmdir()

        typer.echo(f"Successfully created video with ring rendering: {output_path}")
    except Exception as e:
        typer.echo(f"Error processing video: {str(e)}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
