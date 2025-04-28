import cv2
import os
import numpy as np
from pose_estimation import load_depth_from_log, get_transformation_matrix
from rendering import render_ring_on_image
from hand_localization import process_image

def save_frames(video_path, n, output_folder_path):
    """ Extract and save 'n' frames from the video at regular intervals. """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = total_frames // n

    os.makedirs(output_folder_path, exist_ok=True)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % step == 0 and saved_count < n:
            frame_filename = os.path.join(output_folder_path, f"original_frame_{saved_count + 1}.png")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved frame: {frame_filename}")
            saved_count += 1

        frame_count += 1
        if saved_count >= n:
            break

    cap.release()
    print(f"All frames saved in {output_folder_path}.")

def landmarks_on_original_frames(input_dir, output_dir):
    """ Process all images in the specified folder matching the pattern frame_#.png. """
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(1, input_dir + 1):
        if filename.startswith("original_frame_") and filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            print(f"Processing file: {image_path}")
            process_image(image_path, output_dir)
            print(f"Successfully processed: {image_path}")


def save_results(input_folder_path, output_path, n):
    ring_model_path = "D:/Study/Master/5_1_course/ML_Week/Friday/notebooks/data/models/ring/ring.glb"
    for i in range(1, n + 1):
        print(i)
        try:
            if os.path.exists(input_folder_path + f"original_frame_{i}_landmarks.json"):
                rgb_path = input_folder_path + f"original_frame_{i}.png"
                depth_log_path = input_folder_path + f"depth_logs_{i}.txt"
                landmarks_path = input_folder_path + f"original_frame_{i}_landmarks.json"

                intrinsics = np.array([[1464, 0, 960],
                                       [0, 1464, 720],
                                       [0, 0, 1]], dtype=np.float32)

                image = cv2.imread(rgb_path)
                depth_map = load_depth_from_log(depth_log_path, (image.shape[0], image.shape[1]))
                matrix = get_transformation_matrix(rgb_path, depth_log_path, landmarks_path, intrinsics)
                if matrix is not None:
                    render_ring_on_image(ring_model_path, matrix, rgb_path, intrinsics, depth_map, output_path + f"result_frame_{i}.png")
                    print(f"Render {i}")
        except Exception as e:
            print(f"Error processing frame {i}: {e}")
            continue


def create_video_from_images(folder_path, output_path, fps=30.0):
    images = [img for img in os.listdir(folder_path) if img.endswith(".png")]
    images.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    first_image = cv2.imread(os.path.join(folder_path, images[0]))
    height, width, layers = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for image in images:
        img_path = os.path.join(folder_path, image)
        img = cv2.imread(img_path)
        video.write(img)

    video.release()
    cv2.destroyAllWindows()

    print(f"Video successfully created and saved to: {output_path}")
