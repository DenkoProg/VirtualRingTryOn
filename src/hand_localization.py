import cv2
import mediapipe as mp
import os
import json

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def process_image(image_path, output_dir):
    """ Process a single image to detect hand keypoints and save annotated results. """

    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        results = hands.process(image_rgb)
        annotated_image = image.copy()

        if results.multi_hand_landmarks:
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmark_list = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                landmarks.append(landmark_list)

            image_filename = os.path.basename(image_path)
            annotated_image_path = os.path.join(output_dir, f"annotated_{image_filename}")
            cv2.imwrite(annotated_image_path, annotated_image)

            landmarks_path = os.path.join(output_dir, f"{image_filename.split('.')[0]}_landmarks.json")
            with open(landmarks_path, 'w') as f:
                json.dump(landmarks, f, indent=4)

            print(f"Processed {image_path} -> {annotated_image_path}, {landmarks_path}")


def process_images_in_folder(input_folder, output_folder):
    """ Process all images in the input folder and save results in output folder. """
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg')) and f.startswith('original')]

    if not image_files:
        print("No images found in", input_folder)
        return

    for image_file in image_files:
        process_image(os.path.join(input_folder, image_file), output_folder)


# Just to test
if __name__ == "__main__":
    input_folder = "/Users/denys.koval/University/VirtualRingTryOn/data/images"
    output_folder = "/Users/denys.koval/University/VirtualRingTryOn/data/results"
    process_images_in_folder(input_folder, output_folder)