# pylint: disable=all

import cv2
import os


def extract_frames(video_path, output_dir="frames", frame_interval=100):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f"Erro: Não foi possível abrir o vídeo {video_path}")
        return []

    frame_paths = []
    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = video.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(
                video_output_dir, f"frame_{saved_frame_count:05d}.jpg"
            )

            cv2.imwrite(frame_filename, frame)
            frame_paths.append(frame_filename)

            saved_frame_count += 1

        frame_count += 1

    video.release()

    print(f"Extraído {saved_frame_count} frames do vídeo: {video_path}")
    return frame_paths


def process_video_directory(input_dir, output_dir="frames", frame_interval=100):
    os.makedirs(output_dir, exist_ok=True)

    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".wmv"]
    all_files = os.listdir(input_dir)

    video_files = [
        f for f in all_files if os.path.splitext(f)[1].lower() in video_extensions
    ]

    for video_file in video_files:
        video_path = os.path.join(input_dir, video_file)
        extract_frames(video_path, output_dir, frame_interval)


if __name__ == "__main__":
    input_directory = "..."
    process_video_directory(input_directory)
