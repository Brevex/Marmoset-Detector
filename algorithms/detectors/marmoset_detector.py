# pylint: disable=all

from collections import defaultdict
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import torch
import threading
import queue
import time


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("No GPU detected. Using CPU.")
    return device


class VideoCaptureThread:
    def __init__(self, src=0, width=None, height=None, buffer_size=1):
        self.cap = cv2.VideoCapture(src)

        if not self.cap.isOpened():
            raise ValueError(f"Error opening video: {src}")

        if width and height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.buffer_size = buffer_size
        self.queue = queue.Queue(maxsize=buffer_size)
        self.stopped = False
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            if not self.queue.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stop()
                    break
                self.queue.put(frame)
            else:
                time.sleep(0.01)

    def read(self):
        if not self.queue.empty():
            return True, self.queue.get()
        return False, None

    def stop(self):
        self.stopped = True
        self.cap.release()


def main():
    device = get_device()

    model = YOLO(
        "/home/daniel/Documentos/projetos/Marmoset-Detector/trained_data/models/best_marmoset_detector.pt"
    )
    model.to(device)

    video_path = (
        "/home/daniel/Documentos/projetos/Marmoset-Detector/2022-09-14 08-30-58.mkv"
    )
    cap = VideoCaptureThread(src=video_path, buffer_size=2)

    cap_obj = cv2.VideoCapture(video_path)
    w = int(cap_obj.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_obj.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_obj.get(cv2.CAP_PROP_FPS)
    cap_obj.release()

    resize_width = 640
    resize_height = int((resize_width / w) * h)

    out = cv2.VideoWriter(
        "marmoset_detected_output.avi",
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (resize_width, resize_height),
    )

    track_history = defaultdict(lambda: [])

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video processing completed or empty frame.")
            break

        frame_resized = cv2.resize(frame, (resize_width, resize_height))

        annotator = Annotator(frame_resized, line_width=2)

        if device == "cuda":
            results = model.track(frame_resized, persist=True, device=device, half=True)
        else:
            results = model.track(frame_resized, persist=True, device=device)

        if results and len(results):
            result = results[0]
            if result.boxes is not None and result.masks is not None:
                masks = result.masks.xy
                if result.boxes.id is not None:
                    track_ids = result.boxes.id.int().cpu().tolist()
                else:
                    print("Warning: Tracking IDs not found. Assigning default IDs.")
                    track_ids = [0] * len(result.boxes.xyxy)

                for mask, track_id in zip(masks, track_ids):
                    color = colors(int(track_id), True)
                    txt_color = annotator.get_txt_color(color)

                    annotator.seg_bbox(
                        mask=mask,
                        mask_color=color,
                        label=f"Marmoset {track_id}",
                        txt_color=txt_color,
                    )
            else:
                print("Warning: Boxes or masks are None.")
        else:
            print("Warning: No results returned from the model.")

        out.write(frame_resized)

        cv2.imshow("Marmoset Detector", frame_resized)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Process interrupted by user.")
            break

    out.release()
    cap.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
