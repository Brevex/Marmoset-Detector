# pylint: disable=all

from collections import defaultdict
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

track_history = defaultdict(lambda: [])
model = YOLO("...")
video_path = "..."

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error opening video: {video_path}")
    exit()

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    "marmoset_detected_output.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h)
)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video processing completed or empty frame.")
        break

    annotator = Annotator(frame, line_width=2)
    results = model.track(frame, persist=True)

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
        print("Warning: No results returned from model.")

    out.write(frame)

    cv2.imshow("Marmoset Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Process interrupted by user.")
        break

out.release()
cap.release()
cv2.destroyAllWindows()
