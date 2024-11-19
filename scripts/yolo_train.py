from pathlib import Path
from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")

yaml_path = Path("./yolov11_dataset/data.yaml")

if not yaml_path.exists():
    raise FileNotFoundError(f"data.yaml not found: {yaml_path.absolute()}")

results = model.train(data=str(yaml_path), epochs=100, imgsz=640)

metrics = model.val()
print("Mean Average Precision for boxes:", metrics.box.map)
print("Mean Average Precision for masks:", metrics.seg.map)

model.export(format="onnx")
