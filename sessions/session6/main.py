"""
Session 6 — YOLO / Ultralytics object detection (course scratchpad).

COMPATIBILITY — do NOT run with the main Lab env (OpenCV 4.x).
This session needs the isolated vision runtime (Ultralytics + OpenCV 5.x):

  cd sessions/session6
  ..\\..\\runtimes\\vision\\.venv\\Scripts\\python.exe main.py

Or from repo root:

  uv run --project runtimes/vision python sessions/session6/main.py

Promoted Lab UI: projects/detection-studio (same vision runtime).

Course regions below stay commented for learning — uncomment one block at a time.
A small default demo runs when you execute this file as-is.
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

# Resolve assets relative to this file so it works from any cwd.
_SESSION = Path(__file__).resolve().parent
_IMAGES = _SESSION / "images"
_MODELS = _SESSION / "models"
_OUTPUT = _SESSION / "output"
_OUTPUT.mkdir(exist_ok=True)


#region Part 1 Object Detection

# When uncommenting course snippets, prefer paths like:
#   image_path = str(_IMAGES / "sampleImage.jpg")
#   model = YOLO(str(_MODELS / "yolo12n.pt"))
# (plain "images/..." only works if your shell cwd is sessions/session6)

# image_path = "images/sampleImage.jpg"

#----------------------------------------------YoLO 8
# model = YOLO("yolov8x.pt")
#4 persons, 21 cars, 1 truck, 380.7ms

#--------------------------------------------YOLO12
# model = YOLO("yolo12n.pt")
#2 persons, 15 cars, 65.9ms

'''
{0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 
10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 
20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 
40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 
50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 
60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 
70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
'''
#
# result = model.predict(image_path)
# result[0].show()
# print(result[0].names)
#endregion

#region Part 2 Object Detection, YPLO Parameters
# image_path = "images/sampleImage.jpg"
# model = YOLO('models/yolov8x.pt')

# result = model.predict(source=image_path, classes=[0])            #person
# result = model.predict(source=image_path, classes=[7])            #truck
# result = model.predict(source=image_path, classes=[0], save=True)
# result = model.predict(source=image_path, classes=[0], conf=0.1)
# result = model.predict(source=image_path, conf=0.5, save=True, save_txt=True, save_crop=True)


# result[0].save('output/sampleImage_person_detetion.jpg')
# result[0].show()
#endregion

#region Object Detection, Exercise
# image_path = "images/Exercise2.jpg"
# model = YOLO('models/yolo12n.pt')

# result = model.predict(source=image_path, classes=[0], save=True)            #person
# result = model.predict(source=image_path, classes=[7], save=True)            #truck
# result = model.predict(source=image_path, save=True)

# result[0].show()
#endregion

#region Part 3 CPU or GPU or NPU
# import torch
# print(torch.__version__) #2.14.0+cpu
# print(torch.cuda.is_available())
#endregion

#region Part 4 Video
#
# import cv2
# from ultralytics import YOLO
#
# def process_video(video_path:str):
#     cap = cv2.VideoCapture(video_path)
#     model = YOLO('models/yolo12n.pt')
#     # model = YOLO('models/yolov8l.pt')
#     # model = YOLO('models/yolo12n.pt')
#
#
#     while True:
#         ret, frame = cap.read()
#
#         if not ret:
#             break
#
#         results = model.predict(frame, stream=False, classes=2, conf = 0.2)
#         detection_classes = results[0].names
#
#         for result in results:
#             #print(result.boxes.data.tolist())
#             for data in result.boxes.data.tolist():
#                 class_code = data[-1]
#                 draw_box(data= data, image=frame, name=detection_classes[class_code])
#                 print('detected class', detection_classes[class_code])
#         cv2.imshow('image', frame)
#         cv2.waitKey(1)
#
#
#
# def draw_box(data, image, name):
#     (x1, y1, x2, y2, conf, code) = data
#
#     pt_1 = (int(x1), int(y1))
#     pt_2 = (int(x2), int(y2))
#
#     cv2.rectangle(image, pt_1, pt_2, (0, 0, 255), 2)
#     # cv2.putText(image, name , pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#     txt = name + str(round(conf, 2))
#     cv2.putText(image, txt, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#
#     return image
#
#
# process_video(video_path='images/traffic.mp4')


#endregion

#region Part 5 YOLO Video
# import cv2
# from ultralytics import YOLO
#
# model = YOLO('models/yolo12n.pt')
#
# vs = cv2.VideoCapture('images/traffic.mp4')
#
# while True:
#     ret, frame = vs.read()
#
#     if not ret:
#         break
#
#     results = model.predict(frame, stream=False)
#
#     annotated_frame = results[0].plot()
#
#     cv2.imshow('frame', annotated_frame)
#
#     cv2.waitKey(1)

#endregion


def run_default_demo() -> None:
    """Visible demo when course regions are still commented out."""
    image_path = _IMAGES / "sampleImage.jpg"
    model_path = _MODELS / "yolo12n.pt"
    out_path = _OUTPUT / "sampleImage_detect.jpg"

    print("Session 6 — default demo (course regions above are commented).")
    print(f"  image : {image_path}")
    print(f"  model : {model_path}")

    if not image_path.is_file():
        raise SystemExit(f"Missing sample image: {image_path}")
    if not model_path.is_file():
        raise SystemExit(
            f"Missing weights: {model_path}\n"
            "Place yolo12n.pt under sessions/session6/models/"
        )

    model = YOLO(str(model_path))
    results = model.predict(source=str(image_path), conf=0.25, verbose=True)
    results[0].save(filename=str(out_path))
    n = len(results[0].boxes) if results[0].boxes is not None else 0
    print(f"Detected {n} boxes.")
    print(f"Saved annotated image → {out_path}")
    print("Tip: uncomment a #region in this file to try other course exercises.")


if __name__ == "__main__":
    run_default_demo()
