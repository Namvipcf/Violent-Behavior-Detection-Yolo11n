"""
Inference YOLOv11 trên video — vẽ bounding box và đếm biển báo theo frame
Chạy: python scripts/inference_video.py --source video.mp4 --weights weights/best.pt
"""

import argparse
import cv2
from collections import Counter
from ultralytics import YOLO


def run(source: str, weights: str, conf: float, output: str):
    model = YOLO(weights)
    classes = model.names

    cap = cv2.VideoCapture(source)
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, imgsz=640, verbose=False)[0]
        boxes   = results.boxes

        labels = []
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                label  = classes[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                labels.append(label)

        # Đếm số lượng từng loại biển báo
        y_offset = 28
        for lbl, cnt in Counter(labels).items():
            cv2.putText(frame, f"{lbl}: {cnt}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            y_offset += 28

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"✅ Done! Processed {frame_count} frames → {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference YOLOv11 trên video")
    parser.add_argument("--source",  default="video.mp4",      help="Đường dẫn video đầu vào")
    parser.add_argument("--weights", default="weights/best.pt", help="File weights .pt")
    parser.add_argument("--conf",    type=float, default=0.5,   help="Ngưỡng confidence")
    parser.add_argument("--output",  default="output.mp4",      help="File video đầu ra")
    args = parser.parse_args()

    run(args.source, args.weights, args.conf, args.output)
