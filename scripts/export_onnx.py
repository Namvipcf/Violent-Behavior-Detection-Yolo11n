"""
Export best.pt sang ONNX và TensorRT
Chạy: python scripts/export_onnx.py
"""

from ultralytics import YOLO

WEIGHTS = "weights/best.pt"

model = YOLO(WEIGHTS)

# Export ONNX — chạy được trên CPU, dễ deploy
print("Exporting to ONNX...")
model.export(format="onnx", imgsz=640, dynamic=True)
print("✅ Saved: weights/best.onnx")

# Export TensorRT FP16 — tối ưu cho GPU NVIDIA (cần CUDA)
# print("Exporting to TensorRT...")
# model.export(format="engine", imgsz=640, half=True, device=0)
# print("✅ Saved: weights/best.engine")
