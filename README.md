# Real-Time Multi-Person Pose Tracking & Behavior Analysis

This project implements a real-time pipeline for detecting, tracking, and analyzing human behavior from video streams.

## Features
- Person detection using YOLOv8
- Multi-object tracking (DeepSORT)
- Per-person pose estimation
- Pose-based behavior analysis (Standing, Walking, Running)
- Real-time optimization (frame skipping, FP16 inference)

## Pipeline
Video Frame
→ Detection
→ Tracking (ID)
→ Pose Estimation (per track)
→ Behavior Analysis

## Performance
- GPU: RTX 3050 Ti
- FPS: 10–12 (with pose & tracking enabled)
- Resolution: 720p

## Notes
- Tracking IDs are local and temporal.
- Fall detection is implemented as a heuristic demo and not production-grade.
- System prioritizes explainability and low latency over end-to-end accuracy.

## Tech Stack
- Python
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- DeepSORT
