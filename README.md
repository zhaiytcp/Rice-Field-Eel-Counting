# EelTrack-Edge: High-Throughput Rice Field Eel Counting

This repository provides the trained model, training code, image dataset, video dataset, and test samples for **EelTrack-Edge**, an edge-deployable detection–tracking–counting framework for high-throughput rice field eel counting.

The framework is designed for rice field eel chute-transfer scenarios, where dense downstream motion, body entanglement, partial occlusion, and high-speed movement make manual counting and conventional vision-based counting difficult. To improve robustness under occlusion, this project uses the **rice field eel head** as the unified target for detection, tracking, and counting.

## Highlights

- Lightweight eel-head detector based on **YOLO11s-Eel**
- Improved ByteTrack-based multi-object tracking
- Flow-aligned motion-corridor constraint for dense downstream motion
- Dual-threshold hysteresis counting line to reduce duplicate counts
- High-frame-rate video processing for 120 FPS chute scenarios
- Edge deployment support with TensorRT FP16 acceleration

## Repository Structure

```text
EelTrack-Edge/
├── README.md
├── requirements.txt
├── models/
│   ├── yolo11s-eel.pt
│   ├── yolo11s-eel.onnx
│   └── yolo11s-eel.engine
├── train/
│   ├── train.py
│   └── val.py
├── tracker/
│   ├── byte_tracker_eel.py
│   └── kalman_filter_wh.py
├── counting/
│   └── line_counter.py
├── datasets/
│   └── eel_head/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── labels/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── videos/
├── samples/


   
