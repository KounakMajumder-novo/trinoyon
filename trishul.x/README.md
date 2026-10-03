# Offline Vision-X

A local laptop prototype for camera capture, people detection, multi-object tracking, optional calibrated ground-plane coordinates, and a LAN dashboard. It runs without cloud services and now includes an optional YOLO-based path with CUDA/GPU detection when available, plus graceful CPU fallback and local safety/logging layers.

## Run

Install Python 3.10+ and the dependencies, then:

```powershell
python -m pip install -r requirements.txt
python app.py
```

On Windows, `start.bat` creates a local virtual environment, installs the small required dependency set if needed, and starts the app. Open `http://127.0.0.1:8765`. On a trusted local network, open `http://<laptop-LAN-IP>:8765` from another device. Windows Firewall may require allowing Python on Private networks. Camera and network access depend on host permissions.

Use `python app.py --camera 1` for another webcam. Select `--mode industrial` for the industrial rules view. The app remains usable when OpenCV or a camera is absent; status will explicitly show that detection/video is unavailable.

## What works now

- Real webcam capture and MJPEG stream when OpenCV and a camera are available.
- OpenCV's built-in HOG people detector as a **baseline**, never presented as a custom-trained model.
- Session-local motion/appearance-aware association, bounded lost-track handling, local color-histogram ReID, velocity and direction estimates. ReID is heuristic and may refuse a match rather than confidently identify similar-looking people.
- A local JSON API: `/api/status`, `/api/tracks`, `/api/events`.
- Configurable homography from image points to room-plane coordinates (disabled until calibrated).
- Local-only HTTP dashboard and a rule-based industrial zone/dwell event framework. Industrial demo mode tracks people in image pixels unless a camera homography is calibrated; the sample zone is a 640×480 image-space region.

## Detector and GPU

The app uses a local YOLO model from `models/yolov8n.pt`, `weights/yolov8n.pt`, or `yolov8n.pt` when present and Ultralytics is installed. Otherwise it reports and uses OpenCV's HOG people baseline. The ZIP does not bundle YOLO weights or PyTorch; install a CUDA-enabled PyTorch build appropriate to your NVIDIA driver, install Ultralytics, and add a local lightweight person-detection model to enable CUDA inference. The status API reports the actual inference device, not just whether a GPU is detected.

The industrial module is a configurable zone/dwell rule engine, not a factory defect detector. Configure zones and counts in `industrial.yaml`; coordinates are image pixels by default and become ground-plane meters only after camera calibration. Pixel speed is not metric speed. A zone can be marked with `"safety_level":"warning"` or `"safety_level":"critical"` (or `"hazard":true`) to raise a persistent advisory while a tracked person remains inside; the alert clears when the track leaves or disappears. Other rules include unexpected class, wrong zone, excessive dwell, expected-count mismatch, and zone entry/exit. Alerts are for a human controller only: there is no sensor-based machinery fault diagnosis, automatic shutdown, or equipment control. No GPS, drone SDK, thermal camera, or phone-camera protocol is included. Safety labels are demonstration heuristics and require human verification; do not use this prototype as a safety system.

## Public-safety panel

The dashboard's public-safety bar summarizes webcam person-track heuristics, prolonged-stationary checks, missing tracks, and a recommended human-verification action. For a WARNING, CRITICAL, or MISSING track, **Request review** records an operator-confirmed request in `logs/assistance_requests.csv`. It includes a pseudonymous track ID, the track's approximate image-pixel center, timestamp, reason, and current status—no captured frame, face image, name, or biometric template. This is a local review queue, not a volunteer dispatch service; an operator must verify the live scene and use an approved human process. The CSV files `logs/person_tracking.csv` and `logs/person_events.csv` contain continuous image-space telemetry and state-change events. Pixel coordinates and speed are not real-world location or velocity.

The app does not infer criminal or terrorist intent, identify people, or classify someone as dangerous based on appearance or movement. This is not a medical or emergency diagnosis; WARNING/CRITICAL labels mean that the configured webcam heuristic needs human verification. The user-provided [Public-Safety reference](https://github.com/KounakMajumder-novo/Public-Safety) describes separate YOLO/CNN models and external Telegram alerts, but its model weights, training data, alert credentials, and validated detectors are not included in this webcam app. This app reuses only the general pattern of time-confirmed, visible alerts; it does not claim fire, weapon, accident, violence, or threat recognition and sends no remote notifications.

See [RUN.md](RUN.md), [ARCHITECTURE.md](ARCHITECTURE.md), [MODEL_TRAINING.md](MODEL_TRAINING.md), and [LIMITATIONS.md](LIMITATIONS.md).
