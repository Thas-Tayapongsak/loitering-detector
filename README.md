# Loitering Detector

Real-time computer vision system for loitering detection and tracking.

`loitering-detector` uses state-of-the-art object detection (YOLO) and multi-object tracking (BYTETrack or BoT-SORT) to detect when persons remain inside specific Regions of Interest (ROIs) for too long, sending automated alerts to help operators monitor safety and security.

---

## 🌟 Features

* **Real-time Detection & Tracking:** Detects persons and tracks their movements across frames with unique tracking IDs.
* **Custom Regions of Interest (ROIs):** Define customized multi-point polygonal boundary zones (ROIs) for each video stream.
* **Decoupled Monitoring Engine:** Uses a Redis-backed persistence layer to store tracking state, keeping system performance high and resilient to application crashes.
* **Multi-Stream Support:** Processes multiple cameras or stream feeds concurrently using optimized batch inference.
* **Flexible Deployments:** Supports both lightweight headless production workers (Docker/Server) and local debugging GUIs with video overlays.

---

## 🚀 Quick Start & Installation

### Prerequisites
* Python 3.12 or 3.13
* Redis Server (required for loitering state persistence)

### Installation
Clone the repository and install dependencies using `uv` (recommended) or `pip`:

```bash
# Clone the repository
git clone https://github.com/Thas-Tayapongsak/loitering-detector.git
cd loitering-detector

# Install for headless production (no GUI windows)
uv sync --extra headless

# OR install with GUI support (for local calibration and display)
uv sync --extra gui
```

Alternatively, using standard pip:
```bash
pip install -e .[headless]  # for production
# OR
pip install -e .[gui]       # for local debugging with UI
```

---

## ⚙️ Usage

The system is controlled via the `loitering-detector` command-line tool.

### 1. Headless Production Mode (Server)
To run the detection system in production without displaying any GUI window:
```bash
loitering-detector run --config system_config.yml
```

### 2. Local Calibration & Debug UI Mode
To run the system with graphical window overlays that show detection boxes and the boundary polygon (requires GUI package extra):
```bash
loitering-detector debug --config system_config.yml
```
* Press `q` or `ESC` to close the visualizer window and stop the stream.

---

## 🔧 Configuration Guide

All settings are configured through a YAML file (e.g., `system_config.yml`). You can copy the template from `system_config.example.yml` and modify it.

### Example Configuration (`system_config.yml`)
```yaml
# Computer Vision model settings
detection:
  tracker: 'botsort'      # Tracker algorithm: 'botsort' or 'bytetrack'
  path: 'yolo26n.pt'      # Path to YOLO weights (.pt, .onnx, or .engine)
  imgsz: 640              # Model input image size
  conf: 0.3               # Model confidence threshold

# Frame sampling frequency for the detection loop
sample_fps: 10.0

# Loitering threshold rules
loitering:
  threshold: 10.0            # Seconds an object must stay in ROI to alert
  cooldown_percentage: 0.5   # Cooldown time padding when leaving ROI (e.g. 0.5 * 10s = 5s)
  redis:
    host: 'localhost'        # Redis hostname
    port: 6379               # Redis port

# Alert logging rules
alerts:
  interval: 10.0             # How often to log repeat alerts for same object (seconds)

# List of input video streams
streams:
  - id: 0
    source: 0                             # 0 for default webcam, or RTSP/HTTP stream URL
    name: "main_entrance"                 # Unique friendly name for logs and window titles
    roi_polygon: [[0, 0], [0.5, 0], [0.5, 1], [0, 1]] # Normalized boundary points [x, y] from [0.0, 0.0] to [1.0, 1.0]
    timeout: 5.0                          # Video connection timeout in seconds
```

---

## 🐋 Running with Docker

You can run the entire system (Redis + Ingestion Worker) inside Docker.

1. Configure your streams and models in `system_config.docker_example.yml`.
2. Start the services:
   ```bash
   docker-compose up --build
   ```

To run the test suite inside an isolated Docker runner container:
```bash
docker-compose -f docker-compose.test.yml up --build --abort-on-container-exit
```

---

## 🛠️ Development & Quality Assurance

We enforce code quality standards locally before code is committed or pushed.

### 1. Task Runner Tasks
We use `poethepoet` to execute project checks:
* **Format code in-place:** `uv run poe format`
* **Check formatting without modifying files:** `uv run poe format-check`
* **Lint codebase syntax:** `uv run poe lint`
* **Static type checking:** `uv run poe typecheck`
* **Run unit tests:** `uv run poe unit-test`
* **Run the full validation pipeline:** `uv run poe ci`

### 2. Git Pre-Commit Hooks
We use `pre-commit` to prevent committing invalid or improperly formatted code.
To install pre-commit git hooks locally:
```bash
uv run pre-commit install
```
The hooks run automatically on every `git commit`. You can also run them manually on all files:
```bash
uv run pre-commit run --all-files
```

### 3. Offline Testing
Unit tests run entirely offline. They do not require a live Redis instance or download YOLO weights, thanks to:
* Centralized mocks for the Redis persistence layer.
* Automatic global Redis network stubbing safety nets in `conftest.py`.
* Local dummy YOLO weights file creators and synthetic video stream frame generators.

---

## 📄 License

This project is licensed under the terms specified in the `LICENSE` file.
