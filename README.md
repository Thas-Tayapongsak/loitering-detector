# Loitering Detector

Real-time computer vision system for loitering detection and tracking.

`loitering-detector` uses state-of-the-art object detection (YOLO) and multi-object tracking (BYTETrack or BoT-SORT) to detect when persons remain inside specific Regions of Interest (ROIs) for too long, sending automated alerts to help operators monitor safety and security.

---

## 🌟 Features

* **Real-time Detection & Tracking:** Detects persons and tracks their movements across frames with unique tracking IDs.
* **Custom Regions of Interest (ROIs):** Define customized multi-point polygonal boundary zones (ROIs) for each video stream.
* **Decoupled Monitoring Engine:** Implements a clean layer architecture with abstract Interfaces allowing storage repositories (e.g., RedisStateRepository or InMemoryStateRepository) and geometry engines (e.g. OpenCVGeometryEngine) to be cleanly swapped. Persists temporal state atomically using Lua scripts in Redis to keep the worker stateless and resilient.
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

# Install for CPU-only headless production
uv sync --extra headless --extra cpu

# OR install for GPU-enabled headless production (CUDA 12.1)
uv sync --extra headless --extra gpu

# OR install for CPU-only local debugging (with GUI)
uv sync --extra gui --extra cpu

# OR install for GPU-enabled local debugging (with GUI and CUDA 12.1)
uv sync --extra gui --extra gpu
```

Alternatively, using standard pip (note: custom routing index rules are configured via `uv`, so standard `pip` will fall back to downloading standard CUDA packages):
```bash
pip install -e .[headless]            # for production CPU
pip install -e .[headless,gpu]        # for production GPU
# OR
pip install -e .[gui]                 # for local debugging CPU
pip install -e .[gui,gpu]             # for local debugging GPU
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

You can run the entire system (Redis + Ingestion Worker) inside Docker. The container builds are optimized using a multi-stage architecture that:
* **Separates Build Tools**: Compilers (`gcc`, `g++`, `python3-dev`) and packaging toolchains (`uv`) are isolated in a temporary builder stage, keeping the final runtime image secure, lightweight, and compiler-free.
* **Caches Dependencies**: Utilizes `uv sync --no-install-project` and BuildKit cache mounts to prevent reinstalling pip packages when project source files change.
* **Minimizes Footprint**: Resolves pre-compiled binary wheels (`lapx`), routes to CPU-only PyTorch by default (saving 17.5GB+), strips shared library debug symbols (`*.so`), and prunes `.venv` cache/tests, shaving off an additional ~500MB.
* **Direct Entrypoints**: Configures the virtual environment directly in the container `PATH` and runs `python` and `pytest` directly, removing the need for `uv` in the runtime containers.
* **Optimized Layer Ownership**: Copies files directly using non-root `worker` ownership (`COPY --chown=worker:worker`), avoiding costly runtime `chown -R` commands that bloat image layer sizes.

To run the application inside Docker, first configure your streams and models in `system_config.docker_example.yml`, then execute the build and run commands described below.

### ⚙️ Execution and Build Commands

* **Local Test Build**:
  Build the test target image manually:
  ```bash
  docker build -t loitering-detector:test --target test -f docker/worker.Dockerfile .
  ```

* **Local Production Build**:
  Build the production runner target image manually:
  ```bash
  docker build -t loitering-detector:prod --target production -f docker/worker.Dockerfile .
  ```

* **Run Unit Tests inside Container**:
  Build and run unit tests inside the container environment:
  ```bash
  docker-compose -f docker-compose.test.yml up --build --abort-on-container-exit
  ```

* **Run System Integration**:
  Verify the production container successfully initiates, connects to Redis, and runs the stream ingestion:
  ```bash
  docker-compose up --build
  ```

### ⚙️ Customizing Compute (CPU vs. GPU) in Docker

* **Using Docker CLI**:
  Pass the `DEVICE` build argument (options: `cpu` or `gpu`):
  ```bash
  # Build for GPU (CUDA 12.1)
  docker build --build-arg DEVICE=gpu -t loitering-detector:gpu -f docker/worker.Dockerfile .
  ```

* **Using Docker Compose**:
  Modify the `args` parameter under the service build configuration:
  ```yaml
    worker:
      build:
        context: .
        dockerfile: docker/worker.Dockerfile
        target: production
        args:
          - DEVICE=gpu # Options: 'cpu' or 'gpu'
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
* **Mock-Free Domain Unit Tests:** Business rules inside `LoiteringEngine` are tested without database mocks or OpenCV dependencies by injecting the `InMemoryStateRepository` and a simple local geometry stub.
* Centralized script-execution testing for the Redis adapter (`RedisStateRepository`) using mocks.
* Automatic global Redis network stubbing safety nets in `conftest.py`.
* Local dummy YOLO weights file creators and synthetic video stream frame generators.

---

## 📄 License

This project is licensed under the terms specified in the `LICENSE` file.
