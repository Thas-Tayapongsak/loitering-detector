# Loitering Detector

Real-time computer vision system for loitering detection and tracking in polygonal Regions of Interest (ROIs).

---

Loitering Detector is a computer vision-driven monitoring system designed to automatically identify when entities (e.g., persons) remain inside designated zones (ROIs) longer than a defined threshold. By coupling state-of-the-art object detection (YOLO) and multi-object tracking (BYTETrack/BoT-SORT), the system operates asynchronously to emit actionable security and operational alerts, eliminating the need for continuous manual surveillance.

---

## 🌟 Features

* **Real-Time Detection & Multi-Object Tracking:** Dynamic person detection coupled with robust spatial tracking IDs.
* **Custom Polygonal Regions of Interest (ROIs):** Flexible multi-point boundary definition per video feed.
* **Decoupled Architecture:** Clean layered boundaries supporting swappable geometry engines and state storage backends (e.g., Redis).
* **Multi-Stream Batch Ingestion:** Concurrent processing of multiple RTSP feeds, camera inputs, or static resources.
* **Flexible Deployments:** Standard production worker configuration via container orchestration and lightweight local visual debugging tools.

---

## 🚀 Quick Start & Installation

### Prerequisites
* **Python:** 3.12 or 3.13
* **State Persistence:** Redis Server (required for storing tracking state and loitering timers)

### Installation
Clone the repository and install the application environment using `uv` (recommended):

```bash
# Clone the repository
git clone https://github.com/Thas-Tayapongsak/loitering-detector.git
cd loitering-detector

# Choose ONE of the following configurations based on your setup:

# 1. Headless Production Mode (CPU only)
uv sync --extra headless --extra cpu

# 2. Headless Production Mode (GPU - CUDA 12.1 enabled)
uv sync --extra headless --extra gpu

# 3. Local Debugging GUI Mode (CPU only)
uv sync --extra gui --extra cpu

# 4. Local Debugging GUI Mode (GPU - CUDA 12.1 enabled)
uv sync --extra gui --extra gpu
```

### Running the System
The system is controlled using the `loitering-detector` command-line executable.

#### 1. Headless Production Mode
To run the detection worker in production without launching graphical window overlays:
```bash
loitering-detector run --config system_config.example.yml
```

#### 2. Local Calibration & Debug UI Mode
To run the worker with graphical window overlays highlighting detected bounding boxes and ROI boundaries (requires `gui` package extra):
```bash
loitering-detector debug --config system_config.example.yml
```
* Press `q` or `ESC` inside the graphical window to stop execution.

#### 3. Docker Deployment
To build and execute the entire worker stack along with a Redis service instance:
```bash
docker-compose up --build
```

---

## 📖 Documentation Reference

For more detailed setup guides, developer tasks, and architectural insights, refer to the following sub-documents:

*   **[Architecture & Design](ARCHITECTURE.md):** Detailed guide on the decoupled monitoring layers, repository adapters, and system design patterns.
*   **[Contributor Guidelines](CONTRIBUTING.md):** Guidelines for local codebase verification, linting commands, type checkers, and pull request workflows.
*   **[Advanced Usage & Config](USAGE.md):** Complete configuration schema dictionary for streams, geometric parameters, and loitering thresholds.
*   **[Changelog](CHANGELOG.md):** Chronological history of versioned releases, enhancements, and bug fixes.
