# System Usage and Reference Guide

This document is the authoritative operations and configuration manual for the `loitering-detector` system. It defines the structured options for calibrating the computer vision pipelines, configuring multi-stream ingestion, adjusting tracking parameters, and setting up alerts.

---

## Configuration Schema Reference

The system behavior is controlled by a YAML configuration file (typically named `system_config.yml`). This file is parsed into Pydantic settings objects that enforce type validation, defaults, and boundary constraints.

### 1. Global Settings

These settings control system-wide behavior.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `sample_fps` | Float | *Required* | The frame rate at which the system samples incoming stream frames for model inference. E.g., `10.0` means 10 frames per second. |

### 2. Detection Parameters (`detection`)

Defines parameters for model inference, target categories, and multi-object tracking. These options map directly to the `DetectionConfig` schema in `src/loitering_detector/detection/config.py`.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `path` | String | *Required* | Path to the local YOLO weights file (e.g., `yolo26n.pt`, `yolov8n.onnx`, or TensorRT engine). |
| `imgsz` | Integer | `640` | Input image size width/height for the detection model. Must be greater than 0. |
| `conf` | Float | `0.5` | Confidence threshold for object detection. Values must fall within the range `[0.0, 1.0]`. |
| `tracker` | String | `bytetrack` | Multi-object tracking algorithm to use. Supported values: `bytetrack`, `botsort`. |
| `classes` | List [Int] | `[0]` | List of COCO dataset class IDs to detect and track. `0` corresponds to "person". |

### 3. Loitering Logic Parameters (`loitering`)

Defines time thresholds, grace periods, and state persistence configs. These options map to `LoiteringConfig` in `src/loitering_detector/config.py`.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `threshold` | Float | `60.0` | Consecutive duration (in seconds) that an object must occupy the Region of Interest (ROI) before a loitering event is triggered. Must be >= 0. |
| `cooldown_percentage` | Float | `0.1` | Cooldown period defined as a fraction of `threshold`. Indicates how long the system waits after an object leaves the ROI before clearing its state. Value must be in range `[0.0, 1.0]`. |
| `redis` | Object | *Required* | Configuration block for Redis state persistence. Maps to `RedisConfig`. |

#### Redis Configuration (`loitering.redis`)

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `host` | String | *Required* | The hostname or IP address of the running Redis server (e.g., `localhost` or `redis`). |
| `port` | Integer | *Required* | The port of the Redis server (typically `6379`). |

### 4. Alert Parameters (`alerts`)

Defines notification policies. Maps to `AlertConfig` in `src/loitering_detector/config.py`.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `interval` | Float | `60.0` | The minimum interval in seconds before repeating a "still loitering" alert notification for the same object. Must be >= 0. |

### 5. Stream Configurations (`streams`)

An array of video stream inputs to process in parallel. Each item in the list maps to the `StreamConfig` schema in `src/loitering_detector/stream/config.py`.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `id` | Integer | *Required* | Unique, non-negative identifier for the stream. |
| `source` | Str \| Int | *Required* | Video source. Can be an RTSP/HTTP camera stream URL, a local video file path, or a webcam index (e.g., `0`). |
| `name` | String | *Required* | Friendly name for the stream used in window title displays and logging contexts. |
| `roi_polygon` | List [List [Float]] | `[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]` | A list of coordinate pairs defining the polygonal Region of Interest (surveillance zone). Coordinates are normalized coordinates between `0.0` and `1.0`. Requires at least 3 vertices. |
| `timeout` | Float | `5.0` | Connection timeout in seconds. Must be greater than 0. |

---

## Annotated Configuration Template

Below is a fully documented template (`system_config.example.yml`) that can be customized to initialize the loitering detector.

```yaml
# ==============================================================================
# Loitering Detector System Configuration Template
# ==============================================================================

# Global frequency at which the system processes incoming stream frames
sample_fps: 10.0

# Detection & Tracking Model Parameters
detection:
  # Path to target YOLO model weights
  path: "yolo26n.pt"

  # Input image size constraint (YOLO standard sizes: 320, 640, etc.)
  imgsz: 640

  # Model confidence threshold (higher filters false positives, lower increases recall)
  conf: 0.3

  # Tracking algorithm option: 'bytetrack' or 'botsort'
  tracker: "botsort"

  # List of COCO classes to filter for (0 = person)
  classes:
    - 0

# Loitering Decision Engine Parameters
loitering:
  # Duration (seconds) an object must remain in the ROI to be considered loitering
  threshold: 10.0

  # Cooldown period fraction (e.g., 0.5 * 10s = 5 seconds) before state is cleared
  cooldown_percentage: 0.5

  # State persistence (Redis database configuration)
  redis:
    host: "localhost"
    port: 6379

# Repeat Alert Interval Parameters
alerts:
  # The frequency (seconds) to issue logs/alerts for an actively loitering entity
  interval: 10.0

# Surveillance Camera Stream Definitions (add entries to process multiple feeds)
streams:
  - id: 0
    # Webcam index (e.g., 0) or RTSP string ("rtsp://user:pass@ip:port/h264")
    source: 0
    name: "webcam"

    # Normalized coordinates defining the polygon surveillance region.
    # Format: [[x1, y1], [x2, y2], ...] where top-left is [0, 0] and bottom-right is [1, 1].
    roi_polygon:
      - [0.0, 0.0]
      - [0.5, 0.0]
      - [0.5, 1.0]
      - [0.0, 1.0]

    # Maximum elapsed duration (seconds) before assuming stream timeout
    timeout: 5.0
```

---

## Command Line Interface (CLI) Reference

The system provides a unified entrypoint script registered as `loitering-detector`. You can invoke it from the console within your virtual environment.

### Global Flags

*   `-h`, `--help`: Show the help menu listing global options and subcommands.

### Subcommands

The CLI expects one of the following subcommands:

#### 1. `run` (Production Headless Mode)

Runs the loitering detection pipeline headlessly. No visual GUI windows will be initialized. This is the recommended mode for server deployment, background processes, and Docker containers.

```bash
loitering-detector run --config system_config.yml
```

#### 2. `debug` (Local Development & Visualization Mode)

Runs the loitering detection pipeline with live OpenCV graphical user interface (GUI) windows displaying the video feed, model bounding boxes, tracking IDs, and the polygonal Region of Interest (ROI) boundaries.

```bash
loitering-detector debug --config system_config.yml
```

### Options

Both subcommands support the following options:

*   `--config` (default: `system_config.yml`): Specifying the path to the system configuration YAML file.

### Expected Exit Codes

*   `0`: Normal exit. Occurs upon receiving termination signals (`SIGINT` / `SIGTERM`) or closing all visualizer windows.
*   `1`: Failure exit. Occurs due to invalid configuration files, missing display environment when running `debug`, or infrastructure connection losses (e.g., Redis database unreachable).

---

## GUI Interactive Visualizer Controls

When running in `debug` mode, the system launches a window for each active stream. You can interact with these windows using the following controls:

### Keyboard Bindings

Click on any stream window to focus, and use the following keyboard keys:

| Key Binding | Action | Description |
| :--- | :--- | :--- |
| `q` | Exit Program | Instantly closes all windows and shuts down the pipelines. |
| `ESC` (ASCII 27) | Exit Program | Instantly closes all windows and shuts down the pipelines. |

### Window Close Interactions

*   **Window "X" Button:** Clicking the close button on any stream window triggers a window property check (`cv2.getWindowProperty`). The system will automatically stop that stream's thread (`stream.stop()`) and release its resources.
*   **Auto-Termination:** If all active stream windows are closed by the user, the program detects that no streams are left to display and shuts down cleanly with exit code `0`.

### Headless Safety Check

The `debug` subcommand lazy-imports `opencv-python` graphical libraries to remain importable on headless platforms. When executed, it checks for a display server environment (verifying the `DISPLAY` environment variable on Linux systems). If run on a headless server without a display context, it aborts execution and logs a critical error:

```
No GUI environment detected. Use 'run' for headless mode.
```

---

## Environment Variable Overrides

To facilitate containerized deployments and runtime configuration tuning without editing YAML files, the system supports environment variable overrides:

*   `REDIS_HOST`: Overrides the Redis database host path specified in `loitering.redis.host`.
*   `REDIS_PORT`: Overrides the Redis database port specified in `loitering.redis.port`.

### Usage Example

```bash
# Directing the system to a remote Redis cluster on port 6380
REDIS_HOST=192.168.1.100 REDIS_PORT=6380 loitering-detector run --config system_config.yml
```

---

## Docker Deployment Reference

The `loitering-detector` application is fully containerized and runs inside a multi-container pipeline using Docker and Docker Compose. This ensures a consistent runtime environment across staging and production.

### Multi-Container Services

The deployment is orchestrated using the local `docker-compose.yml` file, which sets up two primary services:
1.  **`redis`:** A Redis server using `redis:7.4.0-alpine` with health checking configured to ensure it is fully initialized before the worker starts.
2.  **`worker`:** The loitering detector service built via `docker/worker.Dockerfile` using a multi-stage production builder.

### Volume Configuration Strategy

To minimize Docker image sizes and support fast settings updates, the container relies on host-to-container volume mounts:
*   **Configuration Mount:** The file `system_config.docker_example.yml` is mounted to `/app/system_config.yml` in read-only (`ro`) mode. To use a custom config, update this mount in the `docker-compose.yml` file.
*   **Weights Mount:** Large weight files (e.g., `yolo26n.pt`) are mounted directly from the host system into the container path `/app/yolo26n.pt:ro` to prevent baking heavy files into Docker image layers.

### Hardware Acceleration (Targeting CPU vs. GPU)

The `docker/worker.Dockerfile` utilizes multi-stage builds and supports a global build-time argument `DEVICE` to select hardware acceleration packages.

*   **CPU Targets (`DEVICE=cpu`):** Instructs `uv` to sync the standard CPU wheel variants of PyTorch and ONNX Runtime.
*   **GPU Targets (`DEVICE=gpu`):** Instructs `uv` to resolve and pull CUDA-enabled versions of packages, including `onnxruntime-gpu`, for accelerated inference.

#### Commands

##### 1. Build and Run containers

To build the worker image targeting a CUDA GPU:
```bash
docker compose build --build-arg DEVICE=gpu
```

To start the background daemon services:
```bash
docker compose up -d
```

##### 2. Stop and Clean up containers

To stop and remove containers and network adapters (preserving named Redis database volumes):
```bash
docker compose down
```

---

## Region of Interest (ROI) Calibration Guide

To avoid processing the entire frame and focus exclusively on specific surveillance zones (e.g., doors, corridors, restricted pathways), the system filters detections using a polygonal Region of Interest (ROI).

### Coordinate Normalization Space

Instead of absolute pixel dimensions (which vary across camera streams), the system uses a normalized coordinate system:
*   The top-left corner of the stream frame is always `[0.0, 0.0]`.
*   The bottom-right corner of the stream frame is always `[1.0, 1.0]`.
*   This normalized space remains identical regardless of whether the camera stream is 1080p, 4K, or a lower resolution.

```
[0.0, 0.0] --------------------------- [1.0, 0.0]
    |                                      |
    |          SURVEILLANCE FRAME          |
    |                                      |
[0.0, 1.0] --------------------------- [1.0, 1.0]
```

### Calibration Protocol

Follow these steps to generate a custom polygon for your camera feed:

1.  **Capture a Reference Image:** Take a screenshot of the camera stream feed. Note the resolution (e.g., width = 1920, height = 1080).
2.  **Determine Vertex Pixels:** Identify the vertex pixels of the polygon zone you wish to monitor. For example, if you are monitoring a doorway, you might choose four vertices:
    *   Vertex 1 (Top-Left): `(x=960, y=540)`
    *   Vertex 2 (Top-Right): `(x=1440, y=540)`
    *   Vertex 3 (Bottom-Right): `(x=1440, y=1080)`
    *   Vertex 4 (Bottom-Left): `(x=960, y=1080)`
3.  **Compute Normalized Values:** Divide each pixel coordinate by the image dimensions (width and height):
    *   Vertex 1: `[960 / 1920, 540 / 1080]` -> `[0.5, 0.5]`
    *   Vertex 2: `[1440 / 1920, 540 / 1080]` -> `[0.75, 0.5]`
    *   Vertex 3: `[1440 / 1920, 1080 / 1080]` -> `[0.75, 1.0]`
    *   Vertex 4: `[960 / 1920, 1080 / 1080]` -> `[0.5, 1.0]`
4.  **Insert Into Config:** Add the list of coordinates to the `roi_polygon` array of the target stream:
    ```yaml
    roi_polygon:
      - [0.5, 0.5]
      - [0.75, 0.5]
      - [0.75, 1.0]
      - [0.5, 1.0]
    ```

### Geometric Validation Rules

The configuration parser strictly validates the following rules at startup:
*   **Coordinate Bounds:** Every X and Y value in the polygon must be between `0.0` and `1.0` inclusive.
*   **Polygon Closure:** The system automatically closes the polygon by connecting the last vertex back to the first vertex.
*   **Minimum Vertices:** A polygon must consist of at least 3 vertices (to form a closed area).
*   **Common Templates:**
    *   Full stream processing (Default): `[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]`
    *   Right half frame division: `[[0.5, 0.0], [0.5, 1.0], [1.0, 1.0], [1.0, 0.0]]`
