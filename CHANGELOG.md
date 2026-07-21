# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Added `DetectionResult` and `BoundingBox` models for decoupled inference output.
- Integrated Roboflow `trackers` package (`TrackersByteTrack`) for stateful ByteTrack multi-object tracking.

### Changed
- Decoupled detection providers by moving `YOLODetection` to `loitering_detector.detection.providers.ultralytics`.
- Standardized multi-object tracking configuration on `TrackersByteTrack` and removed legacy BoT-SORT tracker options.

## [0.3.0] - 2026-07-18

[Unreleased]: https://github.com/Thas-Tayapongsak/loitering-detector/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Thas-Tayapongsak/loitering-detector/releases/tag/v0.3.0
