# Contributor Guidelines

This document provides a step-by-step procedural manual for setting up your development environment, validating changes locally, and understanding the codebase release conventions for the `loitering-detector` project.

---

## 🛠️ Local Environment Setup

### Prerequisites

Before setting up the repository, make sure the following dependencies are installed on your host system:

*   **Python:** Version `3.12` or newer (configured in `.python-version` and specified as `requires-python = ">=3.12"` in `pyproject.toml`).
*   **Redis:** An active Redis server instance for state persistence and asynchronous event pub/sub.
*   **uv:** Fast Python package installer and resolver (version `0.1.0` or newer recommended).

### Workspace Initialization

The project dependencies are managed via `uv` and utilize optional dependency extras to tailor the environment to your hardware and execution needs. In `pyproject.toml`, conflict rules are defined under `[tool.uv.conflicts]` to ensure `gui` vs `headless` and `cpu` vs `gpu` environments are mutually exclusive.

Choose the setup that matches your environment and run the corresponding `uv sync` command:

#### 1. Production / Headless CPU Environment
Designed for server deployments where GUI display is not available and CPU inference is used:
```bash
uv sync --extra headless --extra cpu
```

#### 2. Local Debug / GUI GPU Environment
Designed for local machines with desktop display servers and CUDA/GPU hardware acceleration:
```bash
uv sync --extra gui --extra gpu
```

#### 3. Local Debug / GUI CPU Environment
Designed for local machines with desktop display servers and CPU-only inference:
```bash
uv sync --extra gui --extra cpu
```

#### 4. Headless GPU Environment
Designed for server environments with CUDA capabilities but without display interfaces:
```bash
uv sync --extra headless --extra gpu
```

*Note: Dev dependencies (such as Ruff, Mypy, and Pytest) are declared under the `dev` dependency group in `pyproject.toml` and are installed automatically when running `uv sync`.*

---

## ⚓ Git Hooks & Quality Checks

To ensure code style and basic syntax validation are maintained prior to committing changes, the project uses `pre-commit` hooks.

### Installation

Once your `uv` environment is successfully synced, install the git hooks in your local clone:
```bash
uv run pre-commit install
```

This registers the pre-commit scripts in your `.git/hooks` directory.

### Manual Verification

If you want to run all hooks against the entire codebase without making a commit, execute:
```bash
uv run pre-commit run --all-files
```

### Registered Hooks

The hooks defined in `.pre-commit-config.yaml` validate files using the following rules:

1.  **`trailing-whitespace`:** Trims trailing whitespace from lines.
2.  **`end-of-file-fixer`:** Ensures files end with a trailing newline.
3.  **`check-yaml`:** Validates YAML file syntax (e.g. configuration files).
4.  **`check-toml`:** Validates TOML file syntax (e.g. `pyproject.toml`).
5.  **`ruff`:** Lints code and applies auto-fixes (equivalent to running `ruff check . --fix`).
6.  **`black`:** Standardizes Python code formatting matching black specifications.

---

## 🏃 Local Task Runner (Poe the Poet)

The project utilizes `poethepoet` (declared in `pyproject.toml` under `[tool.poe.tasks]`) to define, organize, and execute local validation and development workflows. You should run these tasks to format, lint, typecheck, and test your code locally.

| Command | Underlying Tooling | Purpose |
| :--- | :--- | :--- |
| `uv run poe lint` | `ruff check .` | Executes Ruff static code analysis to catch syntax errors and stylistic issues. |
| `uv run poe format` | `black .` | Formats all Python source files in the project. |
| `uv run poe format-check` | `black --check .` | Verifies formatting compliance without writing changes to the disk. |
| `uv run poe typecheck` | `mypy src tests` | Runs strict static type analyses across both the `src` and `tests` directories. |
| `uv run poe unit-test` | `pytest tests/unit` | Runs the test suite under `tests/unit` and generates coverage reports. |
| `uv run poe ci` | Sequential run of tasks | Runs `lint`, `format-check`, `typecheck`, and `unit-test` in sequence. |

*To guarantee that your branch will pass remote CI build checks, always run the full validation suite before pushing:*
```bash
uv run poe ci
```

---

## 🧪 Local Testing & Fixture Guidelines

The project enforces strict guidelines for unit and integration testing to ensure that tests remain reliable, isolated, and fast.

### Offline Testing Constraints

To prevent tests from depending on external environments, the network is strictly blocked. When writing or running tests, note the following configurations:

1.  **Global Socket Block:** Outgoing socket connections are intercepted during test runs by the `block_network_sockets` fixture located in `tests/unit/backend/conftest.py`. Attempting to communicate over raw network ports will raise a `RuntimeError`.
2.  **Redis Network Stubbing:** The `stub_redis_network` fixture automatically patches `redis.Redis` globally. This ensures that any infrastructure adapters or database client initializations do not connect to a real Redis server.
3.  **Custom Redis Assertions:** If your test relies on verifying specific Redis keys, scripting, or data flow, use the `mock_redis` fixture, which provides a standard mocked client and resets after execution.
4.  **No Model Weights Downloads:** When testing Computer Vision/YOLO components, tests must not download weight files (`.pt` or `.onnx`). Mock YOLO class loading and prediction methods via `unittest.mock.patch` (specifically patching `loitering_detector.detection.providers.ultralytics.YOLO`) to isolate model inference logic from physical model resources.

### Unit Testing Conventions (FIRST Principles)

When contributing new test suites, align with the **FIRST** testing principles:

*   **Fast:** Tests must execute in milliseconds. Always mock slow external systems, model predictions, and video decoding processes.
*   **Independent:** Tests must have no state dependencies on one another. If you modify class-level or module-level structures, always implement teardown procedures to reset mocks or clean up objects (e.g. `mock.reset_mock()`).
*   **Repeatable:** Running tests multiple times must produce identical results. Use the provided fixtures like `synthetic_frame_generator` to generate deterministic frame buffers instead of relying on external files or cameras.
*   **Self-Validating:** Tests must clearly output a binary pass/fail result using explicit `assert` statements. Avoid printing outputs to stdout for manual check.
*   **Timely:** Recommend writing unit tests concurrently with feature development to ensure high codebase testability and coverage compliance (minimum **90%** coverage threshold).

---

## 🌿 Git Branching & Collaboration Workflow

We follow a structured branching and pull request model to maintain repository stability and code quality.

### Branching Strategy

1.  **`main`:** The production-ready branch. Direct commits to `main` are strictly prohibited.
2.  **`dev`:** The primary integration branch. All features, bug fixes, and documentation updates are merged here first.
3.  **Topic Branches:** Create a short-lived branch for your changes branching directly from `dev`. Use the following naming convention:
    *   `feat/feature-name` for new features or capabilities.
    *   `fix/bug-name` for bug fixes.
    *   `docs/doc-name` for updates to documentation.
    *   `refactor/refactor-name` for code reorganization without behavior changes.
    *   `test/test-name` for test suite improvements.

*Note: All Pull Requests (PRs) must target the `dev` branch as their destination.*

### Commit Conventions

This project follows the **Conventional Commits** specification to auto-generate release changelogs and determine automated Semantic Version (`MAJOR.MINOR.PATCH`) bumps. Non-bumping prefix is preferred, unless changes made warrant version bumping:

*   `feat: ...` -> New feature or capability (**triggers MINOR version bump**, e.g. `0.3.0` -> `0.4.0`).
*   `fix: ...` -> Bug fix (**triggers PATCH version bump**, e.g. `0.3.0` -> `0.3.1`).
*   `feat!: ...` or `fix!: ...` (or `BREAKING CHANGE:` in footer) -> Breaking API or behavior change (**triggers MAJOR version bump**, e.g. `0.3.0` -> `1.0.0`).
*   `docs: ...` -> Documentation modifications (*no version bump*).
*   `style: ...` -> Non-functional style edits like formatting or whitespace (*no version bump*).
*   `refactor: ...` -> Code reorganization without external behavior changes (*no version bump*).
*   `test: ...` -> Adding missing tests or correcting existing tests (*no version bump*).
*   `chore: ...` -> Routine tasks, dependency version updates, or build process adjustments (*no version bump*).

*Example:* `feat: add custom polygonal region of interest calculations`


### Pull Request & Review Checklist

Before opening a pull request or requesting a review, verify that your changes comply with the following validation rules:

1.  **Local Execution:** Run `uv run poe ci` locally. The command runs `lint`, `format-check`, `typecheck`, and `unit-test` in sequence. All must pass.
2.  **Strict Type Checking:** Ensure all code contains full, strict typing verified by mypy.
3.  **Test Coverage Threshold:** The minimum test coverage for the project is **90%** (enforced by `tool.coverage.report` in `pyproject.toml`). If coverage drops below this threshold, the CI build will fail.
4.  **Rebase on Target:** Ensure your topic branch is rebased on the latest `dev` branch commits before requesting review.

---

## 📦 Lockfile Synchronization & Release Automation

### Lockfile Integrity

The `uv.lock` file ensures reproducible installations across development and production environments.
*   **Never modify `uv.lock` manually.**
*   To add new library dependencies, run:
    ```bash
    uv add <package-name>
    ```
*   To add packages required only for development (e.g. testing utilities, helpers):
    ```bash
    uv add --dev <package-name>
    ```
*   To synchronize your virtual environment with the lockfile after pulling updates:
    ```bash
    uv sync
    ```

### Release Automation via release-please

The repository utilizes Google's `release-please` action to automate versioning and release creation:
1.  When conventional commits are merged into the `dev`/`main` branches, the release-please bot analyzes the messages and updates a pending **Release Pull Request**.
2.  The Release PR bumps the project version in `pyproject.toml` and updates `CHANGELOG.md` with categorized commit summaries.
3.  Merging the Release PR automatically tags the release (e.g. `v0.4.0`) and publishes it to the repository releases page.
