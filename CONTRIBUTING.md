# Contributing to loitering-detector

Welcome! This document outlines the standards and workflows for contributing to the `loitering-detector` project. Following these guidelines ensures that the codebase remains clean, navigable, and ready for automated tooling.

## 1. Branch Strategy

We use a simplified Gitflow strategy to manage changes.

| Branch | Description |
|--------|-------------|
| `main` | **Production**. Contains only stable, released code. Protection rules require at least 1 approval. |
| `dev`  | **Integration**. The target for all feature work. Protection rules require a PR. |

### Branch Naming Convention
*   **Features**: `feature/<id>-<slug>` (e.g., `feature/US0024-alert-refactor`)
*   **Fixes**: `fix/<id>-<slug>` (e.g., `fix/BG0001-redis-timeout`)
*   **Hotfixes**: `hotfix/<slug>` (Created from `main`)

---

## 2. Commit Message Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification. This format allows us to automatically generate changelogs and version tags.

### Format
```text
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types
| Type | Use Case |
|------|----------|
| `feat` | A new feature or significant enhancement. |
| `fix` | A bug fix. |
| `chore` | Routine maintenance, dependency updates, or documentation of internal tools. |
| `docs` | Documentation changes only. |
| `refactor` | Code changes that neither fix a bug nor add a feature. |
| `test` | Adding missing tests or correcting existing tests. |
| `ci` | Changes to CI/CD configuration files and scripts. |

### Scopes
Scopes map to specific modules or functional areas:

| Scope | Affected Package/Path |
|-------|----------------------|
| `core` | `src/loitering_detector/core/` |
| `detection` | `src/loitering_detector/detection/` |
| `stream` | `src/loitering_detector/stream/` |
| `visualization` | `src/loitering_detector/visualization/` |
| `scripts` | `src/loitering_detector/scripts/` |
| `config` | `src/loitering_detector/config.py` |
| `redis` | Redis infrastructure and adapters |
| `tests` | `tests/` |
| `repo` | Repository-wide configs (`.gitignore`, `pyproject.toml`, `.gitleaks.toml`) |

### Examples
*   `feat(detection): integrate YOLOv26 inference strategy`
*   `fix(stream): resolve memory leak in video consumer`
*   `chore(repo): finalize initial repository initialization`
*   `docs(core): clarify alert escalation logic in alerts.py`

---

## 3. Development Workflows

We use two primary workflows to move code through the repository.

### Workflow A: Adding a Feature or Fix to `dev`
**Purpose**: Daily development work.

1.  **Sync**: Update your local `dev` branch.
    ```bash
    git checkout dev
    git pull origin dev
    ```
2.  **Branch**: Create a "short-lived" feature branch.
    ```bash
    git checkout -b feature/<id>-<slug>
    ```
3.  **Develop**: Commit changes using the [Commit Convention](#2-commit-message-convention).
4.  **Push**: `git push -u origin feature/<id>-<slug>`
5.  **PR**: Open a Pull Request from your branch **into `dev`**.
6.  **Merge**: Once tests pass and you are satisfied, merge into `dev`.
7.  **Cleanup**: Delete the feature branch locally and on remote.

### Workflow B: Releasing `dev` to `main`
**Purpose**: Deploying stable code to production.

1.  **PR**: Open a Pull Request from **`dev` into `main`**.
2.  **Review**: At least **1 peer approval** is required for `main`.
3.  **Merge**: Merge the PR on GitHub. **Note**: The `dev` branch is NEVER deleted.
4.  **Tag**: Create a permanent version snapshot (Tag) on `main`.
    ```bash
    git checkout main
    git pull origin main
    git tag -a vX.X.X -m "Release version X.X.X"
    git push origin vX.X.X
    ```

---

## 4. FAQ & Pro-Tips

*   **Branch Lifecycle**: `main` and `dev` are permanent (long-lived). Only `feature/` and `fix/` branches are deleted.
*   **Why Tags?**: Branches move; tags don't. A tag like `v0.3.0` is a permanent bookmark of exactly what was in production at that version.
*   **Protection Rules**:
    *   `main`: Requires a PR + 1 Approval.
    *   `dev`: Requires a PR but set to **0 required approvals**. This ensures you use the PR interface (for CI checks) but can move fast.

