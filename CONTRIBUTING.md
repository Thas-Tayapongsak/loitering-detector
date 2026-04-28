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

## 3. Development Workflow

1.  **Sync**: Ensure your local `dev` branch is up to date with `origin/dev`.
2.  **Branch**: Create a new feature/fix branch from `dev`.
3.  **Develop**: Make your changes and commit using the convention above.
4.  **Test**: Run the test suite (`pytest`) to ensure no regressions.
5.  **PR**: Open a Pull Request on GitHub from your branch to `dev`.
6.  **Review**: Address review comments.
7.  **Merge**: Once approved, merge into `dev`.
