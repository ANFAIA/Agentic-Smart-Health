# Branching and Release Workflow

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-09-02 |
| **Scope** | Repository workflow, protected branches and release creation |
| **Related** | CI guardians: [`AGENTS.md`](../../AGENTS.md) · UOS format: [`formato-uos.md`](formato-uos.md) |

This repository keeps development and publication separate. `develop` is where the
next version is integrated; `main` is the public line from which version tags and
GitHub Releases are created.

## Branches

| Branch | Purpose | Rules |
|---|---|---|
| `main` | Latest public state of the project | No direct work. Receives reviewed changes from PRs. Version tags and GitHub Releases are created here. |
| `develop` | Integration branch for the next version | Normal work lands here first through PRs. It may be ahead of `main` and is not necessarily release-ready. |
| `feat/...` | New functionality | Created from `develop`, merged back into `develop`. |
| `fix/...` | Non-release bug fixes | Created from `develop`, merged back into `develop`. |
| `docs/...` | Documentation-only changes | Created from `develop`, merged back into `develop`. |
| `chore/...` | Maintenance, CI and repository upkeep | Created from `develop`, merged back into `develop`. |
| `hotfix/...` | Fixes to already released code or documents | Created from `main`, merged into `main`, then mirrored back into `develop`. |

## Normal Change Flow

```text
develop
  -> feat|fix|docs|chore/<short-name>
  -> pull request into develop
  -> CI and documentation guardians pass
  -> merge into develop
```

When `develop` is ready to become public:

```text
develop
  -> pull request into main
  -> CI and documentation guardians pass
  -> merge into main
  -> create version tag on main
  -> create GitHub Release from that tag
```

## Release Rules

- Version tags are created only from `main`.
- GitHub Releases are created only from version tags already present on `main`.
- Release assets such as the UOS specification and white paper are attached to the
  GitHub Release, not committed as generated binaries unless the data guardian allows
  them.
- A pre-release is appropriate for technical snapshots whose format or API may still
  evolve, such as early UOS `0.x` publications.
- A normal release means the project is presenting that tag as the current stable
  public reference.

## Hotfix Flow

Hotfixes start from `main` because they patch something already published.

```text
main
  -> hotfix/<short-name>
  -> pull request into main
  -> CI and documentation guardians pass
  -> merge into main
  -> tag a patch release if needed
  -> merge or cherry-pick the same change back into develop
```

## GitHub Protection

The policy above should be enforced in GitHub, not only documented.

Recommended protection for `main`:

- Require a pull request before merging.
- Require status checks to pass before merging.
- Require branches to be up to date before merging.
- Require conversation resolution before merging.
- Block force pushes.
- Block branch deletion.

Recommended protection for `develop`:

- Require a pull request before merging.
- Require status checks to pass before merging.
- Block force pushes.
- Block branch deletion.

Mandatory review approval is optional while the repository is maintained by one
person. If a supervisor or teammate actively reviews PRs, require one approval.
