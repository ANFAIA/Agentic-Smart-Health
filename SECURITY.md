# Security Policy

## Scope

This policy covers the **Agentic Smart Health** monorepo, including:

- `apps/agent-orchestrator` — the multi-agent orchestration layer.
- `apps/research-agent` — the literature agent: the only component that holds an API
  key and reaches the network.
- `packages/core-schemas` — shared Pydantic schemas for clinical data contracts.
- `packages/ingestion-agents` — ingestion of CBCT/DICOM, intraoral meshes, reports
  and images.
- `packages/fusion-agents` and `packages/analysis-agents` — fusion, registration and
  anatomical segmentation over the digital twin.
- `packages/export-agents` and `packages/uos` — materialisation of the twin, viewer
  assets and `.uos` containers.
- `packages/gaussian-engine` and `packages/3dgs-engine` — Gaussian field and 3DGS
  processing code.
- `packages/tooth-aggregation` — per-tooth aggregation across acquisitions.

Given that this system processes sensitive clinical data (CBCT/DICOM, STL, clinical reports), security issues are treated with the highest priority.

---

## Supported versions

| Version | Supported |
|---|---|
| `main` | Yes |
| `develop` | No; integration branch only |
| Tagged releases | Yes (latest only) |
| Older releases | No |

---

## Reporting a vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

This is especially critical for vulnerabilities that could:

- Expose or leak patient clinical data (DICOM, STL, PDF reports, or any data handled by the ingestion agents).
- Allow unauthorized access to or manipulation of the orchestrator, generated viewer
  assets, exported files or `.uos` containers.
- Compromise the integrity of the Digital Twin (unauthorized writes, data poisoning).
- Bypass anonymisation or pseudonymisation mechanisms.
- Violate GDPR, HIPAA, or equivalent clinical data protection regulations.

### How to report

Please use one of the following private channels:

1. **GitHub private vulnerability reporting** (preferred): navigate to the repository's *Security* tab and select *Report a vulnerability*. This keeps the report confidential until a fix is available.

2. **Email**: send a detailed report to the maintainers at the address listed in the repository's GitHub profile. Encrypt your message with the maintainers' PGP key if available.

### What to include in your report

- A clear description of the vulnerability and its potential impact.
- The affected component(s) and version(s).
- Step-by-step reproduction instructions or a proof-of-concept (without exposing real patient data).
- Any suggested mitigations or patches you may already have.

---

## Response process

| Step | Target timeframe |
|---|---|
| Acknowledgement of receipt | 48 hours |
| Initial severity assessment | 5 business days |
| Fix developed and reviewed | Depends on severity (critical: ≤ 14 days) |
| Advisory published (if applicable) | Coordinated with reporter |

We follow **responsible disclosure**: we will work with you to understand and fix the issue before any public disclosure. We will credit reporters in the release notes unless they prefer to remain anonymous.

---

## General security guidelines for contributors

- Never commit secrets, API keys, or credentials. Use `.env` files (excluded from version control via `.gitignore`) and reference `.env.example` for required variables.
- Never commit real patient data to the repository, even in anonymised form. Use the synthetic datasets provided under `data/`.
- Dependencies are pinned via `uv.lock`. Review dependency updates carefully; run `make test` after any lockfile change.
- Any new external integration, agent-facing tool or generated viewer surface must be
  reviewed for injection risks and excessive permission grants before merging.
- Clinical data flowing through agents must remain within the authorised storage boundaries defined in the architecture (`docs/architecture/`). No agent may forward or persist clinical data to external services without explicit human approval.

---

## Compliance references

This project is designed with the following regulatory frameworks in mind:

- **GDPR** (EU General Data Protection Regulation)
- **HIPAA** (US Health Insurance Portability and Accountability Act)
- Applicable national health data protection regulations

Security decisions that affect regulatory compliance must be documented in `docs/architecture/` and reviewed by a maintainer before merging.
