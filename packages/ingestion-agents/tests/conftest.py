"""Fixtures compartidas: un caso sintético generado una sola vez por sesión.

Generar el volumen CBCT cuesta ~1 s; hacerlo por test multiplicaría la suite sin
aportar aislamiento (los ficheros son de solo lectura para los agentes).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ingestion_agents import ArtifactStore, synthetic


@pytest.fixture(scope="session")
def case_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("caso-sintetico")
    synthetic.write_case(root, spacing=1.2)  # espaciado grueso: suite rápida
    return root


@pytest.fixture(scope="session")
def mesh_path(case_dir: Path) -> Path:
    return case_dir / "scan_upper.obj"


@pytest.fixture(scope="session")
def cbct_dir(case_dir: Path) -> Path:
    return case_dir / "cbct"


@pytest.fixture(scope="session")
def report_path(case_dir: Path) -> Path:
    return case_dir / "informe.txt"


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "artifacts")
