#!/usr/bin/env python
"""metricas.py — las cuatro cifras del brief, MEDIDAS y no prometidas.

    uv run python scripts/metricas.py --casos ~/ruta/a/casos --mallas data/raw/teeth3ds

El brief compromete cuatro numeros y el repositorio los tenia en cuatro estados
distintos: la reversibilidad medida y escrita, la cobertura vigilada por el CI, y la
**latencia y la fiabilidad como objetivos**, es decir como frases. Este script las pone
a las cuatro en el mismo plano: se ejecutan, se cuentan y se fechan.

## Lo que este script NO hace, y hay que leerlo antes que los numeros

**No inventa un dataset de validacion.** No existe uno anotado para estos pacientes, asi
que la fiabilidad se mide donde hay volumen —el `mesh-agent` sobre los cientos de
escaneos de Teeth3DS+— y **por separado** de extremo a extremo sobre los casos clinicos
reales, que son poquisimos. Las dos cifras van con su N al lado. Un N pequeno declarado
es defendible; un N pequeno escondido detras de un porcentaje no lo es.

**«Fiabilidad» aqui significa que el agente se comporta como declara**, no que acierte.
Un `OK` cuenta como exito y un `FAILED` como fallo; un `MISSING` **no cuenta en ninguna
de las dos direcciones**, porque que una adquisicion no traiga fotos no dice nada del
agente. Esa es la definicion que el pipeline puede comprobar sin etiquetas de verdad, y
decir que mide otra cosa seria mentir sobre lo que se ha ejecutado.

**La latencia es la de la INGESTA**, que es lo que el brief acota a 60 s: `run()`, las
cuatro modalidades. No incluye fusion, ni 3DGS, ni exportacion — que tardan mucho mas y
nadie prometio que fueran rapidas.

**Y la cobertura no se mide aqui**: la mide `pytest --cov` con el mismo `fail_under` que
el CI. Este script la lee si le pasas el informe, y si no dice que no la tiene.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

RESUMEN_EN = "Computes the measured metrics promised in the project brief."

RAIZ = Path(__file__).resolve().parent.parent
for _src in sorted(RAIZ.glob("packages/*/src")):
    sys.path.insert(0, str(_src))
sys.path.insert(0, str(RAIZ / "apps/agent-orchestrator/src"))
sys.path.insert(0, str(RAIZ / "scripts"))

from agent_orchestrator import IngestionPipeline  # noqa: E402
from agent_orchestrator.pipeline import LATENCY_BUDGET_S  # noqa: E402
from caso_completo import descubre  # noqa: E402
from core_schemas import ModalityStatus  # noqa: E402
from export_agents.base import REVERSIBILITY_BUDGET_MM  # noqa: E402
from ingestion_agents import ArtifactStore, MeshAgent  # noqa: E402

COBERTURA_MINIMA = 80.0
FIABILIDAD_MINIMA = 95.0

# Extensiones de malla que el `mesh-agent` acepta. Se listan aqui y no se adivinan para
# que un dataset con otro formato salga como «no encontrado» y no como «cero mallas».
MALLAS = ("*.obj", "*.stl", "*.ply")


@dataclass
class Cifra:
    """Una metrica del brief: lo prometido, lo medido, y sobre que se midio."""

    nombre: str
    presupuesto: str
    medido: str | None
    sobre: str
    cumple: bool | None
    notas: list[str] = field(default_factory=list)

    def linea(self) -> str:
        marca = "  ?  " if self.cumple is None else ("  ok " if self.cumple else " FUERA")
        return f"{marca} {self.nombre:<34} {self.medido or 'sin medir':>18}   ({self.presupuesto})"


def _pseudonimo(ruta: Path) -> str:
    """Un nombre estable para el caso que NO es su directorio.

    El directorio de una carpeta de clinica lleva a menudo el nombre o el numero de
    historia del paciente. Lo que sale por pantalla y lo que se guarda en el JSON tiene
    que poder pegarse en un informe sin repasarlo.
    """
    import hashlib

    return "caso-" + hashlib.sha256(ruta.name.encode()).hexdigest()[:8]


# --------------------------------------------------------------------------- #
def mide_ingesta(casos: list[Path], salida: Path) -> tuple[Cifra, Cifra, list[dict]]:
    """Latencia y fiabilidad de extremo a extremo sobre casos reales completos."""
    detalle: list[dict] = []
    latencias: list[float] = []
    conteo: Counter[str] = Counter()

    for i, raiz in enumerate(casos):
        caso = descubre(raiz)
        almacen = ArtifactStore(salida / f"artifacts-{i}")
        pipe = IngestionPipeline(almacen, quarantine_dir=salida / "quarantine")
        resultado = pipe.run(caso)
        latencias.append(resultado.latency_s)
        fila = {"caso": _pseudonimo(raiz), "latency_s": round(resultado.latency_s, 2)}
        for o in resultado.outcomes:
            conteo[o.status.value] += 1
            fila.setdefault("modalidades", []).append(  # type: ignore[union-attr]
                {"modalidad": o.modality.value, "estado": o.status.value,
                 "latency_s": round(o.latency_s or 0.0, 3)}
            )
        detalle.append(fila)

    peor = max(latencias) if latencias else None
    latencia = Cifra(
        nombre="latencia de ingesta",
        presupuesto=f"< {LATENCY_BUDGET_S:.0f} s",
        medido=None if peor is None else f"{peor:.1f} s (peor de {len(latencias)})",
        sobre=f"{len(casos)} caso(s) clinico(s) completo(s)",
        cumple=None if peor is None else peor <= LATENCY_BUDGET_S,
        notas=["es la ingesta (`run`), no el recorrido entero: la fusion y el 3DGS "
               "tardan mucho mas y nadie prometio que fueran rapidos"],
    )

    juzgadas = conteo[ModalityStatus.OK.value] + conteo[ModalityStatus.FAILED.value]
    pct = 100.0 * conteo[ModalityStatus.OK.value] / juzgadas if juzgadas else None
    fiabilidad = Cifra(
        nombre="fiabilidad de ingesta (e2e)",
        presupuesto=f"> {FIABILIDAD_MINIMA:.0f} %",
        medido=None if pct is None else f"{pct:.1f} % ({juzgadas} intentos)",
        sobre=f"{len(casos)} caso(s) x modalidad",
        cumple=None if pct is None else pct > FIABILIDAD_MINIMA,
        notas=[
            f"{conteo[ModalityStatus.MISSING.value]} modalidad(es) MISSING no cuentan: "
            "que una adquisicion no traiga fotos no dice nada del agente",
            "N pequeno. Es el numero de casos clinicos completos que hay, no una muestra",
        ],
    )
    return latencia, fiabilidad, detalle


def mide_mallas(raiz: Path, limite: int, salida: Path) -> Cifra:
    """Fiabilidad del `mesh-agent` donde SI hay volumen: el dataset publico."""
    ficheros: list[Path] = []
    for patron in MALLAS:
        ficheros += sorted(raiz.rglob(patron))
    ficheros = ficheros[:limite]
    if not ficheros:
        return Cifra("fiabilidad del mesh-agent", f"> {FIABILIDAD_MINIMA:.0f} %", None,
                     f"{raiz} (sin mallas)", None,
                     ["no se encontro ninguna malla: la cifra no se puede dar"])

    agente = MeshAgent(ArtifactStore(salida / "artifacts-mallas"))
    conteo: Counter[str] = Counter()
    for f in ficheros:
        conteo[agente.ingest(f).status.value] += 1

    juzgadas = conteo[ModalityStatus.OK.value] + conteo[ModalityStatus.FAILED.value]
    pct = 100.0 * conteo[ModalityStatus.OK.value] / juzgadas if juzgadas else None
    return Cifra(
        nombre="fiabilidad del mesh-agent",
        presupuesto=f"> {FIABILIDAD_MINIMA:.0f} %",
        medido=None if pct is None else f"{pct:.1f} % ({juzgadas} mallas)",
        sobre=f"{raiz.name}",
        cumple=None if pct is None else pct > FIABILIDAD_MINIMA,
        notas=["mide que el agente se comporte como declara, NO que acierte: no hay "
               "anotacion contra la que comparar"],
    )


def mide_reversibilidad(raiz: Path, limite: int, salida: Path) -> Cifra:
    """El viaje fichero -> twin -> fichero, sobre mallas reales."""
    from export_agents import ExportAgent

    ficheros: list[Path] = []
    for patron in MALLAS:
        ficheros += sorted(raiz.rglob(patron))
    ficheros = ficheros[:limite]
    if not ficheros:
        return Cifra("reversibilidad (malla -> STL)", f"< {REVERSIBILITY_BUDGET_MM} mm",
                     None, f"{raiz} (sin mallas)", None, [])

    almacen = ArtifactStore(salida / "artifacts-rev")
    agente = MeshAgent(almacen)
    exportador = ExportAgent(almacen)  # type: ignore[arg-type]
    desviaciones: list[float] = []
    for i, f in enumerate(ficheros):
        salida_i = agente.ingest(f)
        if salida_i.status is not ModalityStatus.OK or salida_i.artifact_ref is None:
            continue
        snapshot = _snapshot_minimo(salida_i.artifact_ref)
        out = exportador.export(snapshot, salida / "rev" / f"{i}.stl")
        if out.max_deviation_mm is not None:
            desviaciones.append(out.max_deviation_mm)

    peor = max(desviaciones) if desviaciones else None
    return Cifra(
        nombre="reversibilidad (malla -> STL)",
        presupuesto=f"< {REVERSIBILITY_BUDGET_MM} mm",
        medido=None if peor is None else f"{peor:.2e} mm ({len(desviaciones)} mallas)",
        sobre=f"{raiz.name}",
        cumple=None if peor is None else peor < REVERSIBILITY_BUDGET_MM,
        notas=["es el error del FORMATO: el STL guarda coordenadas en float32 y la "
               "geometria se conserva entera en el almacen"],
    )


def _snapshot_minimo(ref: str):
    """Un `TwinSnapshot` con lo justo para exportar la malla que se acaba de ingerir."""
    from core_schemas import Modality, Provenance, TwinSnapshot

    return TwinSnapshot(
        acquisition_id="metricas",
        timestamp=datetime.now(UTC),
        gaussian_field_ref=ref,
        surface_ref=ref,
        provenance=Provenance(
            source_file="metricas", modality=Modality.MESH, agent="metricas.py",
            confidence=1.0,
        ),
    )


def mide_cobertura(informe: Path | None) -> Cifra:
    """Lee la cobertura de un informe de `coverage`. No la ejecuta."""
    if informe is None or not informe.exists():
        return Cifra("cobertura de tests", f"> {COBERTURA_MINIMA:.0f} %", None,
                     "no aportada", None,
                     ["se mide con `uv run pytest --cov --cov-report=json`; el CI ya "
                      f"falla por debajo del {COBERTURA_MINIMA:.0f} %"])
    datos = json.loads(informe.read_text())
    pct = float(datos["totals"]["percent_covered"])
    return Cifra("cobertura de tests", f"> {COBERTURA_MINIMA:.0f} %", f"{pct:.1f} %",
                 informe.name, pct > COBERTURA_MINIMA)


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=RAIZ,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001 - sin git tampoco se cae
        return "desconocido"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--casos", type=Path, default=None,
                    help="Directorio con carpetas de caso clinico (una por adquisicion).")
    ap.add_argument("--mallas", type=Path, default=RAIZ / "data/raw/teeth3ds",
                    help="Dataset publico de mallas, para las cifras con N grande.")
    ap.add_argument("--n-mallas", type=int, default=60,
                    help="Cuantas mallas usar. Todas es mejor y tarda mas.")
    ap.add_argument("--cobertura", type=Path, default=None,
                    help="`coverage.json` de un `pytest --cov --cov-report=json`.")
    ap.add_argument("--salida", type=Path, default=RAIZ / "data/processed/metricas")
    ap.add_argument("--json", type=Path, default=None, help="Guarda el detalle aqui.")
    args = ap.parse_args()
    args.salida.mkdir(parents=True, exist_ok=True)

    casos: list[Path] = []
    if args.casos:
        # Un caso es un directorio que trae ALGO que ingerir. Se descubre por forma y no
        # por nombre, igual que en `caso_completo`.
        for d in [args.casos, *sorted(p for p in args.casos.iterdir() if p.is_dir())]:
            c = descubre(d)
            if c.mesh or c.cbct or c.reports or c.images:
                casos.append(d)

    arranque = time.perf_counter()
    cifras: list[Cifra] = []
    detalle: list[dict] = []
    if casos:
        latencia, fiabilidad, detalle = mide_ingesta(casos, args.salida)
        cifras += [latencia, fiabilidad]
    else:
        cifras.append(Cifra("latencia de ingesta", f"< {LATENCY_BUDGET_S:.0f} s", None,
                            "sin casos", None, ["pasa `--casos DIR` con carpetas reales"]))
    cifras.append(mide_mallas(args.mallas, args.n_mallas, args.salida))
    cifras.append(mide_reversibilidad(args.mallas, args.n_mallas, args.salida))
    cifras.append(mide_cobertura(args.cobertura))

    print("=" * 78)
    print(f"METRICAS DEL BRIEF · {datetime.now(UTC):%Y-%m-%d %H:%M} UTC · commit {_commit()}")
    print("=" * 78)
    for c in cifras:
        print(c.linea())
        for n in c.notas:
            print(f"        · {n}")
        print(f"        sobre: {c.sobre}")
    print("-" * 78)
    fuera = [c.nombre for c in cifras if c.cumple is False]
    sin = [c.nombre for c in cifras if c.cumple is None]
    print(f"cumplen {sum(1 for c in cifras if c.cumple)} de {len(cifras)}"
          + (f" · FUERA: {', '.join(fuera)}" if fuera else "")
          + (f" · sin medir: {', '.join(sin)}" if sin else ""))
    print(f"medido en {time.perf_counter() - arranque:.1f} s")

    if args.json:
        args.json.write_text(json.dumps(
            {"fecha": datetime.now(UTC).isoformat(), "commit": _commit(),
             "cifras": [asdict(c) for c in cifras], "casos": detalle},
            indent=2, ensure_ascii=False))
        print(f"detalle en {args.json}")
    return 1 if fuera else 0


if __name__ == "__main__":
    raise SystemExit(main())
