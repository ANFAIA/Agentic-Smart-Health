#!/usr/bin/env python
"""caso_completo.py — El pipeline entero sobre un caso clínico real, etapa por etapa.

    uv run python scripts/caso_completo.py --caso ~/anfaia/histora/another_patient

**Qué prueba, y por qué hace falta.** El recorrido ingesta → fusión → segmentación →
export tiene prueba de integración
([`apps/agent-orchestrator/tests/test_e2e.py`](../apps/agent-orchestrator/tests/test_e2e.py)),
pero corre sobre el **caso sintético**: una arcada paramétrica con su serie DICOM y un
informe generado. Eso demuestra que el recorrido existe, no que aguante dato clínico de
verdad — con su metal, su ruido, sus informes de proveedor y sus cuatro modalidades
llegando en formatos que nadie eligió para nosotros.

Este script apunta el orquestador a una carpeta de paciente y **reporta lo que pasa en
cada etapa**, incluido lo que falla. No afirma nada: imprime estado, confianza y motivos
de revisión humana, que es lo que los agentes devuelven.

**Los ficheros se resuelven por patrón.** Los nombres del proveedor traen apellidos del
paciente y número de caso; escribirlos aquí los publicaría. Misma regla que en
`desplazamiento_relativo.py` y `composicion_cbct_ios.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
for paquete in ("core-schemas", "ingestion-agents", "fusion-agents", "analysis-agents",
                "export-agents", "tooth-aggregation"):
    sys.path.insert(0, str(RAIZ / f"packages/{paquete}/src"))
sys.path.insert(0, str(RAIZ / "apps/agent-orchestrator/src"))

from agent_orchestrator import CaseInput, IngestionPipeline  # noqa: E402
from ingestion_agents import ArtifactStore  # noqa: E402

FOTOS = ("*.jpg", "*.jpeg", "*.png")

# Umbral para la nube con la que se localiza el plano oclusal.
#
# ⚠️ NO son los 2000 HU de `registro_ios_cbct`: alli el umbral se aplica al DICOM crudo,
# que llega a 3476 HU. Aqui se aplica al campo del `cbct-agent`, que SATURA a 2000, asi
# que `>= 2000` selecciona solo los vertices saturados —2.507 de 493.932— y el plano
# oclusal sale de una nube demasiado escasa para ser fiable.
HU_CRESTA = 1500.0


def _texto_extraible(pdf: Path) -> int:
    """Caracteres que `pdftotext` saca del PDF, o 0 si no hay binario ni texto."""
    import subprocess
    try:
        r = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return 0
    return len(b"".join(r.stdout.split()))


def _informes_por_texto(raiz: Path) -> list[Path]:
    """Los PDF ordenados por cantidad de texto, de mas a menos.

    ⚠️ Ordenar por nombre eligia `BRN3C2AF...` —un escaneado sin capa de texto, 0
    caracteres— y dejaba sin tocar el informe de IA con los hallazgos por diente. El
    `report-agent` fallaba correctamente («no contiene texto extraible») sobre un fichero
    que ni siquiera era el informe. Un carpeta de clinica trae papeleo mezclado con lo
    clinico, y el nombre no distingue uno de otro.
    """
    conteo = [(p, _texto_extraible(p)) for p in sorted(raiz.glob("*.pdf"))]
    return [p for p, n in sorted(conteo, key=lambda x: -x[1]) if n > 0]


def descubre(raiz: Path) -> CaseInput:
    """Las cuatro modalidades, localizadas por forma y no por nombre.

    El layout de una carpeta de clínica no es el de `synthetic.write_case`: el CBCT llega
    en un subdirectorio con nombre de caso, la malla es STL y no OBJ, y los informes
    conviven con papeleo que no es clínico. `CaseInput.from_case_dir` no lo cubre, y
    forzarlo a cubrirlo metería en el contrato el layout de un proveedor concreto.
    """
    cbct = next((d for d in sorted(raiz.glob("*_files")) if d.is_dir()
                 and any(d.glob("*.dcm"))), None)
    mallas = sorted(p for p in raiz.glob("*.stl")
                    if "Unsectioned" not in p.name)  # el Unsectioned es derivado del CBCT
    informes = _informes_por_texto(raiz)
    fotos = sorted(p for pat in FOTOS for p in raiz.glob(pat))

    return CaseInput(
        acquisition_id=raiz.name,
        cbct=cbct,
        mesh=mallas[0] if mallas else None,
        report=informes[0] if informes else None,
        images=fotos,
    )


def linea(o) -> str:
    conf = f"{o.provenance.confidence:.2f}" if getattr(o, "provenance", None) else "—"
    return (f"  {o.agent.split('@')[0]:<24} {o.status.value:<8} conf {conf:<5} "
            f"{(o.detail or '')[:70]}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--caso", type=Path, required=True)
    ap.add_argument("--salida", type=Path, default=RAIZ / "data/processed/caso-completo")
    args = ap.parse_args()

    caso = descubre(args.caso)
    print("=" * 78)
    print(f"CASO: {caso.acquisition_id}")
    print("=" * 78)
    print(f"  CBCT     {caso.cbct.name if caso.cbct else '— no encontrado'}"
          + (f"  ({len(list(caso.cbct.glob('*.dcm')))} cortes)" if caso.cbct else ""))
    print(f"  malla    {'sí' if caso.mesh else '— no encontrada'}")
    print(f"  informe  {'sí' if caso.report else '— no encontrado'}")
    print(f"  fotos    {len(caso.images)}")

    pipe = IngestionPipeline(ArtifactStore(args.salida / "artifacts"),
                            quarantine_dir=args.salida / "quarantine")

    print("\n--- 1 · INGESTA ---")
    r = pipe.run(caso)
    for o in r.outcomes:
        print(linea(o))
    print(f"  → latencia {r.latency_s:.1f} s (presupuesto 60) · "
          f"snapshot {'sí' if r.snapshot else 'NO'}")
    if r.snapshot is None:
        print("\n✗ Sin campo gaussiano no hay TwinSnapshot. El recorrido para aquí.")
        for m in r.hitl_reasons:
            print(f"    · {m}")
        return 1
    print(f"  → {r.snapshot.n_primitives:,} primitivas · "
          f"{len(r.snapshot.regional)} observación(es) regional(es)")

    print("\n--- 2 · FUSIÓN (registro malla ↔ campo) ---")
    malla = pipe.store.load(r.snapshot.surface_ref)["positions"] if r.snapshot.surface_ref else None
    campo = pipe.store.load(r.snapshot.gaussian_field_ref)
    if malla is None:
        print("  sin malla: no hay nada que registrar")
        fus = r
    else:
        import numpy as np
        from export_agents.field import densidad_a_hu

        sys.path.insert(0, str(RAIZ / "scripts"))
        from registro_ios_cbct import arcada_del_escaneo, separa_arcadas

        centros = campo["centers"].astype(np.float64) + campo["origin"]

        # El CBCT trae las DOS arcadas y la malla es una sola. Registrar contra el
        # volumen entero mezcla maxilar con mandibula: la primera pasada dio 0,778 mm
        # sobre el 5,9 % solapado, que el propio gate marco como no interpretable.
        # Cual es cual lo dice el nombre del fichero cruzado con el `IPP` del DICOM —
        # nunca el residuo, que esta medido que NO discrimina (0,490 vs 0,509 mm).
        hu = densidad_a_hu(campo["density"], campo["hu_range"])
        cresta = centros[hu >= HU_CRESTA]
        z_alta, z_baja, corte = separa_arcadas(cresta)
        quiere = arcada_del_escaneo(caso.mesh)

        # El objetivo es la CRESTA del lobulo, no el lobulo entero.
        #
        # ⚠️ Corregido tras medirlo: registrando contra todo el tejido duro del lobulo
        # (322.195 gaussianas) el solape salia del 7,4 %, y el gate lo marcaba como no
        # interpretable — con razon. Pero no era un fallo del registro: la malla del
        # escaner ve SUPERFICIE DE CORONA, y la mayor parte de esas gaussianas son hueso e
        # interior, que ningun escaneo intraoral puede ver. Se estaba midiendo el solape
        # contra una nube cuya mayoria no tiene contrapartida posible.
        if quiere == "maxilar":
            objetivo = z_alta
        elif quiere == "mandibular":
            objetivo = z_baja
        else:
            objetivo = cresta
        print(f"  arcada del escaneo: {quiere or 'INDETERMINADA — se registra contra todo'} "
              f"· plano oclusal z={corte:.1f} · cresta {len(cresta):,} pts "
              f"({len(z_alta):,} alta / {len(z_baja):,} baja) · objetivo "
              f"{len(objetivo):,} puntos")

        paso_m = max(1, len(malla) // 4000)
        paso_c = max(1, len(objetivo) // 4000)
        fus = pipe.fuse(r, registration=(np.asarray(malla, dtype=np.float64)[::paso_m],
                                         objetivo[::paso_c]))
        for o in fus.fusion:
            print(linea(o))

    print("\n--- 3 · EXPORTACIÓN ---")
    fin = pipe.exportar(fus, args.salida / "export")
    for e in fin.exports:
        print(f"  {e.agent.split('@')[0]:<24} {e.status.value:<8} "
              f"{'' if e.max_deviation_mm is None else f'{e.max_deviation_mm:.6f} mm'}"
              f"{'' if e.psnr_db is None else f'PSNR {e.psnr_db:.1f} dB'}")
    print(f"  → reversible: {'sí' if fin.reversible else 'NO'}")

    print("\n--- 4 · GATE DE REVISIÓN HUMANA ---")
    if not fin.hitl_reasons:
        print("  sin motivos: el caso pasaría sin revisión")
    for m in fin.hitl_reasons:
        print(f"  · {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
