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
from dataclasses import replace
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
for paquete in ("core-schemas", "ingestion-agents", "fusion-agents", "analysis-agents",
                "export-agents", "tooth-aggregation"):
    sys.path.insert(0, str(RAIZ / f"packages/{paquete}/src"))
sys.path.insert(0, str(RAIZ / "apps/agent-orchestrator/src"))

from agent_orchestrator import CaseInput, IngestionPipeline  # noqa: E402
from core_schemas import ContratoEtapa, revisa_conservacion  # noqa: E402
from fusion_agents import arcada_del_nombre  # noqa: E402
from ingestion_agents import ArtifactStore  # noqa: E402

FOTOS = ("*.jpg", "*.jpeg", "*.png")


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


def _fabrica_segmentador(caso, args):
    """Devuelve la fábrica que el orquestador llamará **después** de registrar.

    El trabajo caro —ingerir el CBCT e inferir el volumen de probabilidad— se hace una vez,
    aquí. Lo que queda dentro de la fábrica es solo mover las coronas con la pose que la
    fusión geométrica acaba de calcular.

    ⚠️ **Y esa pose no se recalcula: se lee de `provenance.transform`.** Antes esta función
    registraba por su cuenta, así que la segmentación podía nombrar dientes con una pose
    distinta de la que el snapshot declara y de la que se exporta en el STL — dos verdades
    sobre el mismo paciente, sin que nada las comparase. La transformación va de escáner a
    twin y se aplica en el sentido en que se midió (ver `ExportAgent._to_twin_frame`).
    """
    import numpy as np
    from analysis_agents import SegmentadorDental
    from fusion_agents.registration import apply, quaternion_to_matrix
    from ingestion_agents import ArtifactStore, CBCTAgent
    from ingestion_agents.cbct_agent import _read_series
    from ingestion_agents.mesh_agent import parse_stl

    sys.path.insert(0, str(RAIZ / "scripts"))
    from composicion_cbct_ios import probabilidad_por_modelo

    almacen = ArtifactStore(args.salida / "_seg")
    campo = almacen.load(CBCTAgent(almacen).ingest(caso.cbct).artifact_ref)
    origen_mm = campo["origin"]

    V = np.asarray(parse_stl(caso.mesh)["positions"], dtype=np.float64)
    etq = np.load(args.fdi).astype(np.int64)

    serie = _read_series(caso.cbct)
    prob = probabilidad_por_modelo(serie.volume, args.modelo,
                                   espaciado=np.asarray(serie.spacing))
    sx, sy, _z = serie.spacing
    z_ord = np.sort(serie.z)

    def probabilidad_en(puntos):
        """De mm a vóxel. El `cbct-agent` construye los centros como
        `(col*sx, fila*sy, z[corte])` y luego resta el centroide, así que se deshace."""
        p = np.asarray(puntos, dtype=np.float64) + origen_mm
        col = np.clip(np.rint(p[:, 0] / sx).astype(int), 0, prob.shape[2] - 1)
        fil = np.clip(np.rint(p[:, 1] / sy).astype(int), 0, prob.shape[1] - 1)
        cor = np.clip(np.searchsorted(z_ord, p[:, 2]), 0, prob.shape[0] - 1)
        return prob[cor, fil, col]

    def fabrica(snapshot):
        t = snapshot.provenance.transform
        if t is None:
            print("  segmentador: el snapshot no trae transformación — sin registro no "
                  "se puede saber DÓNDE está cada corona, así que no se segmenta.")
            return None
        # Al marco CENTRADO, que es en el que el agente pasa los puntos: el registro deja
        # las coronas en coordenadas absolutas del CBCT y `campo["centers"]` no las lleva.
        coronas = apply(
            quaternion_to_matrix(t.rotation), np.asarray(t.translation), V
        ) - origen_mm
        print(f"  segmentador: {args.modelo.name} · {len(coronas):,} coronas con la pose "
              f"de la fusión · {len(set(etq[etq > 0].tolist()))} códigos FDI")
        return SegmentadorDental(probabilidad_en, coronas, etq)

    return fabrica


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
    ap.add_argument(
        "--refina-3dgs", action="store_true",
        help="Optimiza el campo semilla como 3DGS contra los DRR del volumen (necesita "
             "GPU: usar ~/.venvs/dental-gpu/bin/python). Sin esto, el twin se exporta "
             "TAL COMO LO SEMBRO el `cbct-agent`, que es lo que ha pasado hasta hoy.",
    )
    ap.add_argument("--pasos-3dgs", type=int, default=400)
    ap.add_argument(
        "--modelo", type=Path, default=None,
        help="Checkpoint del segmentador de CBCT. Con el, la etapa de SEGMENTACION corre "
             "de verdad y puebla `region_id`; sin el no hay `Segmenter` y la etapa "
             "sencillamente no se ejecuta, que es lo que ha pasado hasta hoy.",
    )
    ap.add_argument(
        "--fdi", type=Path, default=None,
        help="`region_id` por vertice del escaneo intraoral. Es la mitad que dice CUAL es "
             "cada diente: el modelo del CBCT es binario y no puede darla.",
    )
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

    # Se pasa una FÁBRICA, no un `Segmenter`: la segmentación necesita las coronas ya
    # movidas al marco del CBCT, y esa pose no existe hasta que `fuse()` registra.
    fabrica = None
    if args.modelo and args.fdi and caso.mesh and caso.cbct:
        fabrica = _fabrica_segmentador(caso, args)

    pipe = IngestionPipeline(ArtifactStore(args.salida / "artifacts"),
                            quarantine_dir=args.salida / "quarantine",
                            segmenter_factory=fabrica)

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
    if caso.mesh is None or r.snapshot.surface_ref is None:
        print("  sin malla: no hay nada que registrar")
        fus = r
    else:
        # Las dos nubes las elige el ORQUESTADOR, no este script.
        #
        # ⚠️ Aquí vivían 40 líneas de pegamento: aislar la arcada del lóbulo que toca,
        # quedarse con la cresta, submuestrear. Estaba bien medido y estaba en el sitio
        # equivocado — un caso clínico no se podía pasar por el pipeline sin que una
        # persona reescribiera eso. Ahora es `IngestionPipeline.prepara_registro`, con
        # sus tests, y lo que tuvo que dar por bueno entra en el gate de revisión en vez
        # de quedarse en un `print`. Ver `fusion_agents.preparacion`.
        print(f"  arcada del escaneo: "
              f"{arcada_del_nombre(caso.mesh) or 'INDETERMINADA'}")
        # `dos_arcadas=True` lo declara este script porque lo SABE: un CBCT dental de
        # FOV completo trae maxilar y mandíbula, y el escaneo intraoral es de una sola.
        # El dato no puede deducirlo —en oclusión no hay valle que encontrar—, así que lo
        # dice quien tiene el contexto. Ver `fusion_agents.preparacion`.
        fus = pipe.fuse(r, malla=caso.mesh, dos_arcadas=True)
        for o in fus.fusion:
            print(linea(o))
        for a in fus.analysis:
            print(linea(a))
        seg = next((a for a in fus.analysis if a.agent.startswith("segmentation")), None)
        if seg is not None:
            print(f"  → {seg.n_teeth} diente(s) con `region_id` · "
                  f"{seg.unassigned_fraction:.1%} sin asignar")

    if args.refina_3dgs:
        print("\n--- 3 · GAUSSIAN SPLATTING (refinado contra los DRR del volumen) ---")
        # La fase que el ADR 001 describía y que el recorrido nunca ejecutaba: hasta hoy
        # se saltaba de la semilla del `cbct-agent` a exportación, así que el gemelo
        # digital NO se había entrenado nunca como 3DGS.
        #
        # Va aquí y no dentro de `fuse()` porque necesita GPU y torch, y meterlo en el
        # orquestador obligaría a todos sus llamantes a ese entorno. Lo que sí se respeta
        # es el contrato: el campo refinado es un artefacto NUEVO, así que pasa por
        # `revisa_conservacion` igual que cualquier otra etapa.
        from ingestion_agents.cbct_agent import _read_series
        from refina_3dgs import refina

        campo = pipe.store.load(fus.snapshot.gaussian_field_ref)
        arrays, informe = refina(campo, _read_series(caso.cbct),
                                 pasos=args.pasos_3dgs, registro=lambda m: print(f"  {m}"))
        motivos = revisa_conservacion(
            ContratoEtapa(nombre="refinado-3dgs"), campo, arrays
        )
        nuevo_ref = pipe.store.put(**arrays)
        fus = replace(
            fus,
            snapshot=fus.snapshot.model_copy(update={"gaussian_field_ref": nuevo_ref}),
            hitl_reasons=[*fus.hitl_reasons, *motivos],
        )
        print(f"  → semilla {informe['psnr_semilla_db']:.2f} dB → refinado "
              f"{informe['psnr_refinado_db']:.2f} dB ({informe['delta_db']:+.2f}) sobre "
              f"{informe['vistas_retenidas']} vistas RETENIDAS")
        print("  → " + ("APORTA" if informe["aporta"] else "NO aporta sobre la semilla"))

    print("\n--- 4 · EXPORTACIÓN ---")
    fin = pipe.exportar(fus, args.salida / "export")
    for e in fin.exports:
        print(f"  {e.agent.split('@')[0]:<24} {e.status.value:<8} "
              f"{'' if e.max_deviation_mm is None else f'{e.max_deviation_mm:.6f} mm'}"
              f"{'' if e.psnr_db is None else f'PSNR {e.psnr_db:.1f} dB'}")
    print(f"  → reversible: {'sí' if fin.reversible else 'NO'}")

    print("\n--- 5 · GATE DE REVISIÓN HUMANA ---")
    if not fin.hitl_reasons:
        print("  sin motivos: el caso pasaría sin revisión")
    for m in fin.hitl_reasons:
        print(f"  · {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
