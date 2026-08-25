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
# Todos los paquetes del workspace, descubiertos. Estaba escrito a mano y se olvido
# `gaussian-engine` al anadirlo: el caso corrio los tres primeros minutos y murio en la
# etapa de ajuste con un ModuleNotFoundError. Una lista de paquetes que hay que acordarse
# de actualizar es una lista que se queda vieja.
for src in sorted(RAIZ.glob("packages/*/src")):
    sys.path.insert(0, str(src))
sys.path.insert(0, str(RAIZ / "apps/agent-orchestrator/src"))

from agent_orchestrator import CaseInput, IngestionPipeline  # noqa: E402
from core_schemas import ContratoEtapa, revisa_conservacion  # noqa: E402
from fusion_agents import arcada_del_nombre  # noqa: E402
from ingestion_agents import ArtifactStore  # noqa: E402

FOTOS = ("*.jpg", "*.jpeg", "*.png")


def _submuestreo_desde(store: ArtifactStore, snapshot: object, n_final: int) -> dict:
    """Construye el dict de submuestreo desde el artefacto del campo semilla.

    El `cbct-agent` guarda `paso` (array de 3 enteros en orden z,y,x) y `n_origen`
    (total de vóxeles antes de submuestrear). Esta función los lee y los formatea
    para el sidecar del `.uos`.
    """
    import numpy as np

    if snapshot.gaussian_field_ref is None:
        return {"paso_voxeles": [1, 1, 1], "de": n_final, "a": n_final}
    try:
        datos = store.load(snapshot.gaussian_field_ref)
    except (KeyError, OSError, ValueError):
        return {"paso_voxeles": [1, 1, 1], "de": n_final, "a": n_final}
    if "paso" not in datos or "n_origen" not in datos:
        return {"paso_voxeles": [1, 1, 1], "de": n_final, "a": n_final}
    paso_arr = np.asarray(datos["paso"], dtype=int)
    # `paso` viene en orden (z, y, x); lo pasamos a (x, y, z)
    return {
        "paso_voxeles": paso_arr[::-1].tolist(),
        "de": int(datos["n_origen"]),
        "a": n_final,
    }


def _texto_extraible(pdf: Path) -> int:
    """Caracteres que `pdftotext` saca del PDF, o 0 si no hay binario ni texto."""
    import subprocess
    try:
        r = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return 0
    return len(b"".join(r.stdout.split()))


def _informes(raiz: Path) -> list[Path]:
    """**Todos** los PDF, ordenados por cantidad de texto de mas a menos.

    Ordenar importa —el primero es el mas informativo y es el que se lee antes en los
    logs— pero **no se filtra nada**, y ese es el arreglo.

    ⚠️ Antes esta funcion devolvia solo los que tenian texto, asi que un PDF escaneado sin
    capa de texto desaparecia **antes de que ningun agente lo viera**. Y el `report-agent`
    ya sabe declararlo («no contiene texto extraible»): filtrar aqui era quitarle el
    trabajo al agente para hacerlo peor, en silencio. Que haya un PDF ilegible en la
    carpeta de un paciente es justo lo que un clinico tiene que saber.

    ⚠️ Y ordenar por NOMBRE tampoco vale: elegia `BRN3C2AF...` —el escaneado— y dejaba el
    informe de IA con los hallazgos por diente para el final. Una carpeta de clinica trae
    papeleo mezclado con lo clinico, y el nombre no distingue uno de otro.
    """
    conteo = [(p, _texto_extraible(p)) for p in sorted(raiz.glob("*.pdf"))]
    return [p for p, _ in sorted(conteo, key=lambda x: -x[1])]


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
    informes = _informes(raiz)
    fotos = sorted(p for pat in FOTOS for p in raiz.glob(pat))

    return CaseInput(
        acquisition_id=raiz.name,
        cbct=cbct,
        mesh=mallas[0] if mallas else None,
        reports=informes,
        images=fotos,
    )


def _fabrica_segmentador(caso, args, almacen, etq):
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
    from ingestion_agents.cbct_agent import _read_series
    from ingestion_agents.mesh_agent import parse_stl

    sys.path.insert(0, str(RAIZ / "scripts"))
    from composicion_cbct_ios import probabilidad_por_modelo

    V = np.asarray(parse_stl(caso.mesh)["positions"], dtype=np.float64)

    serie = _read_series(caso.cbct)
    prob = probabilidad_por_modelo(serie.volume, args.modelo,
                                   espaciado=np.asarray(serie.spacing))
    sx, sy, _z = serie.spacing
    z_ord = np.sort(serie.z)

    def sonda(origen_mm):
        """De mm a vóxel, para el `origin` que el campo del snapshot declare.

        Va parametrizada por el origen y no cerrada sobre uno fijo: el centroide depende
        de qué vóxeles entraron al campo, así que cambia con el recorte dental. Cerrarla
        sobre el de otra ingesta es lo que desplomó el nombrado de 14 dientes a 1.
        """

        def probabilidad_en(puntos):
            p = np.asarray(puntos, dtype=np.float64) + origen_mm
            col = np.clip(np.rint(p[:, 0] / sx).astype(int), 0, prob.shape[2] - 1)
            fil = np.clip(np.rint(p[:, 1] / sy).astype(int), 0, prob.shape[1] - 1)
            cor = np.clip(np.searchsorted(z_ord, p[:, 2]), 0, prob.shape[0] - 1)
            return prob[cor, fil, col]

        return probabilidad_en

    def fabrica(snapshot):
        # ⚠️ El `origin` se lee DEL CAMPO QUE EL PIPELINE YA SEMBRO, no de una ingesta
        # propia. Aqui se reingeria el CBCT con los ajustes por defecto, y en cuanto el
        # orquestador empezo a recortar a la region dental los dos centroides dejaron de
        # coincidir: las coronas caian en otro sitio y el nombrado se desplomo de 14
        # dientes a 1. Un agente que reconstruye por su cuenta lo que ya esta en el
        # snapshot acaba discrepando de el.
        origen_mm = almacen.load(snapshot.gaussian_field_ref)["origin"]
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
        # Hacia donde va la raiz, MEDIDO del propio escaneo y no supuesto.
        #
        # El margen gingival es el lado del hueso: la encia rodea las coronas por donde
        # sale la raiz. Asi que `media(encia) - media(coronas)` apunta hacia el hueso, sin
        # necesidad de saber la convencion de ejes del DICOM ni si la arcada es la de
        # arriba o la de abajo — vale para las dos. Medido en este caso: [-1,3 4,1 5,4] mm.
        #
        # Hace falta porque sin ella el nombrado funde cada diente con el que lo ocluye:
        # piezas de 44 y 47 mm cruzando la encia. Ver `TOLERANCIA_OCLUSAL_MM`.
        direccion = coronas[etq == 0].mean(axis=0) - coronas[etq > 0].mean(axis=0)
        print(f"  segmentador: {args.modelo.name} · {len(coronas):,} coronas con la pose "
              f"de la fusión · {len(set(etq[etq > 0].tolist()))} códigos FDI")
        return SegmentadorDental(
            sonda(origen_mm), coronas, etq, direccion_raiz=direccion,
            # ⚠️ El recorte apical convierte el ápice en SUPUESTO. Se pide aquí, en el
            # script que monta el caso, y no por defecto en el agente: quien lo active
            # tiene que saber que a partir de ese momento la longitud de la raíz ya no se
            # puede medir sobre el resultado, porque sería medir lo que se ha supuesto.
            recorta_por_longitud=not args.sin_recorte_apical,
        )

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
        "--max-primitivas", type=int, default=1_500_000,
        help="Tope de gaussianas del campo. El defecto del agente (500.000) esta pensado "
             "para un FOV acotado; sobre una cabeza entera deja los dientes con el 7%% de "
             "su volumen.",
    )
    ap.add_argument(
        "--gs-apariencia", type=Path, default=None,
        help="Directorio con el escaner ya entrenado como 3DGS (lo produce "
             "scripts/entrena_gs_escaner.py). Anade dos capas de APARIENCIA al paquete "
             "del visor, llevadas al marco del twin. No sustituye a las capas medidas.",
    )
    ap.add_argument(
        "--entrena-apariencia", action="store_true",
        help="Entrena 3DGS contra fotos intraorales para obtener color REAL del paciente "
             "(gsplat + Blender, necesita GPU). El campo resultante viaja como "
             "`asset.apariencia` en el .uos con perfil 'ash-gs-apariencia/1.0'. "
             "⚠️ REQUIERE `--gs-apariencia` y fotos en image_refs.",
    )
    ap.add_argument(
        "--ajusta-campo", action="store_true",
        help="Ajustar elipsoides anisotropos a la densidad medida. El campo resultante es "
             "DERIVADO: su escala deja de ser el voxel que produjo la gaussiana y pasa a "
             "ser la forma que reconstruye la densidad, asi que NO se puede medir encima.",
    )
    ap.add_argument(
        "--compresion", type=float, default=13.0,
        help="Compresion del FONDO (la region sin nombre: hueso y craneo). Por defecto 13.",
    )
    ap.add_argument(
        "--compresion-dientes", type=float, default=2.0,
        help="Compresion de las piezas con nombre. Por defecto 1: mismo numero de "
             "gaussianas que semillas, pero elipsoides ajustados en vez de esferas del "
             "tamano del voxel. Medido: de 2 a 13 el error es PLANO (45-47 HU), asi que "
             "se elige por resolucion — con 13 el espaciado sube a 0,908 mm y una raiz de "
             "4 mm son cuatro gaussianas; con 2 son diez. Por debajo de 2 es degenerado.",
    )
    ap.add_argument(
        "--sin-recorte-apical", action="store_true",
        help="No recortar la raiz a la longitud anatomica de su tipo. Sin el recorte las "
             "piezas arrastran hueso alveolar (medido: 34,3 mm en un molar de ~20); con "
             "el, el apice pasa a ser SUPUESTO y no se puede medir longitud radicular.",
    )
    ap.add_argument(
        "--sin-recorte-dental", action="store_true",
        help="No acotar el campo a la dentadura. Solo para un CBCT que YA venga acotado: "
             "sobre uno de cabeza entera el compuesto sale en filamentos.",
    )
    ap.add_argument(
        "--modelo", type=Path, default=None,
        help="Checkpoint del segmentador de CBCT. Con el, la etapa de SEGMENTACION corre "
             "de verdad y puebla `region_id`; sin el no hay `Segmenter` y la etapa "
             "sencillamente no se ejecuta, que es lo que ha pasado hasta hoy.",
    )
    ap.add_argument(
        "--sin-originales", action="store_true",
        help="Perfil LIGERO: el .uos lleva el campo gaussiano, la escena y el manifiesto, "
             "y los originales —STL del escáner y serie DICOM— se declaran como assets "
             "externos con su sha256 sin viajar dentro. ⚠️ Cambia la GARANTÍA: con ellos "
             "dentro el contenedor afirma «lo que sale es lo que entró» y el validador lo "
             "comprueba; sin ellos afirma «sé el hash de lo que debería haber ahí», y deja "
             "de cumplir el §1.1 del spec. El validador avisa por cada asset externo.",
    )
    ap.add_argument(
        "--con-volumen", action="store_true",
        help="Mete la serie DICOM ENTERA en el .uos y sube su conformidad a UOS-Vol. "
             "Va detrás de una bandera por PESO: la serie de un CBCT son cientos de "
             "megas y multiplica por diez el tamaño del contenedor. Sin esto el .uos es "
             "UOS-Core y lo declara, en vez de fingir que lleva el volumen.",
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
    print(f"  informes {len(caso.reports)} (los ilegibles NO se filtran: los declara "
          f"el `report-agent`)")
    print(f"  fotos    {len(caso.images)}")

    # Se pasa una FÁBRICA, no un `Segmenter`: la segmentación necesita las coronas ya
    # movidas al marco del CBCT, y esa pose no existe hasta que `fuse()` registra.
    almacen = ArtifactStore(args.salida / "artifacts")
    # ⚠️ Las etiquetas se cargan y se rellenan UNA VEZ, y las usan los dos: el segmentador
    # y el canal del visor. Estaban duplicadas —el segmentador con los huecos cerrados y
    # el visor con las crudas— y las dos mitades discrepaban sobre qué vértice es corona.
    etq_ios = None
    if args.fdi and caso.mesh:
        import numpy as _np
        from analysis_agents import absorbe_islas as _absorbe
        from analysis_agents import afina_fronteras as _afina
        from analysis_agents import quita_motas as _motas
        from analysis_agents import rellena_etiquetas as _rellena
        from analysis_agents import rellena_huecos_interiores as _huecos
        from ingestion_agents.mesh_agent import parse_stl as _stl

        _V = _np.asarray(_stl(caso.mesh)["positions"], dtype=_np.float64)
        _cruda = _np.load(args.fdi).astype(_np.int64)
        # ⚠️ Comprobado ANTES de usarlo, y con el número delante. Pasar las etiquetas de
        # otro paciente no da un resultado raro: da un `IndexError` doscientas líneas más
        # abajo, dentro de un KD-tree, que no dice nada de lo que pasó. Las etiquetas
        # indexan los vértices DEDUPLICADOS de esta malla y de ninguna otra.
        if len(_cruda) != len(_V):
            print(f"  ✗ `--fdi` trae {len(_cruda):,} etiquetas y esta malla tiene "
                  f"{len(_V):,} vértices deduplicados: no son del mismo escaneo.")
            return 1
        etq_ios = _rellena(_V, _cruda)
        cerrados = int((etq_ios > 0).sum()) - int((_cruda > 0).sum())
        if cerrados:
            print(f"  etiquetas del escáner: {int((_cruda > 0).sum()):,} → "
                  f"{int((etq_ios > 0).sum()):,} vértices de corona "
                  f"(+{cerrados:,} huecos cerrados)")
        # ⚠️ Va DESPUÉS de rellenar y ANTES de que nadie las use: estas etiquetas son las
        # SEMILLAS con las que se nombra el CBCT, así que una isla mal etiquetada aquí se
        # convierte allí en una pieza entera que viaja al visor, a las vistas y a la capa
        # clínica. El sitio barato de matarla es éste.
        etq_ios, _islas = _absorbe(_V, etq_ios)
        for _origen, _destino, _n in _islas:
            print(f"  ⚠ FDI {_origen}: {_n:,} vértices metidos dentro del {_destino}, no "
                  f"son una pieza. Absorbidos — el caso pasa a tener "
                  f"{len({int(x) for x in etq_ios if x > 0})} dientes.")
        # El contacto interproximal no tiene borde geométrico, así que la etiqueta sale
        # difuminada y encender una pieza enciende un trozo de la vecina. Ver
        # `afina_fronteras`: NO toca el margen gingival, que sí es una frontera clínica.
        etq_ios, _movidos = _afina(_V, etq_ios)
        if _movidos:
            print(f"  frontera entre piezas contiguas afilada: {_movidos:,} vértices "
                  f"reasignados a su vecino por mayoría")
        # Y el simétrico de `rellena_etiquetas`: un vértice de diente rodeado de encía es
        # encía. Sin esto quedan motas color hueso salpicadas sobre el rosa del visor.
        etq_ios, _motitas = _motas(_V, etq_ios)
        if _motitas:
            print(f"  motas de diente en mitad de la encía: {_motitas:,} quitadas "
                  f"(el margen gingival no se toca)")
        # Y los huecos que el relleno por vecindario no alcanza porque son mayores que el
        # vecindario. El criterio aquí es la distancia a encía de verdad, no los vecinos:
        # un vértice del margen está pegado a la encía y por eso nunca entra.
        etq_ios, _dentro = _huecos(_V, etq_ios)
        if _dentro:
            print(f"  huecos de encía en mitad de una corona: {_dentro:,} rellenados "
                  f"(a más de 1,5 mm de cualquier encía real)")

    fabrica = None
    if args.modelo and args.fdi and caso.mesh and caso.cbct:
        fabrica = _fabrica_segmentador(caso, args, almacen, etq_ios)

    pipe = IngestionPipeline(almacen,
                            quarantine_dir=args.salida / "quarantine",
                            cbct_recorte_dental=not args.sin_recorte_dental,
                            cbct_max_primitivas=args.max_primitivas,
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
          f"{len(r.snapshot.regional)} observación(es) regional(es) · "
          f"{len(r.snapshot.medidas)} medida(s) no regional(es)")
    for m in r.snapshot.medidas:
        marca = "⚠ FUERA" if m.fuera_de_rango else ""
        print(f"      {m.nombre:<10} {m.valor:>7.2f}{m.unidad:<2} {m.lado or ' ':<2}"
              f"[{m.normal_min:g}, {m.normal_max:g}] {marca}")

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

    # Referencia al campo ajustado y su informe de ajuste, para que el `.uos` lleve
    # las DOS capas: la semilla medida (en `fus.snapshot`) y la ajustada (aquí).
    campo_ajustado_ref = None
    ajuste_info = None
    descriptor_ajustado = None

    if args.ajusta_campo:
        print("\n--- 3b · AJUSTE DEL CAMPO (elipsoides contra la densidad medida) ---")
        # Complementario de `--refina-3dgs`, no alternativo: aquel optimiza contra los DRR
        # del volumen y este contra la densidad de las propias semillas, sin renderizador
        # de por medio. Aquí la pérdida sale en HU, que es la unidad del dato.
        #
        # Va DESPUÉS de la segmentación a propósito. Con `region_id` el ajuste se hace
        # región a región, y entonces la etiqueta de cada elipsoide es exacta por
        # construcción en vez de heredada del vecino más cercano — que es lo que el visor
        # necesita para poder seleccionar una pieza sin mentir sobre cuál es.
        #
        # ⚠️ **El snapshot NO se sustituye.** `ajusta_campo` devuelve un snapshot nuevo
        # apuntando al campo ajustado, pero `fus.snapshot` sigue apuntando a la SEMILLA
        # medida — que es la que viaja en el twin y sobre la que se mide la
        # reversibilidad. El campo ajustado es DERIVADO y va aparte en el `.uos`, como
        # `asset.field_fit` con `measured: false`. Si sustituyéramos, el `.uos` llevaría
        # el ajustado SIN la semilla, que es exactamente al revés de lo que el formato
        # defiende. Ver `packages/uos/src/uos/agente.py`.
        from gaussian_engine import ajusta_campo

        antes = pipe.store.load(fus.snapshot.gaussian_field_ref)
        # Sin `region_id` (caso sin segmentación), `ajusta_campo` necesita
        # `n_objetivo` para saber cuántas gaussianas pedir. El objetivo sale del
        # tamaño del campo semilla dividido por la compresión.
        n_obj = None
        if "region_id" not in antes:
            n_obj = max(1, int(len(antes["centers"]) / args.compresion))
        snap_ajustado, aj = ajusta_campo(
            fus.snapshot, pipe.store,
            n_objetivo=n_obj,
            compresion=args.compresion,
            compresion_region=args.compresion_dientes,
        )
        motivos_aj = revisa_conservacion(
            ContratoEtapa(nombre="ajuste-campo"),
            antes, pipe.store.load(snap_ajustado.gaussian_field_ref),
        )
        fus = replace(fus, hitl_reasons=[*fus.hitl_reasons, *motivos_aj])
        campo_ajustado_ref = snap_ajustado.gaussian_field_ref
        ajuste_info = aj
        print(f"  → {len(antes['centers']):,} → {len(aj.centers):,} gaussianas "
              f"(×{aj.compresion:.1f}) · error de reconstrucción {aj.rmse_hu:.1f} HU")
        peor = sorted(aj.rmse_hu_por_region.items(), key=lambda kv: -kv[1])[:3]
        if peor:
            print("  → peores regiones: " +
                  " · ".join(f"{'fondo' if c == 0 else c}: {e:.0f} HU" for c, e in peor))
        print(f"  → perfil `{snap_ajustado.perfil_campo}`: DERIVADO, la escala ya no "
              "es el vóxel · viaja como `asset.field_fit` en el `.uos`")
        # El descriptor del campo ajustado se construye AQUÍ para no acoplar el paquete
        # UOS a `gaussian_engine`. El UOS agent recibe un dict plano y lo vuelca tal cual.
        from gaussian_engine import PERFIL, esquema

        descriptor_ajustado = {
            "role": "campo ajustado contra densidad medida",
            "measured": False,
            "note": (
                "elipsoides optimizados contra la densidad del CBCT; la escala es un "
                "ajuste, NO una medida del tejido"
            ),
            "profile": PERFIL,
            "frame": "frame.ct_001",
            "units": "mm",
            "n_primitives": len(aj.centers),
            "columns": [
                {"name": c.nombre, "unit": c.unidad, "scale": c.escala,
                 "measured": c.medido, "derived_from": c.derivado_de,
                 "meaning": c.significado, "vocabulary": c.vocabulario}
                for c in esquema(aj.rmse_hu)
            ],
            "reconstruction_error_hu": aj.rmse_hu,
            "compression": aj.compresion,
            # La submuestrea es la misma que la semilla (el ajuste parte de los
            # mismos vóxeles), pero el número final es el del campo comprimido.
            "submuestreo": _submuestreo_desde(pipe.store, snap_ajustado, len(aj.centers)),
        }

    # ── Entrenamiento de apariencia (gsplat contra fotos) ───────────────────
    # El `--entrena-apariencia` necesita GPU y bloquea el pipeline. Entrena un campo
    # de gaussianas con color REAL del paciente optimizado contra renders de Blender.
    # El resultado viaja como `asset.apariencia` en el `.uos` con perfil
    # 'ash-gs-apariencia/1.0' y regulatory.layer=1, status="derived".
    if args.entrena_apariencia:
        if args.gs_apariencia is None:
            print("  ⚠ --entrena-apariencia requiere --gs-apariencia")
        elif fus.snapshot.surface_ref is None:
            print("  ⚠ --entrena-apariencia requiere surface_ref (malla del mesh-agent)")
        elif not fus.snapshot.image_refs:
            print("  ⚠ --entrena-apariencia requiere image_refs (fotos intraorales)")
        else:
            print("\n--- 3b · ENTRENAMIENTO APARIENCIA (gsplat + Blender) ---")
            try:
                import numpy as np
                import torch
                from gaussian_engine.apariencia import (
                    ITERACIONES,
                    N_VISTAS,
                    RESOLUCION,
                    entrena_apariencia,
                )
                # Cargar la malla del almacén
                mesh_data = pipe.store.load(fus.snapshot.surface_ref)
                posiciones = np.asarray(mesh_data["positions"], dtype=np.float32)
                caras = np.asarray(mesh_data["faces"], dtype=np.int32)

                # Cargar las rutas de fotos (paso el directorio, no los pixeles)
                dir_caso = args.caso.resolve()
                fotos_paths = []
                for ref in fus.snapshot.image_refs:
                    try:
                        foto_data = pipe.store.load(ref)
                        if "path" in foto_data:
                            fotos_paths.append(Path(str(foto_data["path"])))
                    except (KeyError, OSError):
                        pass
                if not fotos_paths:
                    # Fallback: buscar fotos en el directorio del caso
                    for ext in ("*.jpg", "*.jpeg", "*.png", "*.heic"):
                        fotos_paths.extend(sorted(dir_caso.glob(ext)))

                if not fotos_paths:
                    print("  ⚠ No se encontraron fotos intraorales para entrenar")
                else:
                    destino_ap = args.salida / "apariencia"
                    params_ap, metricas = entrena_apariencia(
                        posiciones, caras, fotos_paths,
                        destino=destino_ap,
                        n_vistas=N_VISTAS,
                        resolucion=RESOLUCION,
                        iteraciones=ITERACIONES,
                        dispositivo="cuda" if torch.cuda.is_available() else "cpu",
                        traza=True,
                    )

                    # Guardar el PLY INRIA en el almacén
                    ply_ap = destino_ap / "apariencia.ply"
                    if ply_ap.exists():
                        ref_ap = pipe.store.put(
                            means=params_ap["means"],
                            scales=params_ap["scales"],
                            quats=params_ap["quats"],
                            opacities=params_ap["opacities"],
                            colors=params_ap["colors"],
                        )
                        # Actualizar el snapshot con la referencia
                        fus = replace(fus, snapshot=fus.snapshot.model_copy(
                            update={"apariencia_ref": ref_ap}
                        ))
                        print(f"  → apariencia: PSNR {metricas.psnr_db:.2f} dB, "
                              f"SSIM {metricas.ssim:.3f}, "
                              f"{metricas.n_gaussianas:,} gaussianas")
                        print(f"  → perfil `{metricas.perfil}`: DERIVADO, color real "
                              "· viaja como `asset.apariencia` en el `.uos`")
            except Exception as e:
                print(f"  ⚠ Error entrenando apariencia: {e}")
                import traceback
                traceback.print_exc()

    print("\n--- 4 · EXPORTACIÓN ---")
    # Las etiquetas del escáner viajan al canal del visor: son las que dan las coronas
    # completas y separadas, que es lo que un clínico reconoce. El compuesto del CBCT
    # cubre el 51 % del volumen de cada pieza y de forma desigual.
    fin = pipe.exportar(
        fus, args.salida / "export",
        etiquetas_ios=None if etq_ios is None else etq_ios.astype("int16"),
        sin_originales=args.sin_originales,
        gs_apariencia=args.gs_apariencia,
        # UOS referencia los ficheros ORIGINALES, no los derivados: el .uos lleva el STL
        # y las fotos tal como entraron, con su sha256, para que quien lo reciba pueda
        # verificar que no los tocamos.
        malla=caso.mesh,
        escena_gs=(None if args.gs_apariencia is None
                   else args.gs_apariencia / "gs_escaner-coronas.ply"),
        imagenes=list(caso.images),
        # La serie DICOM sube el .uos a UOS-Vol. Detrás de bandera: son cientos de megas.
        cbct=caso.cbct if args.con_volumen else None,
        # Para el `meta.json` de `derived/`: qué pesos produjeron la segmentación.
        modelo_segmentacion=args.modelo,
        # El campo ajustado va APARTE en el `.uos`: el twin lleva la semilla medida
        # (en `fus.snapshot.gaussian_field_ref`) y el ajustado como `asset.field_fit`.
        campo_ajustado=campo_ajustado_ref,
        ajuste=ajuste_info,
        campo_ajustado_descriptor=descriptor_ajustado,
    )
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
