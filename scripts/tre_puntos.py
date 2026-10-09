#!/usr/bin/env python
"""TRE de un registro CBCT↔escáner con puntos anatómicos marcados por especialistas.

    uv run python scripts/tre_puntos.py --uos caso.uos \\
        --cbct-puntos obs1-cbct.mrk.json --cbct-puntos obs2-cbct.mrk.json \\
        --ios-puntos obs1-ios.mrk.json --ios-puntos obs2-ios.mrk.json \\
        --stl-cargado-como lps --salida data/processed/<caso>/tre

Mide el registro que declara el `.uos` (de las coordenadas DICOM al frame del escáner) con
puntos que ese registro no usó. Con `--ajuste` mide además un registro rígido por puntos.
El protocolo de marcado está en `docs/spec/uos-protocolo-puntos-anatomicos.md`.

No escribe en el `.uos`: el resultado se guarda aparte, bajo `data/processed/`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RESUMEN_EN = "Measures CBCT-to-scanner target registration error from expert landmarks."
RAIZ = Path(__file__).resolve().parent.parent
for src in sorted(RAIZ.glob("packages/*/src")):
    sys.path.insert(0, str(src))

from fusion_agents.tre import (  # noqa: E402
    a_coordenadas_stl,
    consenso,
    lee_puntos,
    tre_de_registro,
    tre_por_puntos,
)
from uos import read_manifest  # noqa: E402


def registro_del_volumen(manifiesto, ident: str | None) -> tuple[str, np.ndarray, str, str]:
    """El registro que lleva de las coordenadas DICOM del volumen al frame canónico.

    ⚠️ Desde el exportador 0.18.0 hay dos: uno desde el volumen DICOM y otro desde el campo
    gaussiano centrado. Los puntos se marcan sobre el DICOM, así que solo vale el primero.
    """
    volumen = next((a for a in manifiesto.assets if a.kind.value == "volume"), None)
    if volumen is None and ident is None:
        raise SystemExit("El .uos no declara volumen; indica --registro")
    for r in manifiesto.registrations:
        if (r.id == ident) if ident else (r.source_frame == volumen.frame):
            return r.id, np.asarray(r.transform_4x4_row_major), r.source_frame, r.target_frame
    raise SystemExit("No hay registro desde el frame del volumen")


def _tabla(resumen: dict) -> list[str]:
    if not resumen.get("n"):
        return ["Sin puntos.", ""]
    filas = ["| Región | Puntos | RMS (mm) | Máximo (mm) |", "|---|---:|---:|---:|"]
    filas += [f"| {r} | {v['n']} | {v['rms_mm']:.2f} | {v['max_mm']:.2f} |"
              for r, v in resumen["per_region"].items()]
    filas += [f"| **Total** | {resumen['n']} | {resumen['rms_mm']:.2f} "
              f"| {resumen['max_mm']:.2f} |", ""]
    return filas


def informe_md(resultado: dict) -> str:
    obs = resultado["observers"]
    lineas = [
        "# TRE del registro CBCT↔escáner", "",
        f"TRE sobre puntos marcados por {obs['cbct']['observers']} observador(es) en el CBCT "
        f"y {obs['ios']['observers']} en el escáner, en este caso. No es exactitud general "
        "del sistema ni acredita calibración.", "",
        f"Registro evaluado: `{resultado['registration']['id']}`, de "
        f"`{resultado['registration']['source_frame']}` a "
        f"`{resultado['registration']['target_frame']}`. STL cargado como "
        f"{resultado['stl_loaded_as']}.", "",
        "## Registro existente", "",
        "Todos los puntos son de evaluación: el registro no los usó.", "",
        *_tabla(resultado["existing_registration"]),
    ]
    sin = resultado["existing_registration"]["unpaired"]
    if sin["only_cbct"] or sin["only_ios"]:
        lineas += [f"Sin pareja: solo CBCT {sin['only_cbct']}, solo escáner {sin['only_ios']}.",
                   ""]
    if "landmark_fit" in resultado:
        fit = resultado["landmark_fit"]
        lineas += ["## Registro por puntos", "",
                   f"Ajustado con {len(fit['fit_labels'])} puntos. Error en puntos reservados:",
                   "", *_tabla(fit["evaluation"]),
                   "*Leave-one-out* sobre todos los puntos:", "", *_tabla(fit["leave_one_out"]),
                   "Residuo en los puntos de ajuste: "
                   f"{fit['fiducial_residual'].get('rms_mm', 0):.2f} mm RMS. Es un residuo, "
                   "no un error de registro.", ""]
    lineas += ["## Acuerdo entre observadores", ""]
    for lado, nombre in (("cbct", "CBCT"), ("ios", "Escáner")):
        d = obs[lado]
        if d["observers"] > 1 and d.get("max_pairwise_mm") is not None:
            lineas.append(f"- {nombre}: distancia máxima entre observadores "
                          f"{d['max_pairwise_mm']:.2f} mm; media por punto "
                          f"{d['mean_max_pairwise_mm']:.2f} mm.")
        else:
            lineas.append(f"- {nombre}: un observador; no se puede medir el acuerdo.")
        if d["not_marked_by_all"]:
            lineas.append(f"  Excluidos por no marcarlos todos: {d['not_marked_by_all']}.")
    return "\n".join(lineas) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uos", type=Path, required=True)
    ap.add_argument("--cbct-puntos", type=Path, action="append", required=True,
                    help="Puntos sobre el DICOM, uno por observador.")
    ap.add_argument("--ios-puntos", type=Path, action="append", required=True,
                    help="Puntos sobre el STL del escáner, uno por observador.")
    ap.add_argument("--stl-cargado-como", choices=("lps", "ras"), required=True,
                    help="Cómo se cargó el STL en el visor de marcado. No se adivina.")
    ap.add_argument("--sistema-csv", choices=("lps", "ras"),
                    help="Sistema de los ficheros CSV, que no lo declaran.")
    ap.add_argument("--ajuste", default="",
                    help="Etiquetas separadas por comas para un registro por puntos.")
    ap.add_argument("--registro", help="Id del registro; por defecto, el del volumen.")
    ap.add_argument("--salida", type=Path, required=True)
    args = ap.parse_args()

    manifiesto = read_manifest(args.uos)
    ident, matriz, origen, destino = registro_del_volumen(manifiesto, args.registro)
    cbct, disp_cbct = consenso([lee_puntos(p, args.sistema_csv) for p in args.cbct_puntos])
    ios_lps, disp_ios = consenso([lee_puntos(p, args.sistema_csv) for p in args.ios_puntos])
    ios = a_coordenadas_stl(ios_lps, args.stl_cargado_como)

    resultado: dict = {
        "schema": "histora-tre/1.0",
        "registration": {"id": ident, "source_frame": origen, "target_frame": destino},
        "stl_loaded_as": args.stl_cargado_como.upper(),
        "observers": {"cbct": disp_cbct, "ios": disp_ios},
        "existing_registration": tre_de_registro(cbct, ios, matriz),
        "scope": ("TRE on expert landmarks for this case only; not general system accuracy, "
                  "not physical calibration. Not written into the .uos."),
    }
    ajuste = [x.strip() for x in args.ajuste.split(",") if x.strip()]
    if ajuste:
        resultado["landmark_fit"] = tre_por_puntos(cbct, ios, ajuste)

    args.salida.mkdir(parents=True, exist_ok=True)
    (args.salida / "tre.json").write_text(json.dumps(resultado, indent=2) + "\n")
    (args.salida / "tre.md").write_text(informe_md(resultado))
    r = resultado["existing_registration"]
    if r["n"]:
        print(f"{ident}: TRE RMS {r['rms_mm']:.2f} mm, máximo {r['max_mm']:.2f} mm, "
              f"{r['n']} puntos")
    else:
        print(f"{ident}: ningún punto emparejado")
    print(f"Informe: {args.salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
