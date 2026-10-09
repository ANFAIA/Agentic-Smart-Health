#!/usr/bin/env python
"""Inventario de series CBCT sin identificadores, listo para que la clínica lo interprete.

    uv run python scripts/inventario_series_cbct.py \\
        --zip ~/ruta/original.zip --salida data/processed/<caso>/clinic-package

Escribe `inventario.json`, `inventario.md` y `cobertura.svg`. No lee píxeles ni exporta
imágenes de cortes: son anatomía del paciente. Si alguna cabecera tiene un tag
identificable poblado, NO escribe nada salvo que se pase `--permitir-identificables`, y
entonces lo declara en la cabecera del informe. Guardar la salida bajo `data/processed/`.

No envía nada a nadie: el paquete se entrega a mano, cuando el responsable lo decida.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RESUMEN_EN = "Builds a de-identified CBCT series inventory and coverage figure for clinic review."
RAIZ = Path(__file__).resolve().parent.parent
for src in sorted(RAIZ.glob("packages/*/src")):
    sys.path.insert(0, str(src))

from uos.inventario import (  # noqa: E402
    FACTOR_HUECO,
    inventaria,
    lee_cabeceras_de_directorio,
    lee_cabeceras_de_zip,
)

# Un hueco se lista en el informe legible a partir de este salto; el JSON los lleva todos.
# Con paso de 0,15 mm, 1 mm son unos seis cortes ausentes: por debajo, la tabla tendría
# más de cien filas y escondería las pocas que importan.
HUECO_RELEVANTE_MM = 1.0

PREGUNTAS = (
    "¿Qué representa cada serie: la misma exploración reconstruida dos veces, dos "
    "adquisiciones, o una selección de cortes de una reconstrucción mayor?",
    "Los números de instancia ausentes coinciden con los huecos de posición. ¿Los cortes "
    "se omitieron deliberadamente en la exportación? ¿Existe la reconstrucción completa y "
    "puede exportarse?",
    "¿Qué cobertura anatómica tiene cada serie (maxilar, mandíbula, campo completo)?",
    "¿Corresponden este CBCT, el escaneo intraoral, las fotografías y los informes a la "
    "misma visita? ¿Con qué fechas? Las cabeceras de esta exportación no traen fechas.",
    "¿Con qué equipo, protocolo y controles de calidad se adquirió? ¿Hay registro de "
    "calibración geométrica o de intensidad?",
    "¿Para qué tareas considera un especialista adecuada cada serie, revisada en su visor "
    "habitual?",
)


# ── Figura de cobertura ─────────────────────────────────────────────────────────
#
# Un panel por serie y un eje por panel. ⚠️ Las series pueden tener marcos de referencia
# distintos, y entonces sus z NO son comparables: un eje común sugeriría una alineación
# que nadie ha acreditado. Los tramos con cortes son barras; los huecos, línea base vacía.
_ANCHO, _MARGEN_IZQ, _MARGEN_DER, _ALTO_PANEL = 760, 24, 24, 92
_COLOR_SERIE, _TEXTO, _TEXTO_2, _EJE, _FONDO = (
    "#2a78d6", "#0b0b0b", "#52514e", "#b9b8b2", "#fcfcfb")


def _tramos(posiciones: list[float], paso: float) -> list[tuple[float, float]]:
    """Tramos contiguos de cortes: se cortan donde el salto supera `FACTOR_HUECO` pasos."""
    tramos, inicio = [], posiciones[0]
    for a, b in zip(posiciones, posiciones[1:], strict=False):
        if b - a > FACTOR_HUECO * paso:
            tramos.append((inicio, a))
            inicio = b
    tramos.append((inicio, posiciones[-1]))
    return tramos


def _marcas(z0: float, z1: float) -> list[float]:
    for paso in (1, 2, 5, 10, 20, 25, 50):
        if (z1 - z0) / paso <= 8:
            break
    primero = int(-(-z0 // paso)) * paso
    return [float(v) for v in range(primero, int(z1) + 1, paso)]


def figura_cobertura(inventario: dict, posiciones: dict[str, list[float]]) -> str:
    paneles = [s for s in inventario["series"] if posiciones.get(s["series_key"])]
    alto = 40 + _ALTO_PANEL * len(paneles) + 34
    util = _ANCHO - _MARGEN_IZQ - _MARGEN_DER
    partes = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_ANCHO} {alto}" '
        f'width="{_ANCHO}" height="{alto}" font-family="system-ui, sans-serif" '
        f'role="img" aria-labelledby="t d">',
        '<title id="t">Cobertura en z de cada serie CBCT</title>',
        '<desc id="d">Tramos con cortes como barras y huecos como línea base vacía. '
        'Cada serie tiene su propio eje.</desc>',
        f'<rect width="100%" height="100%" fill="{_FONDO}"/>',
        f'<text x="{_MARGEN_IZQ}" y="24" font-size="15" font-weight="600" '
        f'fill="{_TEXTO}">Cortes presentes por posición z (mm)</text>',
    ]
    for i, serie in enumerate(paneles):
        z = sorted(posiciones[serie["series_key"]])
        pos = serie["positions"]
        z0, z1 = z[0], z[-1]
        escala = util / (z1 - z0) if z1 > z0 else 1.0

        def x(v: float, z0: float = z0, escala: float = escala) -> float:
            return _MARGEN_IZQ + (v - z0) * escala

        y = 40 + i * _ALTO_PANEL
        base = y + 52
        n = serie["instance_numbers"]
        partes.append(
            f'<text x="{_MARGEN_IZQ}" y="{y + 18}" font-size="13" font-weight="600" '
            f'fill="{_TEXTO}">Serie {serie["series_key"]}</text>'
            f'<text x="{_MARGEN_IZQ + 150}" y="{y + 18}" font-size="12" fill="{_TEXTO_2}">'
            f'{serie["instances"]} cortes · instancias {n.get("min")}–{n.get("max")} · '
            f'{len(pos["gaps"])} huecos · salto máximo {pos["max_step_mm"]:.2f} mm</text>')
        partes.append(f'<line x1="{_MARGEN_IZQ}" x2="{_ANCHO - _MARGEN_DER}" y1="{base}" '
                      f'y2="{base}" stroke="{_EJE}" stroke-width="1"/>')
        for a, b in _tramos(z, pos["median_step_mm"]):
            ancho = max((b - a) * escala, 2.0)
            partes.append(
                f'<rect x="{x(a):.2f}" y="{base - 20}" width="{ancho:.2f}" height="20" '
                f'rx="2" fill="{_COLOR_SERIE}"><title>{a:.2f} a {b:.2f} mm</title></rect>')
        for marca in _marcas(z0, z1):
            partes.append(
                f'<line x1="{x(marca):.2f}" x2="{x(marca):.2f}" y1="{base}" y2="{base + 4}" '
                f'stroke="{_EJE}"/><text x="{x(marca):.2f}" y="{base + 17}" font-size="11" '
                f'text-anchor="middle" fill="{_TEXTO_2}">{marca:g}</text>')
    nota = ("Las series tienen marcos de referencia distintos: sus ejes no son comparables."
            if inventario["identity_counts"]["FrameOfReferenceUID"] > 1
            else "Eje propio por serie.")
    partes.append(f'<text x="{_MARGEN_IZQ}" y="{alto - 12}" font-size="11" '
                  f'fill="{_TEXTO_2}">{nota}</text></svg>')
    return "\n".join(partes) + "\n"


# ── Informe legible ─────────────────────────────────────────────────────────────

def _dec(serie: dict, tag: str) -> str:
    valores = [v for v in serie["declarations"].get(tag, []) if v]
    return ", ".join(valores) if valores else "no declarado"


def informe_md(inventario: dict) -> str:
    lineas: list[str] = ["# Inventario de las series CBCT", ""]
    if inventario["identifying_tags_present"]:
        lineas += [f"> ⚠️ **Las cabeceras contienen datos identificativos** en los tags "
              f"{', '.join(inventario['identifying_tags_present'])}. Este informe no los "
              "reproduce, pero el paquete no debe salir de la clínica sin revisión.", ""]
    c = inventario["identity_counts"]
    lineas += [
        f"La exportación contiene **{inventario['instances']} instancias** en "
        f"{c['SeriesInstanceUID']} series, {c['StudyInstanceUID']} estudio(s) y "
        f"{c['FrameOfReferenceUID']} marco(s) de referencia. Los identificadores se "
        "muestran como un resumen corto, no reversible, que solo sirve para referirse a "
        "cada serie en esta conversación.",
        "",
        "Este informe no contiene píxeles, nombres ni identificadores DICOM. Describe lo que "
        "declaran las cabeceras; no acredita calibración ni idoneidad clínica.",
        "",
        "![Cobertura en z](cobertura.svg)",
        "",
        "## Resumen por serie",
        "",
        "| Serie | Cortes | Instancias | Números ausentes | Cobertura z | Paso mediano "
        "| Huecos | Salto máximo |",
        "|---|---:|---|---:|---|---:|---:|---:|",
    ]
    for s in inventario["series"]:
        n, p = s["instance_numbers"], s["positions"] or {}
        lineas.append(
            f"| {s['series_key']} | {s['instances']} | {n.get('min')}–{n.get('max')} "
            f"| {n.get('missing_count')} "
            f"| {p.get('z_min_mm', float('nan')):.2f} a {p.get('z_max_mm', float('nan')):.2f}"
            f" mm ({p.get('extent_mm', float('nan')):.2f}) "
            f"| {p.get('median_step_mm', float('nan')):.3f} mm | {len(p.get('gaps', []))} "
            f"| {p.get('max_step_mm', float('nan')):.2f} mm |")
    lineas += ["", "## Declaraciones de adquisición", "",
          "| Serie | Equipo | Tipo de imagen | kV | Matriz | Píxel (mm) | Grosor (mm) "
          "| Rescale |", "|---|---|---|---|---|---|---|---|"]
    for s in inventario["series"]:
        lineas.append(
            f"| {s['series_key']} | {_dec(s, 'Manufacturer')} {_dec(s, 'ManufacturerModelName')}"
            f" | {_dec(s, 'ImageType')} | {_dec(s, 'KVP')} | {_dec(s, 'Rows')}×"
            f"{_dec(s, 'Columns')} | {_dec(s, 'PixelSpacing')} | {_dec(s, 'SliceThickness')} "
            f"| pendiente {_dec(s, 'RescaleSlope')}, intercepto {_dec(s, 'RescaleIntercept')}, "
            f"tipo {_dec(s, 'RescaleType')} |")
    lineas += ["", "`RescaleType` y el espaciado son declaraciones del proveedor; no "
               "sustituyen una referencia independiente de calibración.", ""]
    lineas += ["## Huecos de al menos 1 mm", ""]
    for s in inventario["series"]:
        p = s["positions"] or {}
        grandes = [g for g in p.get("gaps", []) if g["step_mm"] >= HUECO_RELEVANTE_MM]
        lineas += [f"**Serie {s['series_key']}.** {len(p.get('gaps', []))} huecos en total; "
              f"{len(grandes)} de al menos 1 mm. Orden por instancia coherente con la "
              f"posición: {'sí' if p.get('instance_order_matches_position') else 'no'}.", ""]
        if grandes:
            lineas += ["| Desde (mm) | Hasta (mm) | Salto (mm) | Cortes ausentes estimados |",
                  "|---:|---:|---:|---:|"]
            lineas += [f"| {g['from_mm']:.2f} | {g['to_mm']:.2f} | {g['step_mm']:.2f} "
                  f"| {g['estimated_missing']} |" for g in grandes]
            lineas.append("")
    fechas = {t: sorted({f for s in inventario["series"] for f in s["dates"][t]})
              for t in inventario["series"][0]["dates"]}
    lineas += ["## Fechas", ""]
    if any(fechas.values()):
        lineas += [f"- {t}: {', '.join(v) or 'no declarada'}" for t, v in fechas.items()]
        resolucion = "día" if inventario["dates_resolution"] == "day" else "mes"
        lineas += ["", f"Resolución: {resolucion}."]
    else:
        lineas.append("Las cabeceras no declaran fechas de estudio, serie, adquisición "
                      "ni contenido.")
    lineas += ["", "## Preguntas para la clínica", ""]
    lineas += [f"{i}. {q}" for i, q in enumerate(PREGUNTAS, 1)]
    lineas += ["", "## Alcance", "", inventario["scope"], ""]
    return "\n".join(lineas)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    origen = ap.add_mutually_exclusive_group(required=True)
    origen.add_argument("--zip", type=Path, help="ZIP original; se lee entrada a entrada.")
    origen.add_argument("--serie", type=Path, action="append",
                        help="Directorio de una serie. Repetible.")
    ap.add_argument("--salida", type=Path, required=True)
    ap.add_argument("--fechas-completas", action="store_true",
                    help="Fechas al día en vez de al mes.")
    ap.add_argument("--permitir-identificables", action="store_true",
                    help="Escribir aunque haya tags identificables (se declaran por nombre).")
    args = ap.parse_args()

    cabeceras = (lee_cabeceras_de_zip(args.zip) if args.zip
                 else [c for d in args.serie for c in lee_cabeceras_de_directorio(d)])
    inventario = inventaria(cabeceras, fechas_completas=args.fechas_completas)
    if inventario["identifying_tags_present"] and not args.permitir_identificables:
        print("Tags identificables poblados: "
              f"{', '.join(inventario['identifying_tags_present'])}. No se escribe nada; "
              "usar --permitir-identificables para un informe interno.", file=sys.stderr)
        return 2
    from uos.inventario import clave

    posiciones: dict[str, list[float]] = {}
    for ds in cabeceras:
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None:
            posiciones.setdefault(clave(getattr(ds, "SeriesInstanceUID", "")) or "",
                                  []).append(float(ipp[2]))

    args.salida.mkdir(parents=True, exist_ok=True)
    (args.salida / "inventario.json").write_text(
        json.dumps(inventario, indent=2, ensure_ascii=False) + "\n")
    (args.salida / "inventario.md").write_text(informe_md(inventario))
    (args.salida / "cobertura.svg").write_text(figura_cobertura(inventario, posiciones))
    for s in inventario["series"]:
        p = s["positions"] or {}
        print(f"serie {s['series_key']}: {s['instances']} cortes, "
              f"{s['instance_numbers'].get('missing_count')} números ausentes, "
              f"{len(p.get('gaps', []))} huecos")
    print(f"Paquete: {args.salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
