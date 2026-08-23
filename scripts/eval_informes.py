#!/usr/bin/env python
"""eval_informes.py — ¿Cuánto de lo que dice un informe acaba en el contrato?

    uv run python scripts/eval_informes.py                    # backend determinista
    uv run python scripts/eval_informes.py --backend ollama   # modelo local, gratis
    uv run python scripts/eval_informes.py --backend llm      # requiere ANTHROPIC_API_KEY
    uv run python scripts/eval_informes.py --detalle          # caso a caso

**Por qué existe.** El `README.md` fija como métrica de éxito «fiabilidad de los agentes
de ingesta > 95%», y hasta ahora no había forma de calcularla para el informe: los tests
comprueban patrones, no fracciones. Y el docstring del `report-agent` afirma que el
backend determinista «da el suelo medible contra el que comparar al LLM» — el suelo
existía, la medida no. Esto la produce, sobre el corpus anotado de `report_corpus.py`.

**Cómo leerlo.** Los cuatro contadores no son intercambiables:

- `faltan` — el informe lo dice y no llega al twin. Un hueco, visible.
- `sobran` / `errores` — llega al twin algo que el informe no dice, o lo dice distinto.
  Es el fallo del ADR 003: plausible, silencioso, y ya dentro del contrato.

Por eso no hay un porcentaje único. Un extractor con 90% de cobertura y cero invenciones
es mejor que uno con 95% y tres, y una media los ordenaría al revés.

**Sobre el backend `llm`.** Cubre los mismos campos que el determinista —pH, anatomía
radicular y hallazgos—, así que las dos columnas son comparables valor a valor. No
siempre fue así: hasta que se amplió el esquema de la tool, el LLM solo producía pH y los
campos clínicos salían del regex incluso con `backend="llm"`, lo que dejaba al modelo
enchufado al campo tabulado y ausente del que viene en prosa. Esa asimetría la descubrió
este script.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "packages/ingestion-agents/src"))

from ingestion_agents import report_corpus  # noqa: E402
from ingestion_agents.report_agent import (  # noqa: E402
    extract_hallazgos_by_rules,
    extract_medidas_by_rules,
    extract_ph_by_rules,
)
from ingestion_agents.report_corpus import (  # noqa: E402
    CASES,
    Puntuacion,
    ReportCase,
    normaliza,
    puntua,
)

FAMILIAS = (report_corpus.TABULADO, report_corpus.PROSA, report_corpus.ABSTENCION)
CERO = Puntuacion(0, 0, 0, 0)


def puntua_medidas(caso: ReportCase) -> Puntuacion:
    """Los índices se puntúan por recuento: el corpus anota cuántos declara el informe.

    No se comparan valor a valor porque una `Medida` **no está interpretada** —el
    contrato la guarda tal cual—, así que lo único que se puede exigir es que no se
    pierda ninguna ni aparezca una de más.
    """
    extraidas = len(extract_medidas_by_rules(caso.text))
    return Puntuacion(
        aciertos=min(extraidas, caso.medidas),
        errores=0,
        faltan=max(0, caso.medidas - extraidas),
        sobran=max(0, extraidas - caso.medidas),
    )


def extrae(caso: ReportCase, backend: str) -> tuple[dict, dict]:
    """`(pH, clínicos)` de un informe con el backend pedido."""
    if backend == "rules":
        return extract_ph_by_rules(caso.text).findings, extract_hallazgos_by_rules(caso.text)

    from ingestion_agents.report_agent import extract_by_llm, extract_by_local_llm

    extraccion = (
        extract_by_llm(caso.text) if backend == "llm" else extract_by_local_llm(caso.text)
    )
    return (
        {code: valor for code, (valor, _) in extraccion.findings.items()},
        extraccion.clinicos,
    )


def _fila(nombre: str, ph: Puntuacion, clinicos: Puntuacion, ancho: int = 38) -> str:
    total = ph + clinicos
    marca = "  ← inventa" if (total.errores or total.sobran) else ""
    return (
        f"{nombre:<{ancho}} {total.aciertos:>3} {total.errores:>4} "
        f"{total.faltan:>4} {total.sobran:>4}   {total.cobertura:>6.1%}{marca}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--backend", choices=("rules", "llm", "ollama"), default="rules")
    ap.add_argument("--detalle", action="store_true", help="Una línea por caso.")
    ap.add_argument(
        "--volcar", type=Path, default=None, help="Escribe el corpus como .txt en este directorio."
    )
    args = ap.parse_args()

    if args.volcar:
        rutas = report_corpus.write_all(args.volcar)
        print(f"{len(rutas)} informes escritos en {args.volcar}")
        return 0

    if args.backend == "llm" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Falta ANTHROPIC_API_KEY para el backend `llm`.", file=sys.stderr)
        return 2

    valores_ph, valores_clinicos = report_corpus.total_verdad()
    valores_medidas = sum(caso.medidas for caso in CASES)
    print(f"\nCorpus: {len(CASES)} informes · {valores_ph} valores de pH · "
          f"{valores_clinicos} clínicos · {valores_medidas} índices · "
          f"backend `{args.backend}`")

    cabecera = f"\n{'':<38} {'ok':>3} {'err':>4} {'falt':>4} {'sobr':>4}   {'cobert':>6}"
    total_ph = total_clinicos = CERO

    for familia in FAMILIAS:
        casos = report_corpus.by_familia(familia)
        acum_ph = acum_clinicos = CERO
        detalle: list[str] = []
        for caso in casos:
            ph_extraido, clinicos_extraidos = extrae(caso, args.backend)
            p_ph = puntua(ph_extraido, caso.ph)
            p_cl = puntua(normaliza(clinicos_extraidos), normaliza(caso.clinicos))
            p_cl = p_cl + puntua_medidas(caso)
            acum_ph, acum_clinicos = acum_ph + p_ph, acum_clinicos + p_cl
            if args.detalle:
                detalle.append("  " + _fila(caso.name, p_ph, p_cl, ancho=36))
        if not casos:
            continue
        print(cabecera if familia == FAMILIAS[0] else "")
        print(f"{familia.upper()} ({len(casos)} informes)")
        print("\n".join(detalle) if detalle else "", end="\n" if detalle else "")
        print("  " + _fila("— subtotal", acum_ph, acum_clinicos, ancho=36))
        total_ph, total_clinicos = total_ph + acum_ph, total_clinicos + acum_clinicos

    print("\n" + "=" * 76)
    print(_fila("TOTAL", total_ph, total_clinicos))
    print(f"{'  de los cuales pH':<38} {total_ph.aciertos:>3} {total_ph.errores:>4} "
          f"{total_ph.faltan:>4} {total_ph.sobran:>4}   {total_ph.cobertura:>6.1%}")
    print(f"{'  de los cuales clínicos':<38} {total_clinicos.aciertos:>3} "
          f"{total_clinicos.errores:>4} {total_clinicos.faltan:>4} "
          f"{total_clinicos.sobran:>4}   {total_clinicos.cobertura:>6.1%}")

    falsos = total_ph.errores + total_ph.sobran + total_clinicos.errores + total_clinicos.sobran
    print(f"\nValores falsos dentro del contrato: {falsos}")
    print("Objetivo del README (fiabilidad de ingesta): > 95%")

    limitados = [caso for caso in CASES if caso.limite]
    if limitados:
        print(f"\nDeuda documentada — {len(limitados)} informes con límite declarado:")
        for caso in limitados:
            texto = textwrap.fill(caso.limite, width=74, subsequent_indent="    ")
            print(f"  · {caso.name}:\n    {texto}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
