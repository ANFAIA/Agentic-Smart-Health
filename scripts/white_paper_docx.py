#!/usr/bin/env python
"""white_paper_docx.py — El white paper en `.docx`, para que un revisor lo edite.

    uv run python scripts/white_paper_docx.py

**Por qué existe.** El `.tex` es la fuente de verdad y lo seguirá siendo: el `.docx` se
genera para entregárselo a quien tiene que corregir prosa en la herramienta que ya usa, y
sus cambios vuelven como cambios al `.tex`. El `.docx` está en `.gitignore` justo por eso —
versionarlo crearía un segundo documento que se separa del primero en cuanto alguien edite
uno y no el otro, sin forma de saber cuál es el vigente.

**Por qué no es `pandoc` a secas.** El documento usa tres construcciones que el lector de
LaTeX de pandoc no entiende, y cada una falla distinto:

- El entorno `finding`, que son 13 cuadros con título. `pandoc` no expande entornos
  definidos con `\\newenvironment`, así que **se comería el título** y dejaría el cuerpo
  suelto, indistinguible del párrafo anterior. Aquí se traducen a una cita con el título en
  negrita, que en Word es el estilo «Quote» y se sigue leyendo como un cuadro aparte.
- Los tipos de columna `L{3cm}` y `C{0.6cm}`, definidos con `\\newcolumntype`. pandoc no
  los reconoce y **pierde la tabla entera**, que son 12. Se reescriben a `p{3cm}`, que es
  LaTeX estándar y conserva los anchos relativos.
- El bloque de portada centrado, que saldría como párrafos sueltos. El título y el autor se
  pasan como metadatos para que Word les ponga sus estilos y el panel de navegación
  funcione.

Lo que **no** se toca: las 15 fórmulas del apéndice, que pandoc convierte a ecuaciones
nativas de Word y por tanto editables, y la bibliografía.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
FUENTE = RAIZ / "docs" / "spec" / "uos-white-paper.tex"
DESTINO = RAIZ / "docs" / "spec" / "uos-white-paper.docx"

RESUMEN_EN = "Converts the white paper's LaTeX source to .docx for reviewers to edit."


def _findings(tex: str) -> tuple[str, int]:
    """`\\begin{finding}{Título}` → cita con el título en negrita.

    El título es el argumento del entorno, y es lo que pandoc descartaría. Se saca a un
    párrafo en negrita dentro de un `quote` para que siga habiendo un cuadro con nombre.
    """
    abre = re.compile(r"\\begin\{finding\}\{(.+?)\}\n", re.S)
    tex, n = abre.subn(lambda m: "\\begin{quote}\n\\textbf{" + m.group(1) + "}\n\n", tex)
    tex = tex.replace("\\end{finding}", "\\end{quote}")
    return tex, n


def _columnas(tex: str) -> tuple[str, int]:
    """`L{3.0cm}` y `C{0.6cm}` → `p{3.0cm}`, que es LaTeX estándar."""
    return re.subn(r"\b[LC]\{([0-9.]+cm)\}", r"p{\1}", tex)


def _portada(tex: str) -> tuple[str, str, str, str]:
    """Saca título y autor del bloque centrado, y RETIRA el bloque del cuerpo.

    Si no se retira, el documento sale con el título dos veces: una con el estilo `Title`
    de Word, puesto desde los metadatos, y otra como párrafos sueltos justo debajo.
    Se acota al `center` que contiene el `\\LARGE`, porque hay otros `center` en el
    documento que son tablas y esos se quedan.
    """
    partes = re.findall(r"\{\\LARGE ([^}]+)\}", tex)
    titulo = re.sub(r"\s+", " ", partes[0] if partes else "").strip().rstrip(":")
    # El subtitulo va en `\large`, igual que el autor: se distinguen en que el del autor
    # lleva `\textsuperscript` con las marcas de afiliacion y el subtitulo no.
    grandes = re.findall(r"\{\\large ([^}]+)\}", tex)
    subtitulo = next((re.sub(r"\s+", " ", g).strip()
                      for g in grandes if "textsuperscript" not in g), "")
    autor = re.search(r"\{\\large ([^\\}]+)\\textsuperscript", tex)
    # La afiliacion es prosa con `\textsuperscript`, que en un metadato no pinta nada.
    afiliacion = re.search(r"In collaboration with[^\\]*\\textsuperscript\{a\}(\w+)"
                           r".*?\\textsuperscript\{b\}(\w+)", tex, re.S)
    quien = (autor.group(1).strip() if autor else "")
    if afiliacion:
        quien += f" (in collaboration with {afiliacion.group(1)} and {afiliacion.group(2)})"
    patron = r"\\begin\{center\}(?:(?!\\end\{center\}).)*?\\LARGE.*?\\end\{center\}"
    bloque = re.search(patron, tex, re.S)
    if bloque:
        tex = tex[: bloque.start()] + tex[bloque.end() :]
    return tex, titulo, quien, subtitulo


def _reglas(tex: str) -> tuple[str, int]:
    """`\\hrule height 0.8pt` deja «height 0.8pt» de texto suelto en el documento.

    pandoc se come el `\\hrule` y no el resto de la línea, así que salen dos párrafos con
    una medida tipográfica dentro de un paper. Las reglas son decoración de la maqueta de
    LaTeX y en Word no pintan nada.
    """
    return re.subn(r"\\hrule[^\n]*\n", "", tex)


def _citas(tex: str) -> tuple[str, int]:
    """`\\cite{clave}` → `[n]`, numerado como lo numera LaTeX.

    ⚠️ pandoc **descarta** un `\\cite` cuando no se le da un procesador de bibliografía, y
    lo hace en silencio: deja la frase con un espacio antes del punto y sin referencia. Son
    18 en este documento, y un revisor no tiene forma de saber que faltan. La bibliografía
    es manual (`thebibliography`), así que el número es la posición del `\\bibitem`, que es
    exactamente lo que LaTeX imprime.

    Los propios `\\bibitem` también pierden su número, así que se les antepone.
    """
    claves = re.findall(r"\\bibitem\{([^}]+)\}", tex)
    orden = {clave: i for i, clave in enumerate(claves, start=1)}

    def marca(m: re.Match[str]) -> str:
        n = orden.get(m.group(1))
        return f"[{n}]" if n else m.group(0)

    tex, n = re.subn(r"\\cite\{([^},]+)\}", marca, tex)
    # `\begin{thebibliography}{9}` lleva la etiqueta mas ancha como argumento, y pandoc la
    # deja caer como un «9» suelto justo antes de las referencias.
    tex = re.sub(r"(\\begin\{thebibliography\})\{[^}]*\}", r"\1", tex)
    for clave, i in orden.items():
        tex = tex.replace(f"\\bibitem{{{clave}}}\n", f"\\bibitem{{{clave}}}\n[{i}] ")
    return tex, n


def prepara(tex: str) -> tuple[str, str, str, str, dict[str, int]]:
    """El `.tex` en una forma que pandoc lee sin perder nada, y sus metadatos."""
    tex, titulo, autor, subtitulo = _portada(tex)
    tex, n_regla = _reglas(tex)
    tex, n_find = _findings(tex)
    tex, n_col = _columnas(tex)
    tex, n_cita = _citas(tex)
    # `\code` es un `\newcommand` simple y pandoc lo expande, pero a `\texttt{\small ...}`:
    # el `\small` dentro de un `texttt` le hace emitir un span vacio en algun caso. Se
    # normaliza a `\texttt` a secas, que es lo que se quiere en Word.
    tex, n_code = re.subn(r"\\code\{", r"\\texttt{", tex)
    return tex, titulo, autor, subtitulo, {
        "finding": n_find, "columnas": n_col, "code": n_code, "reglas": n_regla,
        "citas": n_cita,
    }


def convierte(fuente: Path, destino: Path, *, verboso: bool = True) -> int:
    if shutil.which("pandoc") is None:
        print("falta `pandoc`. Instalalo con: sudo apt install -y pandoc", file=sys.stderr)
        return 2
    tex, titulo, autor, subtitulo, cuenta = prepara(fuente.read_text(encoding="utf-8"))
    intermedio = destino.with_suffix(".pandoc.tex")
    intermedio.write_text(tex, encoding="utf-8")
    orden = [
        "pandoc", str(intermedio), "-f", "latex", "-o", str(destino),
        # ⚠️ Sin `--toc`. pandoc inserta un CAMPO de Word que Word rellena al abrir y que
        # **Google Docs deja vacío**, y el destino de este fichero es Drive. Los estilos
        # `Heading 1/2` sí dan esquema vivo en los dos, que es lo que un revisor usa para
        # navegar; un índice estático además se queda obsoleto en cuanto alguien edita.
        "--metadata", f"title={titulo}",
        "--metadata", f"subtitle={subtitulo}",
        "--metadata", f"author={autor}",
        # La fecha es la del `.tex` (`\today` al compilar), asi que aqui es la de HOY: el
        # `.docx` es una entrega fechada, y un revisor necesita saber de que dia es la copia
        # que tiene delante.
        "--metadata", f"date={date.today().isoformat()}",
    ]
    resultado = subprocess.run(orden, capture_output=True, text=True, check=False)
    intermedio.unlink(missing_ok=True)
    if resultado.returncode != 0:
        print(resultado.stderr.strip(), file=sys.stderr)
        return resultado.returncode
    if verboso:
        avisos = [x for x in resultado.stderr.splitlines() if x.strip()]
        print(f"✓ {destino.relative_to(RAIZ)}  ({destino.stat().st_size // 1024} KiB)")
        print(f"  titulo: {titulo!r}")
        print(f"  autor:  {autor!r}")
        print(f"  subtitulo: {subtitulo!r}")
        print(f"  reescrito: {cuenta['finding']} cuadros `finding`, "
              f"{cuenta['columnas']} columnas, {cuenta['code']} `\\code`, "
              f"{cuenta['reglas']} reglas horizontales retiradas, "
              f"{cuenta['citas']} citas numeradas")
        # Los avisos de pandoc SE IMPRIMEN. Un «Could not convert» silenciado es
        # exactamente como se pierde un cuadro o una tabla sin que nadie lo note.
        if avisos:
            print(f"  ⚠️ {len(avisos)} aviso(s) de pandoc:")
            for a in avisos[:20]:
                print(f"     {a}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fuente", type=Path, default=FUENTE)
    ap.add_argument("--destino", type=Path, default=DESTINO)
    args = ap.parse_args()
    return convierte(args.fuente, args.destino)


if __name__ == "__main__":
    sys.exit(main())
