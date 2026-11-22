#!/usr/bin/env python
"""consistencia_registro.py — ¿Cuánto depende la pose del registro del escaneo que se use?

    ~/.venvs/dental-gpu/bin/python scripts/consistencia_registro.py \\
        --cbct ~/anfaia/histora/CASE-64A77C_files \\
        --escaneo post="$(ls ~/anfaia/histora/*POST\\ HIGIENE\\ UpperJawScan.stl)" \\
        --escaneo previo=~/anfaia/histora/CASE-EB5070_files/PREVIO\\ UpperJawScan.stl \\
        --escaneo repetido="$(ls ~/anfaia/histora/CASE-EB5070_files/*ESCANEADO*Upper*)" \\
        --escaneo vis=~/anfaia/histora/Visualization_DigitalModelUnsectioned_18-28.stl \\
        --etiquetado post --fdi data/processed/fdi-votada/agosto_upper/etiquetas_upper.npy \\
        --par previo,repetido --par post,vis \\
        --reproduce post \\
        --uos-referencia data/processed/caso-agosto/run-5-modelo/export/entrada.uos \\
        --salida data/processed/caso-agosto/consistencia-registro-1

Protocolo, condiciones de validez y lectura: `docs/research/consistencia-registro-caso-b.md`.
Este script solo ejecuta lo que allí está fijado.

Cada escaneo se registra contra el CBCT **por separado**, con el procedimiento del pipeline
que usa `caso_completo.py` (ingesta + `fuse`, sin segmentación ni ajuste de campo, que van
después y no tocan el registro). Para cada par `A,B` se registra además A sobre B
directamente, y cada vértice de corona de A llega al CBCT por dos caminos: `A→CBCT` y
`B→CBCT ∘ A→B`. La distancia entre las dos llegadas es lo que se mide.

**Mide precisión, no exactitud.** Lo que compartan los dos caminos —el mismo CBCT, el mismo
algoritmo, un sesgo de pose común— se cancela y no aparece.

Ningún nombre de fichero del proveedor queda escrito en las salidas (`resultado.json`,
`run-record.json`, ni como destino de un enlace): llevan iniciales y número de caso. Cada
escaneo se identifica por el nombre que se le da aquí y su SHA-256.

Nada de la medida se imprime antes de quedar escrito: una ejecución interrumpida después
de ver un número no podría repetirse según el protocolo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

RAIZ = Path(__file__).resolve().parent.parent
for _src in sorted(RAIZ.glob("packages/*/src")):
    sys.path.insert(0, str(_src))
sys.path.insert(0, str(RAIZ / "apps/agent-orchestrator/src"))
sys.path.insert(0, str(RAIZ / "scripts"))

RESUMEN_EN = "Measures how much the scan-to-CBCT pose depends on which scan is used."

MAX_PRIMITIVAS = 1_500_000
"""El tope de `caso_completo.py`, el de la ejecución de referencia."""

SUELO_MAXIMO_MM = 0.2
"""V2: mediana máxima del registro directo A→B para que el par sea interpretable."""

REPRODUCCION_MM = 0.01
"""V1: diferencia máxima con el registro de la ejecución de referencia."""

COBERTURA_MINIMA = 0.5
"""Una pieza se lee si sus coronas transferidas cubren al menos esta fracción de las de POST
(corregida por la densidad de vértices de cada malla) y su suelo no pasa de `SUELO_MAXIMO_MM`."""

DISTANCIA_TRANSFERENCIA_MM = 0.5
"""Un vértice toma el código FDI de la corona etiquetada más cercana dentro de esta distancia."""

ANTERIORES = (11, 12, 13, 21, 22, 23)
POSTERIORES = (14, 15, 16, 17, 24, 25, 26, 27)


# ----------------------------------------------------------------------------------------
# Distancia exacta de punto a superficie


def _mas_cercano_en_triangulos(p, a, b, c):
    """Punto más cercano de cada triángulo (Ericson, *Real-Time Collision Detection* 5.1.5)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = p - b
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = p - c
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    out = np.empty_like(p)
    hecho = np.zeros(len(p), bool)

    def pon(mascara, valor):
        m = mascara & ~hecho
        out[m] = valor[m]
        hecho[m] = True

    with np.errstate(divide="ignore", invalid="ignore"):
        pon((d1 <= 0) & (d2 <= 0), a)
        pon((d3 >= 0) & (d4 <= d3), b)
        pon((d6 >= 0) & (d5 <= d6), c)
        pon((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + (d1 / (d1 - d3))[:, None] * ab)
        pon((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + (d2 / (d2 - d6))[:, None] * ac)
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        pon((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), b + w[:, None] * (c - b))
        den = 1.0 / (va + vb + vc)
        pon(np.ones(len(p), bool), a + (vb * den)[:, None] * ab + (vc * den)[:, None] * ac)
    return out


class Superficie:
    """Distancia exacta a una malla, también junto a triángulos largos.

    La distancia al vértice más cercano no vale aquí: con un espaciado de ~0,2 mm se infla
    del orden de 0,1 mm aunque las superficies coincidan, que es el tamaño de lo que se mide.

    **Por qué es exacta.** Sea `d` la distancia al vértice más cercano: el punto más cercano
    de la superficie está a `D ≤ d`, sobre algún triángulo T. Si T es corto (su vértice más
    alejado del centroide está a `R ≤ radio_corto`), sus tres vértices están a menos de
    `d + 2·radio_corto` y T sale de los triángulos que tocan a esos vértices. Si es largo, su
    centroide está a menos de `d + R_max` y sale de un árbol aparte de centroides largos.
    La primera versión tomaba solo los 16 vértices más cercanos y fallaba junto a los
    triángulos largos del modelo Visualization (hasta 0,11 mm en 3 de 3000 puntos).
    """

    def __init__(self, V: np.ndarray, F: np.ndarray, radio_corto: float = 0.3):
        self.V, self.F = np.asarray(V, float), np.asarray(F, np.int64)
        self.radio_corto = radio_corto
        self.arbol = cKDTree(self.V)
        orden = np.argsort(self.F.ravel(), kind="stable")
        self.caras_de = orden // 3
        self.inicio = np.searchsorted(self.F.ravel()[orden], np.arange(len(self.V) + 1))
        tri = self.V[self.F]
        centro = tri.mean(1)
        radio = np.linalg.norm(tri - centro[:, None], axis=2).max(1)
        self.largas = np.nonzero(radio > radio_corto)[0]
        self.arbol_largas = cKDTree(centro[self.largas]) if len(self.largas) else None
        self.radio_largas = float(radio[self.largas].max()) if len(self.largas) else 0.0

    @staticmethod
    def _aplana(listas) -> tuple[np.ndarray, np.ndarray]:
        n = np.fromiter((len(x) for x in listas), dtype=np.int64, count=len(listas))
        if n.sum() == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        return np.repeat(np.arange(len(listas)), n), np.concatenate(
            [np.asarray(x, np.int64) for x in listas if len(x)])

    def _evalua(self, q: np.ndarray, pares_q: np.ndarray, caras: np.ndarray,
                mejor: np.ndarray, tramo: int = 2_000_000) -> None:
        for s in range(0, len(caras), tramo):
            iq, f = pares_q[s:s + tramo], caras[s:s + tramo]
            tri = self.V[self.F[f]]
            cp = _mas_cercano_en_triangulos(q[iq], tri[:, 0], tri[:, 1], tri[:, 2])
            np.minimum.at(mejor, iq, np.linalg.norm(q[iq] - cp, axis=1))

    def distancia(self, P: np.ndarray, lote: int = 5_000) -> np.ndarray:
        P = np.asarray(P, float)
        res = np.empty(len(P))
        for s in range(0, len(P), lote):
            q = P[s:s + lote]
            d, _ = self.arbol.query(q)
            mejor = d.copy()
            iq, v = self._aplana(self.arbol.query_ball_point(q, r=d + 2 * self.radio_corto))
            n = self.inicio[v + 1] - self.inicio[v]
            base = np.repeat(self.inicio[v], n)
            desplaza = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
            self._evalua(q, np.repeat(iq, n), self.caras_de[base + desplaza], mejor)
            if self.arbol_largas is not None:
                iq2, j = self._aplana(
                    self.arbol_largas.query_ball_point(q, r=d + self.radio_largas))
                self._evalua(q, iq2, self.largas[j], mejor)
            res[s:s + lote] = mejor
        return res


# ----------------------------------------------------------------------------------------
# Transformaciones


def _a_4x4(R: np.ndarray, t: np.ndarray, s: float = 1.0) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = s * np.asarray(R, float)
    m[:3, 3] = np.asarray(t, float)
    return m


def aplica(m: np.ndarray, X: np.ndarray) -> np.ndarray:
    return X @ m[:3, :3].T + m[:3, 3]


def _de_icp(r) -> np.ndarray:
    from fusion_agents.registration import quaternion_to_matrix

    return _a_4x4(quaternion_to_matrix(r.rotation), np.asarray(r.translation))


def _umeyama(S: np.ndarray, D: np.ndarray) -> np.ndarray:
    ms, md = S.mean(0), D.mean(0)
    S0, D0 = S - ms, D - md
    U, sv, Vt = np.linalg.svd(D0.T @ S0 / len(S))
    G = np.eye(3)
    G[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ G @ Vt
    s = float(np.trace(np.diag(sv) @ G) / S0.var(0).sum())
    return _a_4x4(R, md - s * R @ ms, s)


def angulo_grados(m: np.ndarray) -> float:
    R = m[:3, :3] / np.cbrt(np.linalg.det(m[:3, :3]))
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def _arranques(fuente: np.ndarray, objetivo: np.ndarray) -> list[np.ndarray]:
    """Dos poses de partida: ICP desde la identidad y barrido global de SO(3)."""
    from fusion_agents.registration import icp, icp_global

    return [_de_icp(icp(fuente, objetivo, trim=0.9)),
            _de_icp(icp_global(fuente, objetivo, seed=0))]


def similitud(fuente: np.ndarray, objetivo: np.ndarray) -> tuple[np.ndarray, float]:
    """Ajuste con escala sobre el 70 % de correspondencias más cercanas; el mejor arranque."""
    arbol = cKDTree(objetivo)
    mejor, residuo = None, np.inf
    for m in _arranques(fuente, objetivo):
        for _ in range(30):
            d, i = arbol.query(aplica(m, fuente))
            k = d <= np.percentile(d, 70)
            m = _umeyama(fuente[k], objetivo[i[k]])
        d, _ = arbol.query(aplica(m, fuente))
        r = float(np.median(d[d <= np.percentile(d, 70)]))
        if r < residuo:
            mejor, residuo = m, r
    return mejor, residuo


def directo(fuente: np.ndarray, malla: Superficie) -> tuple[np.ndarray, np.ndarray]:
    """A→B rígido: dos arranques, recortes 0,9 → 0,8 → 0,5; el de menor mediana a la superficie."""
    from fusion_agents.registration import icp

    mejor, mejor_d = None, None
    for m in _arranques(fuente, malla.V):
        for trim in (0.8, 0.5):
            m = _de_icp(icp(aplica(m, fuente), malla.V, trim=trim, scales=(1.0,))) @ m
        d = malla.distancia(aplica(m, fuente))
        if mejor_d is None or np.median(d) < np.median(mejor_d):
            mejor, mejor_d = m, d
    return mejor, mejor_d


# ----------------------------------------------------------------------------------------
# Registro contra el CBCT por el pipeline


def registra_contra_cbct(nombre: str, escaneo: Path, cbct: Path, salida: Path,
                         etiquetas: np.ndarray | None) -> dict:
    """El procedimiento de `caso_completo.py`: ingesta y `fuse`, nada más."""
    from agent_orchestrator import CaseInput, IngestionPipeline
    from fusion_agents.registration import quaternion_to_matrix
    from ingestion_agents import ArtifactStore

    caso_dir = salida / "casos" / nombre / "entrada"   # mismo acquisition_id que run-5
    caso_dir.mkdir(parents=True)
    malla = caso_dir / "escaneo-superior.stl"           # misma arcada leída del nombre
    # Enlace DURO y no simbólico: un simbólico deja escrito en la salida el nombre del
    # fichero del proveedor, que lleva las iniciales del paciente.
    try:
        os.link(escaneo.resolve(), malla)
    except OSError:
        shutil.copyfile(escaneo.resolve(), malla)
    caso = CaseInput(acquisition_id=caso_dir.name, cbct=cbct, mesh=malla, reports=[],
                     images=[])
    pipe = IngestionPipeline(ArtifactStore(salida / "artifacts"),
                             quarantine_dir=salida / "quarantine",
                             cbct_recorte_dental=True, cbct_max_primitivas=MAX_PRIMITIVAS,
                             segmenter_factory=None)
    r = pipe.run(caso)
    if r.snapshot is None:
        raise SystemExit(f"{nombre}: la ingesta no produjo snapshot.")
    fus = pipe.fuse(r, malla=malla, dos_arcadas=True, etiquetas_origen=etiquetas)
    t = fus.snapshot.provenance.transform if fus.snapshot else None
    if t is None:
        raise SystemExit(f"{nombre}: la fusión no dejó transformación.")
    m = _a_4x4(quaternion_to_matrix(t.rotation), np.asarray(t.translation))
    geo = [o for o in fus.fusion if o.agent.startswith("geometric")]
    return {"m": m, "rms_mm": t.rms_mm, "confidence": fus.snapshot.provenance.confidence,
            "status": [str(o.status) for o in geo],
            "detail": [o.detail for o in geo],
            "hitl_reasons": list(fus.hitl_reasons)}


# ----------------------------------------------------------------------------------------


def _resumen(d: np.ndarray) -> dict:
    if len(d) == 0:
        return {"n": 0}
    return {"n": int(len(d)), "median_mm": round(float(np.median(d)), 4),
            "p95_mm": round(float(np.percentile(d, 95)), 4)}


def _sha(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 22), b""):
            h.update(bloque)
    return h.hexdigest()


def _digesto_serie(directorio: Path) -> str:
    """`nombre\\0sha256\\n` ordenado, como `uos.manifiesto.digesto_de_partes` sin UID."""
    h = hashlib.sha256()
    for p in sorted(x for x in directorio.iterdir() if x.is_file()):
        h.update(f"{p.name}\0{_sha(p)}\n".encode())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cbct", type=Path, required=True)
    ap.add_argument("--escaneo", action="append", required=True, metavar="NOMBRE=RUTA")
    ap.add_argument("--etiquetado", required=True, help="Escaneo con etiquetas FDI.")
    ap.add_argument("--fdi", type=Path, required=True)
    ap.add_argument("--par", action="append", required=True, metavar="A,B",
                    help="El primero es el par principal.")
    ap.add_argument("--reproduce", required=True,
                    help="Escaneo que debe reproducir el registro de referencia (V1).")
    ap.add_argument("--uos-referencia", type=Path, required=True)
    ap.add_argument("--salida", type=Path, required=True)
    args = ap.parse_args()

    from caso_completo import entorno_de_software, revision_de_codigo
    from ingestion_agents.mesh_agent import parse_stl

    if args.salida.exists():
        raise SystemExit(f"{args.salida} ya existe: una ejecución no se pisa.")
    np.random.seed(0)
    escaneos = dict(e.split("=", 1) for e in args.escaneo)
    escaneos = {k: Path(v).expanduser() for k, v in escaneos.items()}
    pares = [tuple(p.split(",")) for p in args.par]
    registro = {"schema": "histora-run-record/1.0", "argv": [a if "anfaia" not in a else
                                                             "<ruta local>" for a in sys.argv],
                "seed": 0, "code": revision_de_codigo(),
                "environment": entorno_de_software(),
                "started_at": datetime.now(UTC).isoformat(),
                "inputs": {"scans_sha256": {k: _sha(v) for k, v in escaneos.items()},
                           "fdi_sha256": _sha(args.fdi),
                           "reference_uos_sha256": _sha(args.uos_referencia),
                           "cbct_files_sha256": _digesto_serie(args.cbct)},
                "scope": "Identifies code and environment of one run; not physical calibration"}
    args.salida.mkdir(parents=True)

    mallas = {k: parse_stl(v) for k, v in escaneos.items()}
    V = {k: np.asarray(m["positions"], dtype=np.float64) for k, m in mallas.items()}
    cruda = np.load(args.fdi).astype(np.int64)
    if len(cruda) != len(V[args.etiquetado]):
        raise SystemExit("Las etiquetas FDI no corresponden a los vértices del escaneo.")

    # V1 PRIMERO: si el procedimiento no es el de la ejecución de referencia, se aborta
    # antes de registrar ningún otro escaneo y sin ningún resultado de la medida.
    print("· V1: el escaneo de referencia contra el CBCT")
    import zipfile
    with zipfile.ZipFile(args.uos_referencia) as z:
        manifiesto = json.loads(z.read("manifest.json"))
    reg = next(r for r in manifiesto["registrations"] if r["id"] == "reg.ct_to_ios")
    m_ref = np.linalg.inv(np.asarray(reg["transform_4x4_row_major"], float).reshape(4, 4))
    cbct = {args.reproduce: registra_contra_cbct(
        args.reproduce, escaneos[args.reproduce], args.cbct, args.salida,
        cruda if args.reproduce == args.etiquetado else None)}
    corona_rep = V[args.reproduce][cruda > 0] if args.reproduce == args.etiquetado \
        else V[args.reproduce]
    v1 = float(np.linalg.norm(aplica(cbct[args.reproduce]["m"], corona_rep)
                              - aplica(m_ref, corona_rep), axis=1).max())
    print(f"  máx {v1:.2e} mm (límite {REPRODUCCION_MM})")
    if v1 > REPRODUCCION_MM:
        registro.update(finished_at=datetime.now(UTC).isoformat(), aborted="V1",
                        v1_max_mm=v1)
        (args.salida / "run-record.json").write_text(json.dumps(registro, indent=2) + "\n")
        raise SystemExit("V1 no se cumple: el procedimiento no es el de la ejecución de "
                         "referencia. Se aborta antes de medir.")

    # Etiquetas: las mismas limpiezas que `caso_completo.py` aplica antes de usarlas como
    # semillas: rellenar huecos y absorber islas (el «17» de 198 vértices dentro del 16).
    from analysis_agents import absorbe_islas, rellena_etiquetas

    limpia, actas = absorbe_islas(V[args.etiquetado], rellena_etiquetas(V[args.etiquetado],
                                                                        cruda))
    etq = {args.etiquetado: limpia}
    ref = V[args.etiquetado][limpia > 0]
    etq_ref = limpia[limpia > 0]
    transferencia = {}
    for k in escaneos:
        if k in etq:
            continue
        m, residuo = similitud(ref, V[k])
        d, i = cKDTree(aplica(m, ref)).query(V[k])
        etq[k] = np.where(d <= DISTANCIA_TRANSFERENCIA_MM, etq_ref[i], 0)
        transferencia[k] = {"scale": round(float(np.cbrt(np.linalg.det(m[:3, :3]))), 5),
                            "trimmed_median_mm": round(residuo, 4),
                            "crown_vertices": int((etq[k] > 0).sum())}

    print("· registro del resto de escaneos contra el CBCT")
    for k, ruta in escaneos.items():
        if k not in cbct:
            cbct[k] = registra_contra_cbct(k, ruta, args.cbct, args.salida,
                                           etq[k] if k == args.etiquetado else None)

    print("· pares (los números se escriben antes de mostrarse)")
    sup = {k: Superficie(V[k], mallas[k]["faces"]) for k in mallas}
    n_ref = {int(c): int((etq_ref == c).sum()) for c in np.unique(etq_ref)}
    resultados = []
    for a, b in pares:
        corona = V[a][etq[a] > 0]
        codigos = etq[a][etq[a] > 0]
        m_ab, suelo = directo(corona, sup[b])
        llega_a = aplica(cbct[a]["m"], corona)
        llega_b = aplica(cbct[b]["m"], aplica(m_ab, corona))
        d = np.linalg.norm(llega_a - llega_b, axis=1)
        delta = cbct[b]["m"] @ m_ab @ np.linalg.inv(cbct[a]["m"])
        centro = llega_a.mean(0)
        densidad = len(V[a]) / len(V[args.etiquetado])
        por_pieza, legibles = {}, []
        for c in np.unique(codigos):
            sel = codigos == c
            cobertura = float(sel.sum() / (n_ref.get(int(c), 0) * densidad)) \
                if n_ref.get(int(c)) else 0.0
            legible = bool(cobertura >= COBERTURA_MINIMA
                           and np.median(suelo[sel]) <= SUELO_MAXIMO_MM)
            if legible:
                legibles.append(int(c))
            por_pieza[int(c)] = {"discrepancy": _resumen(d[sel]), "floor": _resumen(suelo[sel]),
                                 "coverage": round(cobertura, 3), "readable": legible}
        en = np.isin(codigos, legibles)
        resultados.append({
            "pair": [a, b], "principal": (a, b) == pares[0],
            "direct_floor": _resumen(suelo),
            "v2_floor_ok": bool(np.median(suelo) <= SUELO_MAXIMO_MM),
            "readable_teeth": legibles,
            "discrepancy_all_teeth": _resumen(d),
            "discrepancy_readable": _resumen(d[en]),
            "discrepancy_anterior_readable": _resumen(d[en & np.isin(codigos, ANTERIORES)]),
            "discrepancy_posterior_readable": _resumen(d[en & np.isin(codigos, POSTERIORES)]),
            "rigid_difference": {"angle_deg": round(angulo_grados(delta), 4),
                                 "translation_at_crown_centroid_mm": round(float(
                                     np.linalg.norm(aplica(delta, centro[None])[0] - centro)), 4)},
            "per_tooth": por_pieza,
        })

    salida = {
        "scope": "Precision of the scan-to-CBCT registration across scans of the same "
                 "dentition. Not accuracy, not a TRE.",
        "protocol": "docs/research/consistencia-registro-caso-b.md",
        "v1_reproduces_reference_max_mm": v1,
        "v1_ok": True,
        "label_cleaning": {"islands_absorbed": [list(map(int, x)) for x in actas]},
        "label_transfer": transferencia,
        "registrations": {k: {kk: vv for kk, vv in c.items() if kk != "m"}
                          | {"transform_4x4": c["m"].round(9).tolist()}
                          for k, c in cbct.items()},
        "pairs": resultados,
    }
    (args.salida / "resultado.json").write_text(json.dumps(salida, indent=2,
                                                           ensure_ascii=False) + "\n")
    registro["finished_at"] = datetime.now(UTC).isoformat()
    registro["outputs"] = {"resultado.json": _sha(args.salida / "resultado.json")}
    (args.salida / "run-record.json").write_text(json.dumps(registro, indent=2,
                                                            ensure_ascii=False) + "\n")

    for f in resultados:
        print(f"· {f['pair'][0]}→{f['pair'][1]}{' (principal)' if f['principal'] else ''}: "
              f"suelo {f['direct_floor']} · legibles {f['readable_teeth']} · "
              f"discrepancia {f['discrepancy_readable']} · anteriores "
              f"{f['discrepancy_anterior_readable']} · posteriores "
              f"{f['discrepancy_posterior_readable']} · {f['rigid_difference']}")
    v2 = all(f["v2_floor_ok"] for f in resultados if f["principal"])
    print(f"· validez: V1 sí · V2 (principal) {'sí' if v2 else 'NO'}")
    return 0 if v2 else 2

if __name__ == "__main__":
    raise SystemExit(main())
