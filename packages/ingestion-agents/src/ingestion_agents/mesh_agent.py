"""`mesh-agent` — malla intraoral (OBJ / STL) → soporte **superficial**.

Modalidad `mesh`, soporte `SURFACE`. Es un worker **determinista**: la entrada ya
tiene esquema (formato de fichero conocido), así que no hay nada que un LLM pueda
aportar. Determinismo = reproducibilidad = medible, que es lo que exige la
métrica de fiabilidad >95%.

**Qué produce.** Vértices, caras, normales por vértice y `color_superficie` (RGB
de la malla intraoral, ADR 001 §Color) persistidos como un artefacto
direccionado por contenido. **No** produce el campo gaussiano —eso es del
`cbct-agent`— ni etiquetas FDI —eso es del `segmentation-agent`—.

🔒 **Guardarraíl de reversibilidad.** El brief exige regenerar el STL desde el
twin con **<0,1 mm** de error, y una nube splatteada no llega (es lossy). Por eso
este agente conserva la **superficie de origen tal cual** —posiciones en float64
y la topología de caras completa—, no una versión remuestreada: el artefacto de
la malla es la copia fiel desde la que se reconstruye. El round-trip
fichero → artefacto → fichero tiene error **cero**, no «pequeño».

**Por qué un parser propio y no VTK.** El OBJ/STL de un escáner intraoral es un
formato trivial y VTK son ~100 MB de dependencia binaria para una capa de ingesta
que debe correr en CI. El notebook de investigación sí usa VTK; el agente de
producción no lo necesita.

**OBJ vs STL.** El OBJ del escáner puede traer **color por vértice** (`v x y z r g b`);
el **STL siempre es «pelado»** (solo geometría), así que su `color_superficie` es
`None` por construcción — es la distinción que exige la regla del `mesh-agent`
(ADR 004). Bite2Text entrega STL binario; Teeth3DS+, OBJ.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from core_schemas import Modality, Support

from ingestion_agents.base import BaseIngestionAgent, IngestionOutput
from ingestion_agents.store import ArtifactStore

# El escáner escribe el color en [0,1]; el contrato (`Color`) lo quiere en [0,255].
_COLOR_SCALE = 255.0


def parse_obj(path: Path) -> dict[str, np.ndarray]:
    """Lee un OBJ de malla intraoral: posiciones, color por vértice y caras.

    Soporta la forma que produce el dataset (`v x y z r g b`, color opcional) y
    los índices de cara con o sin `v/vt/vn`. Devuelve caras **0-indexadas**
    (el OBJ las escribe 1-indexadas).
    """
    positions: list[tuple[float, float, float]] = []
    colors: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            if not raw or raw[0] not in "vf":
                continue
            parts = raw.split()
            if not parts:
                continue
            tag = parts[0]
            if tag == "v":
                vals = parts[1:]
                if len(vals) < 3:
                    raise ValueError(f"Vértice con menos de 3 coordenadas: {raw!r}")
                positions.append((float(vals[0]), float(vals[1]), float(vals[2])))
                if len(vals) >= 6:
                    colors.append((float(vals[3]), float(vals[4]), float(vals[5])))
            elif tag == "f":
                idx = [int(tok.split("/")[0]) for tok in parts[1:]]
                if len(idx) < 3:
                    raise ValueError(f"Cara con menos de 3 vértices: {raw!r}")
                # Triangula en abanico cualquier polígono (quads incluidos).
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0] - 1, idx[k] - 1, idx[k + 1] - 1))

    if not positions:
        raise ValueError(f"El OBJ no contiene vértices: {path}")

    out: dict[str, np.ndarray] = {
        # float64, NO float32: guardarraíl de reversibilidad. El OBJ escribe ~8
        # decimales; float64 los representa exactamente, así que el round-trip
        # fichero → artefacto → fichero tiene error CERO. Con float32 el error
        # sería de ~10⁻⁵ mm — despreciable frente a los 0,1 mm de la métrica, pero
        # es una pérdida gratuita en la única copia fiel de la superficie.
        "positions": np.asarray(positions, dtype=np.float64),
        "faces": np.asarray(faces, dtype=np.int32).reshape(-1, 3),
    }
    if colors:
        if len(colors) != len(positions):
            raise ValueError(
                f"Color parcial: {len(colors)} colores para {len(positions)} vértices."
            )
        out["colors"] = np.asarray(colors, dtype=np.float32)
    return out


def _dedup_triangle_soup(tri_verts: np.ndarray) -> dict[str, np.ndarray]:
    """(3T, 3) vértices por triángulo → posiciones únicas + caras 0-indexadas.

    El STL es una **sopa de triángulos**: repite cada vértice compartido. Se
    deduplican con `np.unique` (que además da el índice inverso), reconstruyendo
    la topología que el formato no guarda. Sin esto, las normales por vértice y
    cualquier etiqueta posterior no podrían compartirse entre caras vecinas.
    """
    positions, inverse = np.unique(tri_verts, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3).astype(np.int32)
    return {"positions": positions.astype(np.float64), "faces": faces}


def parse_stl(path: Path) -> dict[str, np.ndarray]:
    """Lee un STL (binario o ASCII) → posiciones únicas + caras. **Sin color.**

    El STL no lleva color ni topología compartida: se reconstruye la topología
    deduplicando vértices. Se detecta binario por el tamaño exacto que impone el
    formato (cabecera 80 B + `uint32` de conteo + 50 B por triángulo); cualquier
    otra cosa se trata como ASCII.
    """
    data = path.read_bytes()
    if len(data) >= 84:
        n_tri = struct.unpack_from("<I", data, 80)[0]
        if len(data) == 84 + n_tri * 50:  # tamaño exacto ⇒ binario, sin ambigüedad
            tri = np.frombuffer(
                data,
                dtype=np.dtype([("n", "<3f4"), ("v", "<3,3f4"), ("attr", "<u2")]),
                count=n_tri,
                offset=84,
            )
            return _dedup_triangle_soup(tri["v"].reshape(-1, 3))

    # ASCII: líneas 'vertex x y z', cada 3 forman un triángulo.
    verts: list[tuple[float, float, float]] = []
    for raw in data.decode("utf-8", errors="replace").splitlines():
        s = raw.strip()
        if s.startswith("vertex"):
            _, x, y, z = s.split()[:4]
            verts.append((float(x), float(y), float(z)))
    if not verts:
        raise ValueError(f"El STL no contiene vértices (¿fichero corrupto?): {path}")
    if len(verts) % 3 != 0:
        raise ValueError(f"STL ASCII con vértices no múltiplos de 3: {path}")
    return _dedup_triangle_soup(np.asarray(verts, dtype=np.float32))


def read_mesh(path: Path) -> dict[str, np.ndarray]:
    """Despacha por extensión: `.obj` → `parse_obj`, `.stl` → `parse_stl`."""
    suffix = path.suffix.lower()
    if suffix == ".obj":
        return parse_obj(path)
    if suffix == ".stl":
        return parse_stl(path)
    raise ValueError(
        f"`mesh-agent` ingiere malla intraoral OBJ o STL, no {suffix!r}."
    )


def vertex_normals(positions: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Normales por vértice = media de las normales de cara ponderada por área.

    Se acumulan las normales de cara **sin normalizar** (su módulo es 2·área), lo
    que da la ponderación por área gratis, y se normaliza al final. Preserva el
    conteo y el orden de los vértices —requisito para que el color, las normales
    y las futuras etiquetas FDI sigan alineados índice a índice—.
    """
    normals = np.zeros_like(positions, dtype=np.float64)
    if faces.size:
        v0, v1, v2 = (positions[faces[:, i]].astype(np.float64) for i in range(3))
        face_n = np.cross(v1 - v0, v2 - v0)
        for i in range(3):
            np.add.at(normals, faces[:, i], face_n)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    # Un vértice suelto (sin caras) queda con normal nula: se deja en cero en vez
    # de inventar una dirección.
    np.divide(normals, lengths, out=normals, where=lengths > 0)
    return normals.astype(np.float32)


class MeshAgent(BaseIngestionAgent):
    """Ingiere la malla intraoral y deja la superficie lista para la fusión."""

    name = "mesh-agent"
    version = "0.1.0"
    modality = Modality.MESH
    support = Support.SURFACE

    def __init__(
        self, store: ArtifactStore, *, quarantine_dir: str | Path | None = None
    ) -> None:
        super().__init__(quarantine_dir=quarantine_dir)
        self.store = store

    def _ingest(self, source: Path) -> IngestionOutput:
        mesh = read_mesh(source)  # valida la extensión (OBJ/STL) y lanza si no
        positions, faces = mesh["positions"], mesh["faces"]
        arrays: dict[str, np.ndarray] = {
            "positions": positions,
            "faces": faces,
            "normals": vertex_normals(positions, faces),
        }

        # El color es el aporte propio de esta modalidad. Si falta, la ingesta
        # sigue siendo válida pero vale menos: se declara bajando la confianza,
        # nunca inventando un color por defecto.
        #
        # Un color CONSTANTE en toda la malla no es color: es el *placeholder* que
        # mete el exportador (Teeth3DS+ escribe 0.502 en los 110k vértices). Se
        # trata igual que su ausencia — `color_superficie = None` — porque
        # persistirlo haría que la fusión geométrica pintara las gaussianas con
        # una apariencia que nadie midió.
        colors = mesh.get("colors")
        has_real_color = colors is not None and float(colors.std()) > 0.0
        if has_real_color:
            confidence = 1.0
            arrays["colors_rgb8"] = np.clip(
                colors * _COLOR_SCALE, 0, 255  # type: ignore[operator]
            ).astype(np.uint8)
        else:
            confidence = 0.6 if colors is not None else 0.5

        return self._success(
            source,
            confidence=confidence,
            artifact_ref=self.store.put(**arrays),
            n_primitives=int(positions.shape[0]),
        )
