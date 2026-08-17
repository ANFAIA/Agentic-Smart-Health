"""Catálogo de **casos límite**: entradas rotas, degeneradas o mal etiquetadas.

`synthetic.py` genera el caso que *sale bien* —tres modalidades coherentes de la
misma boca— y con él se comprueba que la ingesta funciona. Esto genera lo otro: lo
que llega un martes desde una clínica y no se parece a nada de lo previsto.

**Por qué hace falta un catálogo y no un `write_bytes` en cada test.** El diseño
*fail-loud* del pipeline (`base.py`) solo se ejercita de verdad con entradas
hostiles, y esas entradas estaban dispersas e improvisadas: cada test inventaba su
propio fichero roto, así que ningún agente se enfrentaba a los mismos casos que los
demás. Aquí están en una lista única, cada uno con **qué se espera** y **por qué**,
de modo que añadir un caso lo aplica a todos los tests que recorren `CASES`.

**Tres resultados posibles, y los tres son información.** No todo caso límite tiene
que fallar: un informe con un pH imposible debe ingerirse bien *descartando esa
línea* (`OK`), mientras que un DICOM sin píxeles no puede producir un volumen
(`FAILED`). Lo que ningún caso puede hacer es lanzar una excepción, ni —peor—
devolver `OK` con un dato inventado.

Todo es sintético y determinista: ni un byte procede de un paciente.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from core_schemas import ModalityStatus

from ingestion_agents import synthetic

# Volumen mínimo para las variantes de DICOM: lo que importa es la cabecera, no el
# contenido, y un volumen grande multiplicaría la suite sin añadir señal.
_CORTES, _FILAS, _COLUMNAS = 4, 12, 12


@dataclass(frozen=True)
class EdgeCase:
    """Una entrada límite, qué agente la recibe y qué debe pasar con ella."""

    name: str
    modality: str
    """`mesh` · `cbct` · `report` · `image` · `any` (lo prueban todos)."""
    expected: ModalityStatus | None
    """Estado exigido. `None` = cualquiera vale, pero nunca una excepción."""
    why: str
    build: Callable[[Path], Path]
    """(directorio de trabajo) -> ruta de la fuente ya escrita."""


# --------------------------------------------------------------------------- #
# CBCT / DICOM
# --------------------------------------------------------------------------- #
def _serie_base(destino: Path) -> Path:
    volumen = np.full((_CORTES, _FILAS, _COLUMNAS), -1000.0, dtype=np.float32)
    volumen[:, 4:8, 4:8] = 1500.0  # un bloque denso para que haya algo que umbralizar
    return synthetic.write_dicom_series(destino, volumen, (0.5, 0.5, 0.5))


def _retocar(directorio: Path, cambio: Callable[[object], None]) -> Path:
    """Aplica `cambio` a cada corte de una serie ya escrita."""
    import pydicom

    for fichero in sorted(directorio.glob("*.dcm")):
        ds = pydicom.dcmread(str(fichero))
        cambio(ds)
        pydicom.dcmwrite(str(fichero), ds, enforce_file_format=True)
    return directorio


def _dicom_cabecera_corrupta(tmp: Path) -> Path:
    """Serie válida a la que se le trunca la cabecera del primer corte."""
    destino = _serie_base(tmp / "dicom-cabecera-corrupta")
    primero = sorted(destino.glob("*.dcm"))[0]
    primero.write_bytes(primero.read_bytes()[:64])
    return destino


def _dicom_sin_pixeldata(tmp: Path) -> Path:
    destino = _serie_base(tmp / "dicom-sin-pixeldata")

    def quitar(ds: object) -> None:
        del ds.PixelData  # type: ignore[attr-defined]

    return _retocar(destino, quitar)


def _dicom_modalidad_ajena(tmp: Path) -> Path:
    """Resonancia magnética con extensión `.dcm`: es DICOM, pero no es un CBCT."""
    destino = _serie_base(tmp / "dicom-modalidad-ajena")

    def cambiar(ds: object) -> None:
        ds.Modality = "MR"  # type: ignore[attr-defined]

    return _retocar(destino, cambiar)


def _dicom_sin_modalidad(tmp: Path) -> Path:
    """Sin etiqueta `Modality`: no se puede afirmar que sea otra cosa."""
    destino = _serie_base(tmp / "dicom-sin-modalidad")

    def quitar(ds: object) -> None:
        del ds.Modality  # type: ignore[attr-defined]

    return _retocar(destino, quitar)


def _dicom_sin_extension(tmp: Path) -> Path:
    """Cortes escritos sin extensión, como los emiten los exportadores clínicos."""
    destino = _serie_base(tmp / "dicom-sin-extension")
    for i, corte in enumerate(sorted(destino.glob("*.dcm")), start=1):
        corte.rename(destino / f"i{i:07d}")
    return destino


def _dicom_espaciado_cero(tmp: Path) -> Path:
    """`PixelSpacing` a cero: un vóxel sin tamaño no tiene coordenadas en mm."""
    destino = _serie_base(tmp / "dicom-espaciado-cero")

    def cambiar(ds: object) -> None:
        ds.PixelSpacing = [0.0, 0.0]  # type: ignore[attr-defined]
        ds.SliceThickness = 0.0  # type: ignore[attr-defined]

    return _retocar(destino, cambiar)


def _dicom_espaciado_negativo(tmp: Path) -> Path:
    destino = _serie_base(tmp / "dicom-espaciado-negativo")

    def cambiar(ds: object) -> None:
        ds.PixelSpacing = [-0.5, -0.5]  # type: ignore[attr-defined]

    return _retocar(destino, cambiar)


def _dicom_cortes_heterogeneos(tmp: Path) -> Path:
    """Dos tamaños de corte en el mismo directorio: no es un volumen único."""
    destino = _serie_base(tmp / "dicom-cortes-heterogeneos")
    otro = _serie_base(tmp / "_extra")
    import pydicom

    ds = pydicom.dcmread(str(sorted(otro.glob("*.dcm"))[0]))
    ds.Rows, ds.Columns = _FILAS // 2, _COLUMNAS // 2
    ds.PixelData = np.zeros((_FILAS // 2, _COLUMNAS // 2), dtype=np.uint16).tobytes()
    ds.ImagePositionPatient = [0.0, 0.0, 99.0]
    pydicom.dcmwrite(str(destino / "slice_9999.dcm"), ds, enforce_file_format=True)
    return destino


def _dicom_corte_suelto(tmp: Path) -> Path:
    """Un único corte: es una radiografía, no una serie volumétrica."""
    destino = _serie_base(tmp / "dicom-corte-suelto")
    for sobrante in sorted(destino.glob("*.dcm"))[1:]:
        sobrante.unlink()
    return destino


def _dicom_directorio_sin_dcm(tmp: Path) -> Path:
    destino = tmp / "dicom-vacio"
    destino.mkdir(parents=True, exist_ok=True)
    (destino / "leeme.txt").write_text("aquí no hay DICOM", encoding="utf-8")
    return destino


# --------------------------------------------------------------------------- #
# Malla
# --------------------------------------------------------------------------- #
def _obj_sin_vertices(tmp: Path) -> Path:
    destino = tmp / "sin-vertices.obj"
    destino.write_text("# malla vacía\nf 1 2 3\n", encoding="utf-8")
    return destino


def _obj_indice_fuera_de_rango(tmp: Path) -> Path:
    destino = tmp / "indice-fuera.obj"
    destino.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 999\n", encoding="utf-8")
    return destino


def _obj_con_nan(tmp: Path) -> Path:
    """Coordenadas no finitas: envenenan cualquier normal o caja envolvente."""
    destino = tmp / "con-nan.obj"
    destino.write_text("v nan 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    return destino


def _stl_truncado(tmp: Path) -> Path:
    """STL binario que anuncia más triángulos de los que trae."""
    destino = tmp / "truncado.stl"
    cabecera = b"\0" * 80 + (1000).to_bytes(4, "little")
    destino.write_bytes(cabecera + b"\0" * 50)  # un triángulo donde dice mil
    return destino


# --------------------------------------------------------------------------- #
# Informe
# --------------------------------------------------------------------------- #
def _informe_vacio(tmp: Path) -> Path:
    destino = tmp / "informe-vacio.txt"
    destino.write_text("", encoding="utf-8")
    return destino


def _informe_valores_imposibles(tmp: Path) -> Path:
    """pH fuera de escala y un FDI que no existe: se descartan, no se ingieren."""
    destino = tmp / "informe-imposible.txt"
    destino.write_text(
        "Hallazgos:\n  - Diente 16: pH 15.0\n  - Diente 99: pH 6.5\n  - Diente 21: pH -2.0\n",
        encoding="utf-8",
    )
    return destino


def _pdf_falso(tmp: Path) -> Path:
    destino = tmp / "informe-falso.pdf"
    destino.write_bytes(b"%PDF-1.4\nesto no es la estructura de un PDF")
    return destino


# --------------------------------------------------------------------------- #
# Imagen
# --------------------------------------------------------------------------- #
def _png_truncado(tmp: Path) -> Path:
    from PIL import Image

    completa = tmp / "_completa.png"
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(completa)
    destino = tmp / "truncada.png"
    destino.write_bytes(completa.read_bytes()[:40])
    return destino


# --------------------------------------------------------------------------- #
# Genéricos: los recibe cualquier agente
# --------------------------------------------------------------------------- #
def _fichero_vacio(tmp: Path) -> Path:
    destino = tmp / "vacio.dat"
    destino.write_bytes(b"")
    return destino


def _binario_aleatorio(tmp: Path) -> Path:
    destino = tmp / "ruido.bin"
    destino.write_bytes(np.random.default_rng(7).bytes(4096))
    return destino


def _ruta_unicode(tmp: Path) -> Path:
    """Nombre con acentos, guion largo y paréntesis: existe en clínicas reales."""
    destino = tmp / "informe ñandú — copia (1).txt"
    destino.write_text("Diente 16: pH 5.2\n", encoding="utf-8")
    return destino


def _enlace_roto(tmp: Path) -> Path:
    destino = tmp / "enlace-roto.obj"
    destino.symlink_to(tmp / "destino-que-no-existe.obj")
    return destino


def _directorio_donde_va_un_fichero(tmp: Path) -> Path:
    destino = tmp / "esto-es-un-directorio.obj"
    destino.mkdir(parents=True, exist_ok=True)
    return destino


# --------------------------------------------------------------------------- #
# El catálogo
# --------------------------------------------------------------------------- #
CASES: tuple[EdgeCase, ...] = (
    # --- CBCT ---------------------------------------------------------------
    EdgeCase(
        "dicom-cabecera-corrupta",
        "cbct",
        ModalityStatus.FAILED,
        "Un corte ilegible invalida la serie: media boca no es un volumen. Todo "
        "fichero no accesorio es candidato a corte, y uno que no supere la firma "
        "`DICM` es un fallo — nunca algo que saltarse en silencio.",
        _dicom_cabecera_corrupta,
    ),
    EdgeCase(
        "dicom-sin-pixeldata",
        "cbct",
        ModalityStatus.FAILED,
        "Sin píxeles no hay atenuación que convertir en gaussianas.",
        _dicom_sin_pixeldata,
    ),
    EdgeCase(
        "dicom-modalidad-ajena",
        "cbct",
        ModalityStatus.FAILED,
        "Una resonancia no es un CBCT: sus valores no son unidades Hounsfield, "
        "y tratarlos como tales produce densidades inventadas.",
        _dicom_modalidad_ajena,
    ),
    EdgeCase(
        "dicom-sin-modalidad",
        "cbct",
        ModalityStatus.OK,
        "Decisión declarada: sin etiqueta `Modality` no se puede afirmar que "
        "NO sea un CT, y rechazar tumbaría estudios anonimizados con "
        "herramientas agresivas. Se ingiere; lo que se rechaza es la "
        "modalidad ajena declarada, no la ausente.",
        _dicom_sin_modalidad,
    ),
    EdgeCase(
        "dicom-sin-extension",
        "cbct",
        ModalityStatus.OK,
        "Encontrado sobre CBCT real (Carestream CS 9600): 578 cortes llamados "
        "`i0000567`, sin extensión. Filtrar por `.dcm` veía un directorio vacío y "
        "declaraba un fallo del dato que era nuestro. Se reconoce por la firma "
        "`DICM` del byte 128, que es lo único fiable.",
        _dicom_sin_extension,
    ),
    EdgeCase(
        "dicom-espaciado-cero",
        "cbct",
        ModalityStatus.FAILED,
        "Un vóxel sin tamaño no tiene coordenadas en mm: el campo saldría colapsado en un punto.",
        _dicom_espaciado_cero,
    ),
    EdgeCase(
        "dicom-espaciado-negativo",
        "cbct",
        ModalityStatus.FAILED,
        "Espaciado negativo refleja el volumen: izquierda y derecha del "
        "paciente se intercambian en silencio.",
        _dicom_espaciado_negativo,
    ),
    EdgeCase(
        "dicom-cortes-heterogeneos",
        "cbct",
        ModalityStatus.FAILED,
        "Dos tamaños de corte en un directorio son dos estudios, no uno.",
        _dicom_cortes_heterogeneos,
    ),
    EdgeCase(
        "dicom-corte-suelto",
        "cbct",
        None,
        "Un solo corte es una radiografía. Se acepta o se rechaza, pero de forma declarada.",
        _dicom_corte_suelto,
    ),
    EdgeCase(
        "dicom-directorio-sin-dcm",
        "cbct",
        ModalityStatus.FAILED,
        "Modalidad anunciada y no aportada: hay directorio, no hay estudio.",
        _dicom_directorio_sin_dcm,
    ),
    # --- Malla --------------------------------------------------------------
    EdgeCase(
        "obj-sin-vertices",
        "mesh",
        ModalityStatus.FAILED,
        "Caras que apuntan a vértices inexistentes: no hay superficie.",
        _obj_sin_vertices,
    ),
    EdgeCase(
        "obj-indice-fuera-de-rango",
        "mesh",
        ModalityStatus.FAILED,
        "Un índice fuera de rango leería memoria ajena o daría una cara inventada.",
        _obj_indice_fuera_de_rango,
    ),
    EdgeCase(
        "obj-con-nan",
        "mesh",
        ModalityStatus.FAILED,
        "Un NaN se propaga a normales y caja envolvente sin dar error: es el "
        "peor fallo posible, el que no se nota.",
        _obj_con_nan,
    ),
    EdgeCase(
        "stl-truncado",
        "mesh",
        ModalityStatus.FAILED,
        "La cabecera anuncia mil triángulos y trae uno.",
        _stl_truncado,
    ),
    # --- Informe ------------------------------------------------------------
    EdgeCase(
        "informe-vacio",
        "report",
        None,
        "Un informe sin hallazgos es legítimo: cero observaciones, no un fallo.",
        _informe_vacio,
    ),
    EdgeCase(
        "informe-valores-imposibles",
        "report",
        ModalityStatus.OK,
        "pH 15, pH -2 y el diente 99 se descartan línea a línea; el resto del "
        "informe sigue siendo válido.",
        _informe_valores_imposibles,
    ),
    EdgeCase(
        "pdf-falso",
        "report",
        ModalityStatus.FAILED,
        "Extensión `.pdf` con contenido que no lo es.",
        _pdf_falso,
    ),
    # --- Imagen -------------------------------------------------------------
    EdgeCase(
        "png-truncado",
        "image",
        ModalityStatus.FAILED,
        "Descarga a medias: el PNG empieza bien y se corta.",
        _png_truncado,
    ),
    # --- Genéricos ----------------------------------------------------------
    EdgeCase("fichero-vacio", "any", None, "Cero bytes con extensión válida.", _fichero_vacio),
    EdgeCase(
        "binario-aleatorio",
        "any",
        None,
        "Ruido puro: nada debe interpretarlo como dato.",
        _binario_aleatorio,
    ),
    EdgeCase(
        "ruta-unicode",
        "any",
        None,
        "Acentos, guion largo y paréntesis en el nombre. Existe en clínicas "
        "reales y rompe cualquier manejo ingenuo de rutas.",
        _ruta_unicode,
    ),
    EdgeCase(
        "enlace-roto",
        "any",
        ModalityStatus.MISSING,
        "Un symlink colgando es una modalidad no aportada, no un fallo.",
        _enlace_roto,
    ),
    EdgeCase(
        "directorio-donde-va-un-fichero",
        "any",
        None,
        "Directorio con extensión de fichero.",
        _directorio_donde_va_un_fichero,
    ),
)


def by_modality(modality: str) -> tuple[EdgeCase, ...]:
    """Los casos de una modalidad más los genéricos."""
    return tuple(c for c in CASES if c.modality in (modality, "any"))


def write_all(root: Path) -> dict[str, Path]:
    """Materializa el catálogo entero. Útil para inspeccionarlo a mano."""
    root = Path(root)
    salidas = {}
    for caso in CASES:
        trabajo = root / caso.name
        trabajo.mkdir(parents=True, exist_ok=True)
        salidas[caso.name] = caso.build(trabajo)
    return salidas
