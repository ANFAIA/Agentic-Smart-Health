"""§5.2 — el sidecar `<id>.volume.json` de una serie DICOM.

**Para que existe.** El DICOM va intacto dentro del `.uos`, que es el punto del formato:
la fuente de verdad no se transcodifica. Pero un visor web no deberia necesitar un parser
DICOM completo —397 ficheros, sintaxis de transferencia, rescale por corte— solo para
saber que dimensiones tiene el volumen y en que rango estan sus valores. El sidecar es esa
descripcion, y **se lee de los mismos bytes que viajan**, no de una copia derivada que
alguien calculo antes en otro sitio.

⚠️ **El sidecar declara el frame, y NADA MAS sobre la alineacion.** La transformada del
CBCT al canonico vive en `registrations` (§6) y no se duplica aqui. Un sistema con dos
sitios donde vive la misma transformada acaba teniendo dos transformadas distintas, y la
que se aplique dependera de a quien le toque leer.

⚠️ **La orientacion se LEE del `ImageOrientationPatient`, no se supone.** Es la misma regla
que en el resto del proyecto: el eje anatomico sale de la cabecera del DICOM o no sale. Un
volumen al que se le supone la orientacion se renderiza igual de bien y espejado.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

SIDECAR = "volume/{id}.volume.json"

# Tags DICOM que identifican a una persona o al sitio donde se le atendio. No es la lista
# completa del perfil de confidencialidad de DICOM (PS3.15 E.1) —esa tiene mas de cien
# entradas y muchas son de equipamiento—, es el nucleo que un export clinico trae poblado
# cuando NO se ha anonimizado. Si alguno de estos tiene valor, la serie no esta
# seudonimizada y el `.uos` no puede decir que si.
#
# ⚠️ `PatientID` NO esta aqui a proposito. Un export anonimizado lo rellena con un
# identificador opaco —el de esta serie es un UUID— y exigir que este vacio rechazaria
# series perfectamente anonimas. Lo que delata es el NOMBRE, la fecha de nacimiento y la
# institucion; el identificador es justo lo que sobrevive a una seudonimizacion correcta.
TAGS_IDENTIFICABLES = (
    "PatientName",
    "PatientBirthDate",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "OtherPatientNames",
    "OtherPatientIDs",
    "InstitutionName",
    "InstitutionAddress",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "OperatorsName",
    "AccessionNumber",
)

# Valores que un anonimizador deja como marcador y que NO son un nombre. Se comparan en
# minusculas y sin espacios: `Anonymized3`, `ANONYMOUS`, `Patient^^^^`.
_MARCADORES = ("anonymous", "anonymized", "anonimo", "removed", "none", "unknown", "test")

# Presets de funcion de transferencia que el spec nombra (§5.2). Se declaran porque el
# visor los ofrece, no porque este fichero los traiga: son nombres, no datos.
PRESETS = ("cbct_bone", "cbct_soft", "cbct_metal_suppress")


def describe_serie(carpeta: Path, *, frame: str) -> tuple[dict[str, Any], list[str]]:
    """`(sidecar, avisos)` de la serie en `carpeta`, leyendo sus cabeceras.

    No carga los pixeles de los 397 cortes: para las dimensiones y la geometria basta la
    cabecera de cada uno, y el rango de valores sale del que el propio DICOM declara
    cuando lo declara. Cuando no, se dice que no se sabe en vez de barrer 259 MB o —peor—
    inventar un rango plausible.
    """
    import pydicom

    ficheros = sorted(p for p in carpeta.rglob("*") if p.is_file())
    cabeceras = []
    for f in ficheros:
        try:
            cabeceras.append(pydicom.dcmread(str(f), stop_before_pixels=True))
        except Exception:  # noqa: BLE001 — un fichero que no es DICOM no es un corte
            continue
    if not cabeceras:
        raise ValueError(f"{carpeta} no contiene ninguna cabecera DICOM legible.")

    avisos: list[str] = []
    primera = cabeceras[0]

    def _z(ds: object) -> float:
        ipp = getattr(ds, "ImagePositionPatient", None)
        return float(ipp[2]) if ipp is not None else float(getattr(ds, "InstanceNumber", 0))

    cabeceras.sort(key=_z)
    primera = cabeceras[0]

    px = [float(v) for v in getattr(primera, "PixelSpacing", [1.0, 1.0])]
    dz = float(getattr(primera, "SliceThickness", 1.0) or 1.0)
    if len(cabeceras) > 1:
        medido = abs(_z(cabeceras[1]) - _z(cabeceras[0]))
        if medido > 0:
            # El espaciado REAL manda sobre el declarado, igual que en el `cbct-agent`.
            dz = medido

    iop = getattr(primera, "ImageOrientationPatient", None)
    if iop is None:
        avisos.append(
            f"la serie de {frame} no trae `ImageOrientationPatient`: el sidecar declara la "
            "orientacion identidad, que es una SUPOSICION y no una medida"
        )
        orientacion = np.eye(3)
    else:
        fila = np.asarray([float(v) for v in iop[:3]], dtype=np.float64)
        columna = np.asarray([float(v) for v in iop[3:]], dtype=np.float64)
        orientacion = np.stack([fila, columna, np.cross(fila, columna)])

    ipp = getattr(primera, "ImagePositionPatient", None)
    if ipp is None:
        avisos.append(
            f"la serie de {frame} no trae `ImagePositionPatient`: sin el, el origen del "
            "volumen en milimetros no se puede declarar y el sidecar lo deja nulo"
        )
    origen = None if ipp is None else [float(v) for v in ipp]

    bajo = getattr(primera, "SmallestImagePixelValue", None)
    alto = getattr(primera, "LargestImagePixelValue", None)
    pendiente = float(getattr(primera, "RescaleSlope", 1.0))
    corte = float(getattr(primera, "RescaleIntercept", 0.0))
    if bajo is None or alto is None:
        # ⚠️ **Se MIDE, y antes se dejaba nulo «porque barrer la serie es caro» (T-2).**
        # El argumento no se sostenia: el escritor ya lee cada byte de cada corte para
        # calcular su `sha256`, asi que sacar el minimo y el maximo en la misma pasada es
        # gratis. Y la consecuencia de dejarlo nulo era real —un visor sin ventana de
        # visualizacion—, o sea que se pagaba un coste que no existia con un defecto que si.
        #
        # `null` queda reservado para el caso legitimo: un contenedor de otro emisor que no
        # tuvo acceso a los pixeles. Nosotros los tenemos, siempre.
        rango = _rango_medido(ficheros, pendiente, corte)
        if rango is None:
            avisos.append(
                f"la serie de {frame} no declara `Smallest/LargestImagePixelValue` y sus "
                "pixeles no se han podido leer: `value_range` se queda nulo y un visor "
                "tendra que calcular su ventana"
            )
    else:
        rango = [float(bajo) * pendiente + corte, float(alto) * pendiente + corte]

    # D-1 · el identificador que DICOM ya define para este sistema de coordenadas. Se LEE,
    # nunca se inventa: es lo unico que permite a un lector que reciba la serie por otro
    # canal saber que es ESA serie, en vez de fiarse de que `frame.ct_001` signifique algo.
    for_uid = getattr(primera, "FrameOfReferenceUID", None)
    if for_uid is None:
        avisos.append(
            f"la serie de {frame} no trae `FrameOfReferenceUID` (0020,0052): el sidecar lo "
            "deja nulo y el frame queda identificado solo por un nombre que se invento el "
            "escritor"
        )
    return {
        "frame": frame,
        "dicom_frame_of_reference_uid": None if for_uid is None else str(for_uid),
        "series_instance_uid": str(getattr(primera, "SeriesInstanceUID", "") or "") or None,
        "study_instance_uid": str(getattr(primera, "StudyInstanceUID", "") or "") or None,
        # D-2 · DICOM impone LPS. No es una eleccion nuestra y por eso va fijo: declarar
        # otra cosa aqui seria describir mal el dato que se esta empaquetando.
        "anatomical": "LPS",
        "dimensions": [int(primera.Columns), int(primera.Rows), len(cabeceras)],
        "spacing_mm": [px[1], px[0], dz],
        "orientation": [[round(float(x), 9) for x in fila] for fila in orientacion],
        "origin_mm": origen,
        "rescale": {"slope": pendiente, "intercept": corte},
        # ⚠️ **Que `rescale` exista NO significa que el resultado sean HU (D-7).** Un CBCT
        # trae las etiquetas y sus grises **no estan calibrados**: dependen del equipo, del
        # campo de vision y de la posicion dentro del volumen, y no son comparables entre
        # escaneres ni convertibles a Hounsfield sin un fantoma. Una TC convencional si.
        # Sin este campo, un lector aplica `slope`/`intercept` y cree tener HU.
        "calibrated_hu": str(getattr(primera, "Modality", "") or "").upper() == "CT",
        "value_range": rango,
        "pixel_encoding": _codificacion(primera),
        "modality": str(getattr(primera, "Modality", "") or ""),
        "transfer_function_presets": list(PRESETS),
        "nota": (
            "leido de las cabeceras de la serie que viaja en este contenedor. La "
            "transformada al frame canonico NO esta aqui: vive en `registrations`. "
            "`calibrated_hu` distingue una TC (grises en Hounsfield) de un CBCT (grises "
            "del equipo): aplicar `rescale` a un CBCT NO da unidades Hounsfield."
        ),
    }, avisos



def _rango_medido(ficheros: list, pendiente: float, corte: float) -> list[float] | None:
    """El minimo y el maximo REALES de la serie, midiendo los pixeles (T-2).

    ⚠️ **La justificacion para dejarlo nulo era el coste, y el coste no es el que decia.**
    El escritor ya lee cada byte de cada corte —para el `sha256` del fichero y, desde D-3,
    para el hash de `PixelData`—, asi que los pixeles ya pasan por memoria. Y la
    consecuencia de dejarlo nulo es concreta: un visor sin ventana de visualizacion, que
    tiene que barrer la serie el mismo o inventarse un rango.

    Devuelve `None` solo cuando los pixeles no se pueden leer, que es el unico caso en el
    que `value_range: null` es una afirmacion honesta y no una excusa.
    """
    import numpy as np
    import pydicom

    bajo = alto = None
    for f in ficheros:
        try:
            px = pydicom.dcmread(str(f)).pixel_array
        except Exception:  # noqa: BLE001 - un fichero ilegible no es un corte
            continue
        b, a = float(np.min(px)), float(np.max(px))
        bajo = b if bajo is None else min(bajo, b)
        alto = a if alto is None else max(alto, a)
    if bajo is None or alto is None:
        return None
    return [bajo * pendiente + corte, alto * pendiente + corte]


def _codificacion(cabecera: object) -> str:
    """`int16-le` y demas: cuantos bits y en que orden vienen los pixeles.

    El orden de bytes lo fija la sintaxis de transferencia del fichero, no una convencion:
    casi todo el DICOM moderno es little-endian, pero «casi todo» no es «todo», y un visor
    que lo suponga lee basura con muy buen aspecto en el que no lo sea.
    """
    bits = int(getattr(cabecera, "BitsAllocated", 16))
    signo = "int" if int(getattr(cabecera, "PixelRepresentation", 1)) == 1 else "uint"
    meta = getattr(cabecera, "file_meta", None)
    sintaxis = str(getattr(meta, "TransferSyntaxUID", "") or "")
    orden = "be" if "Big Endian" in sintaxis else "le"
    return f"{signo}{bits}-{orden}"


def identificables_en(carpeta: Path, *, muestra: int = 8) -> list[str]:
    """Los tags identificables poblados en la serie, o vacio si esta seudonimizada.

    **Por que hace falta.** El DICOM viaja INTACTO, que es el punto del formato, y sus
    cabeceras llevan el nombre del paciente, su fecha de nacimiento y la institucion. El
    manifiesto afirma `phi_state: pseudonymized`; sin mirar los tags, esa afirmacion es una
    suposicion sobre el trabajo de otro. Un contenedor que dice estar seudonimizado y lleva
    el nombre dentro es peor que uno que declara `identified`, porque quien lo reciba
    confiara en el campo y no abrira 397 cabeceras a comprobarlo.

    ⚠️ **No se miran todos los cortes.** Un export clinico es homogeneo: los tags de
    paciente vienen del mismo estudio y se repiten corte a corte. Se toma una muestra
    repartida —principio, medio y final— porque leer 397 cabeceras para encontrar el mismo
    nombre 397 veces no anade nada, y porque un tag que solo aparece en un corte suelto es
    un fichero intruso, que es otro problema y lo caza el validador por su cuenta.
    """
    import pydicom

    ficheros = sorted(p for p in carpeta.rglob("*") if p.is_file())
    if not ficheros:
        return []
    paso = max(1, len(ficheros) // muestra)
    poblados: set[str] = set()
    for f in ficheros[::paso][:muestra]:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
        except Exception:  # noqa: BLE001
            continue
        for tag in TAGS_IDENTIFICABLES:
            valor = str(getattr(ds, tag, "") or "").strip()
            limpio = valor.replace("^", "").replace(" ", "").lower()
            if limpio and not any(m in limpio for m in _MARCADORES):
                poblados.add(tag)
    return sorted(poblados)
