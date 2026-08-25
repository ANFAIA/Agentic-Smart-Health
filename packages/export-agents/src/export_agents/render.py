"""Render multivista del campo gaussiano del twin, y las métricas del canal de imagen.

Tercer canal de la familia de exportación: `gaussian_field_ref` → PNG. El de malla
materializa lo que midió el escáner; el de campo, el interior que sembró el CBCT; éste
convierte ese interior en algo que un humano puede mirar y aprobar.

⚠ **No son las fotos intraorales.** Son renders del gemelo. De las fotos originales solo
se guardaron muestras de apariencia (`image_refs`), así que no hay contra qué compararlas y
no se finge que las haya: lo que se mide aquí es el ciclo *twin → fichero → render*.

## Por qué NO es rasterización de splats

Un rasterizador de 3DGS compone color con `alpha blending`: el orden importa y el
resultado es una radiancia. Este campo no tiene color, y su `density` **no es opacidad**:
es σₙ, atenuación radiológica. La integral correcta a lo largo de un rayo es
**Beer-Lambert**, `I = exp(−∫σ ds)`, que es aditiva en la profundidad óptica y por tanto
**independiente del orden**. Eso no es un detalle de eficiencia: hace el render
determinista sin tener que ordenar por profundidad, y lo vuelve una **radiografía
sintética** (DRR) en vez de una foto inventada.

## Proyección ortográfica, y vistas nombradas por ángulo

Ortográfica a propósito: una perspectiva exigiría intrínsecos de cámara, que son una
decisión de producto pendiente del ADR de motor de render. Para una DRR de una arcada la
ortográfica es además lo estándar.

Y las vistas se nombran por **ángulo**, nunca «oclusal» o «vestibular». El significado
anatómico de un eje depende de cómo el equipo escriba el DICOM, y en este proyecto suponer
eso en vez de leerlo ya salió mal tres veces sobre el mismo paciente. Un nombre como
`az000_el000` no puede mentir; `oclusal` sí.

## Se deposita MASA, no amplitud

Para una gaussiana isótropa de desviación `s` y densidad `σ`, la profundidad óptica en el
plano es `τ(r) = σ·√(2π)·s·exp(−r²/2s²)`. Lo evidente sería evaluar eso en el centro de
cada píxel — y **está mal**, con un error que no se ve: las semillas del `cbct-agent` miden
`s = 0,15 mm` y un píxel ronda 0,7 mm, así que el centro del píxel cae a más de 4σ de casi
todas y la imagen saldría del aliasing, no del campo.

Recortar σ a medio píxel para taparlo es peor, porque infla la amplitud cuando el píxel es
grande: medido sobre `histora`, τ pasaba de 34 a 256 px a **226** a 128 px. Una integral de
línea no puede depender del tamaño del detector.

Así que se deposita la **masa** —`∫∫τ = σ·(2π)^{3/2}·s³`, la integral de σ en el volumen—
con un perfil normalizado a sumar 1, y al final se divide por el área del píxel. Con eso τ
es la profundidad óptica **media del píxel**: converge al refinar, no satura, y el truncado
del soporte a 3σ no pierde masa porque el perfil se renormaliza.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from core_schemas import ModalityStatus, TwinSnapshot

from export_agents.base import (
    RENDER_PSNR_BUDGET_DB,
    RENDER_SSIM_BUDGET,
    BaseExportAgent,
    ExportOutput,
    SurfaceStore,
)
from export_agents.field import _CLAVES_MINIMAS, lee_ply, metadatos_ply

SIGMAS_SOPORTE = 3.0
"""Radio del depósito, en sigmas. Más allá de 3σ queda < 1,1 % de la masa de la gaussiana."""

MARGEN_MM = 2.0
RESOLUCION = 256


@dataclass(frozen=True)
class Vista:
    """Una dirección de cámara. El nombre sale de los ángulos, no de la anatomía."""

    azimut_deg: float
    elevacion_deg: float

    @property
    def nombre(self) -> str:
        return f"az{int(round(self.azimut_deg)) % 360:03d}_el{int(round(self.elevacion_deg)):+03d}"

    @property
    def base(self) -> np.ndarray:
        """`(3, 3)`: filas `[derecha, arriba, vista]`, ortonormal y dextrógira.

        La fila `vista` es la dirección en la que se integra; las otras dos son los ejes
        de la imagen. Se construye por rotaciones y no eligiendo tres vectores a mano,
        que es como se cuela un marco especular.
        """
        az, el = np.radians(self.azimut_deg), np.radians(self.elevacion_deg)
        vista = np.array(
            [np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)], dtype=np.float64
        )
        # `arriba` de referencia: +z salvo que la vista sea casi paralela a él.
        referencia = (
            np.array([0.0, 1.0, 0.0]) if abs(vista[2]) > 0.99 else np.array([0.0, 0.0, 1.0])
        )
        derecha = np.cross(referencia, vista)
        derecha /= np.linalg.norm(derecha)
        arriba = np.cross(vista, derecha)
        return np.stack([derecha, arriba, vista])


VISTAS_POR_DEFECTO: tuple[Vista, ...] = (
    Vista(0.0, 0.0),
    Vista(90.0, 0.0),
    Vista(180.0, 0.0),
    Vista(0.0, 90.0),
)
"""Cuatro direcciones ortogonales. Multivista con el mínimo que cubre los tres ejes."""


def profundidad_optica(
    centers: np.ndarray,
    scales: np.ndarray,
    density: np.ndarray,
    *,
    vista: Vista,
    resolucion: int = RESOLUCION,
    margen_mm: float = MARGEN_MM,
    encuadre: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Integra σ en la dirección de `vista` y devuelve `(resolucion, resolucion)` de τ.

    `encuadre` fija el rectángulo del mundo que se mapea a la imagen. Se pasa **el mismo**
    para todas las vistas de un render y para los dos lados de una comparación: recalcularlo
    por imagen la haría depender de los propios datos, y dos renders del mismo campo con
    encuadres distintos no se pueden comparar píxel a píxel.
    """
    if len(centers) == 0:
        return np.zeros((resolucion, resolucion), dtype=np.float64)

    base = vista.base
    proyectado = centers @ base.T          # (N, 3): [u, v, profundidad]
    uv = proyectado[:, :2]

    if encuadre is None:
        bajo = uv.min(axis=0) - margen_mm
        alto = uv.max(axis=0) + margen_mm
    else:
        bajo, alto = encuadre
    # Lado cuadrado: el mayor de los dos, para no deformar la escala entre u y v.
    lado = max(float(np.max(np.asarray(alto) - np.asarray(bajo))), 1e-9)
    mm_por_pixel = lado / resolucion

    # σ isótropa efectiva: la media geométrica de las tres escalas. Las semillas del
    # `cbct-agent` son isótropas por construcción, así que no se pierde nada; si el
    # optimizador las hace anisótropas, esto declara la aproximación en vez de fingir
    # que el splat es exacto.
    sigma_mm = np.cbrt(np.prod(np.asarray(scales, dtype=np.float64), axis=1))

    # **Masa**, no amplitud: la integral de τ sobre todo el plano, que es la integral de
    # σ en el volumen (σ · (2π)^{3/2} · s³). Depositar masa y dividir después por el área
    # del píxel es lo que hace la imagen consistente con la resolución.
    #
    # ⚠ Muestrear τ en el centro del píxel —lo evidente— NO vale aquí, y el error es
    # silencioso: con s = 0,15 mm y píxeles de ~0,7 mm, el centro del píxel cae a más de
    # 4σ de casi todas las gaussianas, así que la mayoría aportaría ~0 y la imagen saldría
    # de un aliasing, no del campo. Y recortar σ a medio píxel para taparlo infla la
    # amplitud cuando el píxel es grande: medido, τ pasaba de 34 a 256 px a **226** a
    # 128 px. Una integral de línea no puede depender del tamaño del detector.
    masa = (
        np.asarray(density, dtype=np.float64)
        * (2.0 * np.pi) ** 1.5
        * sigma_mm**3
    )

    # A píxeles. El centro del rectángulo del mundo cae en el centro de la imagen.
    centro_mundo = (bajo + alto) / 2.0
    pix = (uv - centro_mundo) / mm_por_pixel + resolucion / 2.0
    # El recorte a medio píxel se queda, pero ahora solo define la FORMA del reparto
    # (antialiasing de una gaussiana subpíxel), no cuánta masa se deposita.
    sigma_pix = np.maximum(sigma_mm / mm_por_pixel, 0.5)
    radio = np.ceil(SIGMAS_SOPORTE * sigma_pix).astype(np.int64)

    tau = np.zeros((resolucion, resolucion), dtype=np.float64)
    # Se agrupa por radio para vectorizar: todas las primitivas con el mismo soporte
    # comparten la misma ventana y se depositan de una vez con `np.add.at`.
    for r in np.unique(radio):
        sel = radio == r
        d = np.arange(-r, r + 1)
        dy, dx = np.meshgrid(d, d, indexing="ij")
        # (K, 1, 1) contra (1, 2r+1, 2r+1)
        cx = pix[sel, 0][:, None, None]
        cy = pix[sel, 1][:, None, None]
        ix = np.round(cx).astype(np.int64) + dx[None]
        iy = np.round(cy).astype(np.int64) + dy[None]
        rr2 = (ix - cx) ** 2 + (iy - cy) ** 2
        s2 = (sigma_pix[sel] ** 2)[:, None, None]
        perfil = np.exp(-rr2 / (2.0 * s2))
        # Normalizado a sumar 1 sobre el soporte: así el truncado a 3σ no pierde masa,
        # y una gaussiana subpíxel deposita lo mismo que una grande de igual masa.
        perfil /= perfil.sum(axis=(1, 2), keepdims=True)
        peso = masa[sel][:, None, None] * perfil
        dentro = (ix >= 0) & (ix < resolucion) & (iy >= 0) & (iy < resolucion)
        np.add.at(tau, (iy[dentro], ix[dentro]), peso[dentro])
    # Masa depositada → profundidad óptica media del píxel.
    return tau / (mm_por_pixel**2)


def beer_lambert(tau: np.ndarray) -> np.ndarray:
    """τ → intensidad en [0, 1]. Lo denso sale **oscuro**, como una radiografía."""
    return np.exp(-np.asarray(tau, dtype=np.float64))


def a_uint8(intensidad: np.ndarray) -> np.ndarray:
    """Cuantiza a 8 bits. Redondeo explícito: `astype` truncaría y sesgaría a oscuro."""
    return np.round(np.clip(intensidad, 0.0, 1.0) * 255.0).astype(np.uint8)


def psnr(a: np.ndarray, b: np.ndarray, *, maximo: float = 255.0) -> float:
    """PSNR en dB. `inf` si son idénticas — que es el resultado esperado aquí."""
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"No se puede comparar {x.shape} con {y.shape}.")
    mse = float(np.mean((x - y) ** 2))
    return float("inf") if mse == 0.0 else 10.0 * float(np.log10(maximo**2 / mse))


def ssim(a: np.ndarray, b: np.ndarray, *, maximo: float = 255.0, ventana: int = 7) -> float:
    """SSIM medio con ventana uniforme.

    Ventana uniforme (la del artículo original) y no gaussiana: es una media móvil, así que
    sale de dos sumas acumuladas y no arrastra una dependencia nueva al paquete solo para
    ponderar el vecindario. La diferencia entre las dos variantes es de decimales, y aquí
    el umbral está en 0,99 porque el ciclo no debería perder nada.
    """
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"No se puede comparar {x.shape} con {y.shape}.")
    if min(x.shape) < ventana:
        raise ValueError(f"Imagen {x.shape} más pequeña que la ventana {ventana}.")
    c1, c2 = (0.01 * maximo) ** 2, (0.03 * maximo) ** 2

    def media(z: np.ndarray) -> np.ndarray:
        acc = np.cumsum(np.cumsum(np.pad(z, ((1, 0), (1, 0))), axis=0), axis=1)
        k = ventana
        return (
            acc[k:, k:] - acc[:-k, k:] - acc[k:, :-k] + acc[:-k, :-k]
        ) / (k * k)

    mx, my = media(x), media(y)
    sxx = media(x * x) - mx * mx
    syy = media(y * y) - my * my
    sxy = media(x * y) - mx * my
    num = (2 * mx * my + c1) * (2 * sxy + c2)
    den = (mx**2 + my**2 + c1) * (sxx + syy + c2)
    return float(np.mean(num / den))


def escribe_png(destino: Path, imagen: np.ndarray) -> None:
    """PNG en escala de grises. Import perezoso, como en el `image-agent`."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - entorno sin la dependencia
        raise RuntimeError(
            "El render del campo necesita `pillow` (dependencia de `export-agents`)."
        ) from exc
    destino.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(imagen, dtype=np.uint8), mode="L").save(destino, format="PNG")


def lee_png(ruta: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(ruta) as img:
        return np.asarray(img.convert("L"), dtype=np.uint8)


def _desplazamiento(ply: Path) -> np.ndarray:
    """Lo que hay que restarle al PLY para llevarlo al marco del twin, en mm.

    El marco lo DECLARA la cabecera (`comment frame twin|cbct`), y por eso se lee de ahí:
    el twin va centrado y el CBCT lleva `centers + origin`, así que comparar sin saber
    cuál es mediría el desplazamiento en vez del ciclo.

    ⚠️ **Antes se restaba el centroide de la nube**, lo cual era exacto *mientras* `origin`
    fuese la media —el `cbct-agent` la escribe así—. El `gaussian-engine` ajusta elipsoides
    a la densidad y el centroide del campo ajustado ya no es cero: medido sobre un caso
    real, `[0,04, −1,84, 3,97]` mm. Aquella resta inocua se convirtió en una traslación de
    4,4 mm sobre una escena de 87 mm y el ciclo se desplomó a **14,1 dB** contra un
    presupuesto de 40, en silencio y con las imágenes de buen aspecto. Es el fallo que el
    propio comentario de este sitio anunciaba, y el motivo de que ahora se lea.

    Un marco que no se declara es un **error**, no un caso por defecto: adivinar es
    exactamente lo que hizo falta arreglar.
    """
    meta = metadatos_ply(ply)
    marco = meta.get("frame")
    if marco == "twin":
        # El twin ya va centrado. No hay nada que restar, y restar «casi nada» tampoco.
        return np.zeros(3)
    if marco == "cbct":
        if "origin_mm" not in meta:
            raise ValueError(
                f"{ply.name} declara `frame cbct` y no trae `origin_mm` en la cabecera: "
                "sin el desplazamiento no se puede volver al marco del twin, y estimarlo "
                "del centroide es lo que rompió esta comparación."
            )
        return np.asarray([float(v) for v in meta["origin_mm"].split()], dtype=np.float64)
    raise ValueError(
        f"{ply.name} no declara un marco conocido en su cabecera "
        f"(`comment frame …` dice {marco!r}): no se puede comparar contra el twin sin "
        "saber en qué sistema vienen las coordenadas."
    )


class RenderExportAgent(BaseExportAgent):
    """Materializa el campo del twin como PNG multivista, y mide el ciclo.

    `destination` es un **directorio**: un render multivista son varios ficheros. `path`
    apunta a él y `paths` lista lo escrito, en el orden de las vistas.
    """

    name = "render-export-agent"
    version = "0.1.1"

    def __init__(
        self,
        store: SurfaceStore,
        *,
        vistas: tuple[Vista, ...] = VISTAS_POR_DEFECTO,
        resolucion: int = RESOLUCION,
        verify: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not vistas:
            raise ValueError("Hacen falta vistas: un render multivista sin vistas no existe.")
        if resolucion < 16:
            raise ValueError(f"Resolución {resolucion} demasiado baja para medir SSIM.")
        self.store = store
        self.vistas = vistas
        self.resolucion = resolucion
        self.verify = verify

    def _export(  # type: ignore[override]
        self, snapshot: TwinSnapshot, destination: Path, *, ply: Path | None = None
    ) -> ExportOutput:
        arrays = self.store.load(snapshot.gaussian_field_ref)
        faltan = [k for k in _CLAVES_MINIMAS if k not in arrays]
        if faltan:
            raise ValueError(
                f"El artefacto {snapshot.gaussian_field_ref} no contiene "
                f"{', '.join(f'`{k}`' for k in faltan)}: no hay campo que renderizar."
            )
        centers = np.asarray(arrays["centers"], dtype=np.float64)
        scales = np.asarray(arrays["scales"], dtype=np.float64)
        density = np.asarray(arrays["density"], dtype=np.float64)

        # Un encuadre COMÚN para todas las vistas y para la comparación. Si cada imagen
        # eligiera el suyo, dos renders del mismo campo no serían comparables píxel a
        # píxel y el SSIM mediría el encuadre en vez del contenido.
        encuadre = self._encuadre(centers)

        destination.mkdir(parents=True, exist_ok=True)
        escritos: list[Path] = []
        propias: list[np.ndarray] = []
        for vista in self.vistas:
            img = a_uint8(
                beer_lambert(
                    profundidad_optica(
                        centers, scales, density,
                        vista=vista, resolucion=self.resolucion, encuadre=encuadre,
                    )
                )
            )
            ruta = destination / f"{snapshot.acquisition_id}_{vista.nombre}.png"
            escribe_png(ruta, img)
            escritos.append(ruta)
            propias.append(img)

        motivos = self._partial_reasons(snapshot)
        db = indice = None
        # El nombre del PLY se guarda aquí, dentro de la única rama donde existe, en vez
        # de volver a leerlo al componer el `detail`: allí `ply` sigue siendo opcional
        # para el tipo aunque `db` no sea None solo si lo hubo, y eso es un `Optional`
        # que el lector tiene que demostrarse a sí mismo cada vez.
        verificado_contra = ""
        if self.verify and ply is not None:
            db, indice = self._verify(ply, encuadre, propias)
            verificado_contra = ply.name
            if db < RENDER_PSNR_BUDGET_DB or indice < RENDER_SSIM_BUDGET:
                motivos.append(
                    f"el render del PLY exportado no reproduce el del twin (PSNR {db:.1f} dB, "
                    f"SSIM {indice:.4f}; mínimos {RENDER_PSNR_BUDGET_DB} dB y "
                    f"{RENDER_SSIM_BUDGET}): el ciclo pierde algo que debería ser exacto."
                )

        return self._outcome(
            ModalityStatus.OK,
            path=destination,
            paths=escritos,
            format="png",
            n_vertices=int(len(centers)),
            psnr_db=db,
            ssim=indice,
            hitl_reasons=motivos,
            detail=(
                f"{len(escritos)} vistas a {self.resolucion}×{self.resolucion} "
                f"({', '.join(v.nombre for v in self.vistas)})"
                + (
                    ""
                    if db is None
                    else f"; ciclo verificado contra {verificado_contra}: PSNR "
                    + ("exacto" if db == float("inf") else f"{db:.1f} dB")
                    + f", SSIM {indice:.4f}"
                )
            ),
        )

    def _encuadre(self, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Cubo del mundo que encuadra el campo, común a todas las vistas.

        Se toma el radio máximo desde el centroide, así que el encuadre es el mismo
        cualquiera que sea el ángulo: una vista no puede recortar lo que otra sí muestra.
        """
        if len(centers) == 0:
            return np.zeros(2), np.ones(2)
        centro = centers.mean(axis=0)
        radio = float(np.linalg.norm(centers - centro, axis=1).max()) + MARGEN_MM
        return np.array([-radio, -radio]), np.array([radio, radio])

    def _verify(
        self,
        ply: Path,
        encuadre: tuple[np.ndarray, np.ndarray],
        propias: list[np.ndarray],
    ) -> tuple[float, float]:
        """Renderiza el PLY exportado y compara con el render del twin.

        Es el ciclo completo: twin → PLY → render, contra twin → render. Cierra el bucle
        que el canal de malla cierra releyendo el STL, y por la misma razón: una
        estimación no vería un eje intercambiado, y esto sí.

        El marco del PLY se **lee** de su cabecera, no se deduce de los datos. Ver
        `_desplazamiento`.
        """
        leido = lee_ply(ply)
        centers = np.column_stack([leido["x"], leido["y"], leido["z"]]).astype(np.float64)
        scales = np.column_stack(
            [leido["scale_0"], leido["scale_1"], leido["scale_2"]]
        ).astype(np.float64)
        density = np.asarray(leido["density"], dtype=np.float64)
        centers = centers - _desplazamiento(ply)

        peores = (float("inf"), 1.0)
        for vista, propia in zip(self.vistas, propias, strict=True):
            otra = a_uint8(
                beer_lambert(
                    profundidad_optica(
                        centers, scales, density,
                        vista=vista, resolucion=self.resolucion, encuadre=encuadre,
                    )
                )
            )
            peores = (
                min(peores[0], psnr(propia, otra)),
                min(peores[1], ssim(propia, otra)),
            )
        return peores
