"""Corpus de evaluación del `report-agent`: informes con su **verdad anotada**.

`edge_cases.py` cataloga lo que llega roto; esto cataloga lo que llega *entero* y
mide **cuánto de ello acaba en el contrato**. Es la pieza que faltaba para poder
afirmar el ">95% de fiabilidad de los agentes de ingesta" del `README.md`: hasta
ahora había un informe real en `data/` y tests unitarios de una línea, que
comprueban que un patrón encaja pero no dicen qué fracción de un informe se pierde.

**Por qué la verdad y el suelo van separados.** Cada caso declara dos cosas:

- `ph` / `clinicos`: lo que el informe **dice**, leído por una persona. Es la
  referencia, y no cambia porque cambie el extractor.
- `rules_ph` / `rules_clinicos`: lo que el backend determinista saca **hoy**.
  `None` significa "lo mismo que la verdad" — el caso lo cubre entero.

Mezclarlas convertiría el corpus en un test de regresión disfrazado de métrica: si
la referencia fuese "lo que el regex saca", el regex tendría 100% por definición y
no habría nada contra lo que comparar un LLM. Separadas, la diferencia entre las
dos columnas **es** el trabajo pendiente, y está escrita caso por caso en `limite`.

**Los tres tipos de fallo no valen lo mismo.** Un valor que no se extrae (`faltan`)
es un hueco visible en el twin. Un valor extraído que el informe no dice (`sobran`)
o que dice distinto (`errores`) es el fallo caro del ADR 003: plausible, silencioso
y ya dentro del contrato. Por eso `puntua()` los cuenta por separado en vez de dar
un porcentaje único, y por eso hay casos aquí cuyo único propósito es comprobar que
**nadie inventa** — negaciones, antecedentes, vocabulario fuera de la lista.

Todo es sintético: ni un byte procede de un paciente (ADR de anonimización §1,
"sintético primero"). Los formatos imitan informes reales — tabulado de pH, informe
de CBCT con IA por pieza, tabla de índices de oclusión — pero los valores no.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

from core_schemas import Hallazgo

# Familias del corpus. No es decoración: sirven para leer el resultado por bloques
# ("el determinista cubre el tabulado y se cae en la prosa") en vez de por un total
# que promedia cosas que no se parecen.
TABULADO = "tabulado"
"""Formato de tabla o de línea por diente: el terreno del backend determinista."""
PROSA = "prosa"
"""Texto corrido, negaciones, sinónimos: donde la entrada no tiene esquema."""
ABSTENCION = "abstencion"
"""Casos cuya respuesta correcta es **no extraer nada**."""


@dataclass(frozen=True)
class ReportCase:
    """Un informe sintético, lo que dice de verdad y lo que el regex saca de él."""

    name: str
    familia: str
    why: str
    """Qué mide este caso. Si no se puede escribir, el caso no aporta."""
    text: str

    # --- la verdad: lo que una persona lee en el informe --------------------- #
    ph: dict[str, float] = field(default_factory=dict)
    clinicos: dict[str, dict[str, Any]] = field(default_factory=dict)
    """`FDI → {n_raices, n_conductos, hallazgos}`. Las claves ausentes son `None`/`[]`."""
    medidas: int = 0
    """Cuántos índices con rango de referencia declara el informe."""

    # --- el suelo: lo que el backend `rules` saca hoy ------------------------ #
    rules_ph: dict[str, float] | None = None
    rules_clinicos: dict[str, dict[str, Any]] | None = None
    """`None` = idéntico a la verdad: el determinista cubre el caso entero."""
    limite: str = ""
    """Por qué el determinista no llega, cuando no llega. Vacío si llega."""

    def esperado_rules(self) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
        """La expectativa del backend determinista, resolviendo el atajo `None`."""
        ph = self.ph if self.rules_ph is None else self.rules_ph
        clinicos = self.clinicos if self.rules_clinicos is None else self.rules_clinicos
        return ph, clinicos

    def write(self, destino: Path) -> Path:
        """Escribe el informe como `.txt` — el mismo texto que sale de un PDF."""
        destino = Path(destino)
        destino.mkdir(parents=True, exist_ok=True)
        ruta = destino / f"{self.name}.txt"
        ruta.write_text(self.text, encoding="utf-8")
        return ruta


# --------------------------------------------------------------------------- #
# Puntuación
# --------------------------------------------------------------------------- #
class Puntuacion(NamedTuple):
    """Los cuatro desenlaces de una extracción, contados por separado.

    `aciertos` + `faltan` = lo que el informe decía. `errores` y `sobran` son los
    que llegan al twin siendo falsos, y por eso no se compensan con aciertos en
    ninguna media: un informe con 20 aciertos y 1 valor inventado no es un 95%.
    """

    aciertos: int
    errores: int
    """Extraído, pero con un valor distinto del que dice el informe."""
    faltan: int
    """El informe lo dice y no se extrajo."""
    sobran: int
    """Extraído sin que el informe lo diga (falso positivo)."""

    @property
    def total_verdad(self) -> int:
        return self.aciertos + self.errores + self.faltan

    @property
    def cobertura(self) -> float:
        """Fracción de lo que el informe dice que llega correcta al contrato."""
        return self.aciertos / self.total_verdad if self.total_verdad else 1.0

    def __add__(self, otra: object) -> Puntuacion:  # type: ignore[override]
        if not isinstance(otra, Puntuacion):
            return NotImplemented
        return Puntuacion(*(a + b for a, b in zip(self, otra, strict=True)))


def normaliza(clinicos: dict[str, dict[str, Any]]) -> dict[tuple[str, str], Any]:
    """`{FDI: {campo: valor}}` → `{(FDI, campo): valor}`, saltando lo no declarado.

    Aplanar permite puntuar los tres campos clínicos con el mismo contador que el
    pH: cada `(diente, campo)` es un valor independiente que puede acertarse,
    fallarse, perderse o inventarse. Los `hallazgos` se comparan como **conjunto**
    porque el orden lo fija el extractor, no el informe.
    """
    plano: dict[tuple[str, str], Any] = {}
    for code, campos in clinicos.items():
        for nombre in ("n_raices", "n_conductos"):
            valor = campos.get(nombre)
            if valor is not None:
                plano[(code, nombre)] = valor
        hallazgos = campos.get("hallazgos") or []
        if hallazgos:
            plano[(code, "hallazgos")] = frozenset(hallazgos)
    return plano


def puntua(extraido: dict[Any, Any], verdad: dict[Any, Any]) -> Puntuacion:
    """Compara una extracción con la verdad anotada, clave a clave."""
    aciertos = errores = 0
    for clave, valor in extraido.items():
        if clave not in verdad:
            continue
        if valor == verdad[clave]:
            aciertos += 1
        else:
            errores += 1
    return Puntuacion(
        aciertos=aciertos,
        errores=errores,
        faltan=len(set(verdad) - set(extraido)),
        sobran=len(set(extraido) - set(verdad)),
    )


def puntua_clinicos(
    extraido: dict[str, dict[str, Any]], verdad: dict[str, dict[str, Any]]
) -> Puntuacion:
    """Como `puntua`, sobre los campos clínicos aplanados a `(diente, campo)`."""
    return puntua(normaliza(extraido), normaliza(verdad))


# --------------------------------------------------------------------------- #
# Los informes
# --------------------------------------------------------------------------- #
_CABECERA = (
    "INFORME ODONTOLÓGICO (CASO SINTÉTICO)\n"
    "=====================================\n"
    "Fecha: 2026-03-14\n"
    "Paciente: caso sintético para evaluación de la ingesta.\n\n"
)


def _informe(cuerpo: str, *, cabecera: bool = True) -> str:
    return (_CABECERA if cabecera else "") + cuerpo.strip() + "\n"


CASES: tuple[ReportCase, ...] = (
    # --- Tabulado: el terreno del determinista ----------------------------- #
    ReportCase(
        name="ph-tabulado",
        familia=TABULADO,
        why="El formato canónico: un diente y su pH por línea. Si esto falla, no hay suelo.",
        text=_informe(
            "Hallazgos por diente (numeración ISO-FDI):\n"
            "  - Diente 16: pH 5.2 — riesgo de desmineralización.\n"
            "  - Diente 21: pH 6.8 — dentro de rango.\n"
            "  - Diente 26: pH 5.9 — dentro de rango.\n"
            "  - Diente 36: pH 7.1 — dentro de rango.\n"
        ),
        ph={"16": 5.2, "21": 6.8, "26": 5.9, "36": 7.1},
    ),
    ReportCase(
        name="ph-coma-decimal",
        familia=TABULADO,
        why="Un informe en español escribe 5,4. Es la mitad del mundo, no un caso raro.",
        text=_informe(
            "  - Diente 17 — pH 5,4.\n"
            "  - Diente 27 — pH 6,15.\n"
            "  - Diente 37 — pH 7,0.\n"
        ),
        ph={"17": 5.4, "27": 6.15, "37": 7.0},
    ),
    ReportCase(
        name="ph-notacion-punto-y-orden-invertido",
        familia=TABULADO,
        why="FDI también se escribe 1.6, y el pH puede ir antes que el diente.",
        text=_informe(
            "  - El diente 1.6 presenta pH 6.2.\n"
            "  - pH 5.8 en el diente 3.6.\n"
            "  - Diente 2.4 pH= 6.4\n"
        ),
        ph={"16": 6.2, "36": 5.8, "24": 6.4},
        rules_ph={"16": 6.2, "24": 6.4},
        limite=(
            "«3.6.» al final de la frase no se ve: el lookahead `(?![\\d.,])` de "
            "`_TOOTH_TOKEN_RE` rechaza el punto de cierre. Con «1.6 presenta» sí encaja, "
            "así que el mismo diente se lee o no según dónde caiga en la oración."
        ),
    ),
    ReportCase(
        name="cbct-ia-por-pieza",
        familia=TABULADO,
        why=(
            "El formato del informe de CBCT con IA: anatomía radicular y hallazgos por "
            "pieza, sin pH. Es el que motivó `extract_hallazgos_by_rules`."
        ),
        text=_informe(
            "Análisis automático por pieza:\n"
            "Diente 16 Diente, 3 raíces, 4 canales, Signos de caries (Dentina).\n"
            "Diente 17 Diente, 3 raíces, 3 conductos, Restauración presente.\n"
            "Diente 21 Diente, 1 raíz, 1 conducto.\n"
            "Diente 36 Diente, 2 raíces, 3 canales, Cálculo pulpar.\n"
            "Diente 38 Ausente.\n"
        ),
        clinicos={
            "16": {"n_raices": 3, "n_conductos": 4, "hallazgos": [Hallazgo.CARIES]},
            "17": {"n_raices": 3, "n_conductos": 3, "hallazgos": [Hallazgo.RESTAURACION]},
            "21": {"n_raices": 1, "n_conductos": 1, "hallazgos": []},
            "36": {"n_raices": 2, "n_conductos": 3, "hallazgos": [Hallazgo.CALCULO_PULPAR]},
            "38": {"hallazgos": [Hallazgo.AUSENTE]},
        },
    ),
    ReportCase(
        name="cbct-ia-descripcion-partida",
        familia=TABULADO,
        why=(
            "El PDF corta la descripción a mitad de línea. El extractor parte por bloques "
            "`Diente NN` justamente para no perder la segunda mitad del hallazgo."
        ),
        text=_informe(
            "Diente 46 Diente, 2 raíces, 3 canales, Signos de caries (Dentina,\n"
            "oclusal), Pérdida de hueso periodontal leve en la furca.\n"
            "Diente 47 Diente, 2 raíces, 3 canales, Aparato de ortodoncia\n"
            "(banda) presente.\n"
        ),
        clinicos={
            "46": {
                "n_raices": 2,
                "n_conductos": 3,
                "hallazgos": [Hallazgo.CARIES, Hallazgo.PERDIDA_OSEA_PERIODONTAL],
            },
            "47": {
                "n_raices": 2,
                "n_conductos": 3,
                "hallazgos": [Hallazgo.APARATO_ORTODONCICO],
            },
        },
    ),
    ReportCase(
        name="indices-de-oclusion",
        familia=TABULADO,
        why=(
            "Tabla de índices con su rango impreso al lado: se captura sin interpretar "
            "(`Medida`). Dos de los ocho caen fuera de su propio rango normal."
        ),
        text=_informe(
            "ANÁLISIS DE OCLUSIÓN\n"
            "POC TA      88.09%  I        83≤(%)≤100\n"
            "POC ECM     81.20%  D        83≤(%)≤100\n"
            "TORS        89.34%          90≤(%)≤100\n"
            "ASIM         4.58%          -10≤(%)≤10\n"
        ),
        medidas=4,
    ),
    ReportCase(
        name="informe-mixto",
        familia=TABULADO,
        why=(
            "pH, anatomía por pieza e índices en el mismo documento. Los tres extractores "
            "corren sobre el mismo texto y no deben pisarse."
        ),
        text=_informe(
            "Mediciones de superficie:\n"
            "  - Diente 16: pH 5.1\n"
            "  - Diente 26: pH 6.9\n"
            "\nAnálisis automático por pieza:\n"
            "Diente 16 Diente, 3 raíces, 4 canales, Signos de caries (Esmalte).\n"
            "Diente 26 Diente, 3 raíces, 3 conductos.\n"
            "\nÍNDICES\n"
            "ASIM         3.10%          -10≤(%)≤10\n"
        ),
        ph={"16": 5.1, "26": 6.9},
        clinicos={
            "16": {"n_raices": 3, "n_conductos": 4, "hallazgos": [Hallazgo.CARIES]},
            "26": {"n_raices": 3, "n_conductos": 3, "hallazgos": []},
        },
        medidas=1,
    ),
    ReportCase(
        name="deciduos",
        familia=TABULADO,
        why="La dentición temporal (5x-8x) es FDI válido. Un pediátrico no es un caso raro.",
        text=_informe(
            "  - Diente 54: pH 5.6\n"
            "  - Diente 64: pH 6.3\n"
            "Diente 75 Diente, 2 raíces, 2 conductos, Signos de caries.\n"
        ),
        ph={"54": 5.6, "64": 6.3},
        clinicos={"75": {"n_raices": 2, "n_conductos": 2, "hallazgos": [Hallazgo.CARIES]}},
    ),

    # --- Prosa: donde la entrada no tiene esquema ---------------------------- #
    ReportCase(
        name="prosa-ph-preposicion",
        familia=PROSA,
        why=(
            "«pH de 5,2» en vez de «pH 5,2». El patrón admite `:` y `=` pero no una "
            "preposición, y un informe dictado las lleva todas."
        ),
        text=_informe(
            "El diente 16 muestra un pH de 5,2, compatible con desmineralización activa.\n"
            "En el diente 26 se midió un pH cercano a 6,8, dentro de la normalidad.\n"
        ),
        ph={"16": 5.2, "26": 6.8},
        rules_ph={},
        limite="`_PH_RE` exige el número pegado a «pH» (o tras `:`/`=`); «de» rompe el patrón.",
    ),
    ReportCase(
        name="prosa-negacion",
        familia=PROSA,
        why=(
            "El fallo caro: el informe **niega** el hallazgo y el patrón lo afirma. "
            "No es un hueco en el twin, es un dato falso dentro del contrato."
        ),
        text=_informe(
            "Diente 26 Diente, 3 raíces, 3 conductos, sin signos de caries ni de "
            "restauración previa.\n"
        ),
        clinicos={"26": {"n_raices": 3, "n_conductos": 3, "hallazgos": []}},
        rules_clinicos={
            "26": {
                "n_raices": 3,
                "n_conductos": 3,
                "hallazgos": [Hallazgo.CARIES, Hallazgo.RESTAURACION],
            }
        },
        limite="Un regex de presencia no sabe leer «sin»: cuenta la palabra, no la frase.",
    ),
    ReportCase(
        name="prosa-antecedente",
        familia=PROSA,
        why=(
            "«Antecedente de caries, hoy restaurado»: el hallazgo actual es uno y el "
            "histórico otro. Confundirlos falsea la serie temporal del diente."
        ),
        text=_informe(
            "Diente 36 Diente, 2 raíces, 3 conductos. Antecedente de caries tratada en "
            "2024; actualmente restaurado, sin lesión activa.\n"
        ),
        clinicos={"36": {"n_raices": 2, "n_conductos": 3, "hallazgos": [Hallazgo.RESTAURACION]}},
        rules_clinicos={
            "36": {"n_raices": 2, "n_conductos": 3, "hallazgos": [Hallazgo.CARIES]}
        },
        limite=(
            "Se equivoca dos veces en la misma línea: registra la caries histórica (el "
            "patrón no tiene noción de tiempo verbal) y pierde la restauración actual "
            "(«restaurado» no encaja con `\\brestauraci[óo]n\\b`). El twin acaba diciendo "
            "lo contrario de lo que dice el informe."
        ),
    ),
    ReportCase(
        name="prosa-rango-de-dientes",
        familia=PROSA,
        why="«Dientes 16-18» son tres piezas. Es la forma normal de abreviar en un informe.",
        text=_informe(
            "Se observa restauración de composite en los dientes 16, 17 y 18.\n"
            "Pérdida de hueso periodontal en el sector 34-36.\n"
        ),
        clinicos={
            "16": {"hallazgos": [Hallazgo.RESTAURACION]},
            "17": {"hallazgos": [Hallazgo.RESTAURACION]},
            "18": {"hallazgos": [Hallazgo.RESTAURACION]},
            "34": {"hallazgos": [Hallazgo.PERDIDA_OSEA_PERIODONTAL]},
            "35": {"hallazgos": [Hallazgo.PERDIDA_OSEA_PERIODONTAL]},
            "36": {"hallazgos": [Hallazgo.PERDIDA_OSEA_PERIODONTAL]},
        },
        rules_clinicos={},
        limite="Sin el marcador «Diente NN» no hay bloque; y expandir un rango no es un patrón.",
    ),
    ReportCase(
        name="prosa-numeros-en-letra",
        familia=PROSA,
        why="«tres raíces» es tan común como «3 raíces» en un informe redactado.",
        text=_informe(
            "Diente 16 Diente, con tres raíces y cuatro conductos, sin hallazgos "
            "reseñables.\n"
        ),
        clinicos={"16": {"n_raices": 3, "n_conductos": 4, "hallazgos": []}},
        rules_clinicos={},
        limite=(
            "`_RAICES_RE`/`_CONDUCTOS_RE` exigen dígitos; el cardinal en letra no existe "
            "para ellos."
        ),
    ),
    ReportCase(
        name="prosa-sinonimo",
        familia=PROSA,
        why=(
            "«Obturación» y «empaste» son «restauración». El vocabulario controlado existe "
            "precisamente para absorber esa variación — pero alguien tiene que hacerlo."
        ),
        text=_informe(
            "Diente 46 Diente, 2 raíces, 3 conductos, obturación de amalgama en oclusal.\n"
            "Diente 47 Diente, 2 raíces, 3 conductos, empaste de composite.\n"
        ),
        clinicos={
            "46": {"n_raices": 2, "n_conductos": 3, "hallazgos": [Hallazgo.RESTAURACION]},
            "47": {"n_raices": 2, "n_conductos": 3, "hallazgos": [Hallazgo.RESTAURACION]},
        },
        rules_clinicos={
            "46": {"n_raices": 2, "n_conductos": 3, "hallazgos": []},
            "47": {"n_raices": 2, "n_conductos": 3, "hallazgos": []},
        },
        limite="La lista de patrones tiene un término por hallazgo; los sinónimos no están.",
    ),
    ReportCase(
        name="prosa-hallazgo-fuera-de-vocabulario",
        familia=PROSA,
        why=(
            "Una fractura radicular no está en los seis términos del vocabulario. Lo "
            "correcto es **no registrarla**, no aproximarla al término más cercano."
        ),
        text=_informe(
            "Diente 21 Diente, 1 raíz, 1 conducto, fractura radicular vertical en tercio "
            "medio.\n"
        ),
        clinicos={"21": {"n_raices": 1, "n_conductos": 1, "hallazgos": []}},
        limite="",
    ),

    # --- Abstención: la respuesta correcta es no extraer nada ---------------- #
    ReportCase(
        name="ph-medio-de-arcada",
        familia=ABSTENCION,
        why="Un pH sin diente no es regional. Colgarlo de una pieza sería inventar el sitio.",
        text=_informe("pH medio de la arcada superior: 6.4\npH medio de la inferior: 6.6\n"),
    ),
    ReportCase(
        name="ph-de-contexto-ajeno",
        familia=ABSTENCION,
        why="No todo «pH» de un informe es de un diente. Este es del agua de la unidad.",
        text=_informe(
            "Control de la unidad dental: pH del agua de irrigación 7.2 (conforme).\n"
        ),
    ),
    ReportCase(
        name="fdi-inexistente",
        familia=ABSTENCION,
        why="El diente 19 no existe. Rechazarlo está bien; perderlo sin rastro, no.",
        text=_informe("  - Diente 19: pH 6.0\n  - Diente 16: pH 5.5\n"),
        ph={"16": 5.5},
    ),
    ReportCase(
        name="ph-fuera-de-rango",
        familia=ABSTENCION,
        why="«pH 74» es un 7.4 mal tecleado o mal OCReado. Debe caer, y debe constar.",
        text=_informe("  - Diente 47: pH 74\n  - Diente 46: pH 6.1\n"),
        ph={"46": 6.1},
    ),
    ReportCase(
        name="diente-repetido",
        familia=ABSTENCION,
        why=(
            "Dos valores para el mismo diente: el informe se contradice. Quedarse con el "
            "primero en silencio elegiría por el clínico."
        ),
        text=_informe("  - Diente 16: pH 5.2\n  - Diente 16: pH 6.4\n"),
        ph={"16": 5.2},
    ),
    ReportCase(
        name="notacion-universal",
        familia=ABSTENCION,
        why=(
            "«Tooth #3» es notación Universal (= 16 en FDI). Convertirla sin que el informe "
            "declare el sistema es adivinar: el prompt del extractor lo prohíbe."
        ),
        text=_informe("Tooth #3: pH 5.4\nTooth #14: pH 6.1\n"),
        rules_ph={"14": 6.1},
        limite=(
            "El peor caso medido del corpus. `#14` en Universal es el 26 en FDI, pero «14» "
            "**también** es un FDI válido, así que el valor entra en el twin colgado del "
            "diente equivocado, con confianza 0,9 y sin descarte que lo delate: plausible, "
            "silencioso e irreversible. `#3` se salva solo porque tiene un dígito."
        ),
    ),
    ReportCase(
        name="sin-hallazgos-regionales",
        familia=ABSTENCION,
        why=(
            "Un informe legible del que no se extrae nada regional. No es un éxito vacío: "
            "la confianza baja a 0 y el gate humano lo para."
        ),
        text=_informe(
            "Exploración dentro de la normalidad. No se observan lesiones cariosas ni "
            "signos de enfermedad periodontal. Se recomienda revisión en 12 meses.\n"
        ),
    ),
)


def by_familia(familia: str) -> list[ReportCase]:
    """Los casos de una familia (`TABULADO`, `PROSA`, `ABSTENCION`)."""
    return [caso for caso in CASES if caso.familia == familia]


def write_all(destino: Path) -> list[Path]:
    """Materializa el corpus como ficheros `.txt` en `destino`."""
    return [caso.write(destino) for caso in CASES]


def total_verdad(casos: Iterable[ReportCase] = CASES) -> tuple[int, int]:
    """`(valores de pH, valores clínicos)` que el corpus declara. El denominador."""
    casos = list(casos)
    return (
        sum(len(caso.ph) for caso in casos),
        sum(len(normaliza(caso.clinicos)) for caso in casos),
    )
