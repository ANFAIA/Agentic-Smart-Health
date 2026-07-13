# ADR 001: Estructura de modelos del Digital Twin dental

**Fecha:** 2026-07-13
**Estado:** Aceptado
**Decisor:** Equipo de desarrollo
**Contexto:** Packages/core-schemas/models.py

---

## Contexto

El proyecto Agentic Smart Health necesita definir los modelos de datos centrales para el Digital Twin dental. Estos modelos deben soportar:

1. **Tres tipos de atributos clínicos** con soportes geométricos distintos (volumétrico, superficial, regional)
2. **Series temporales** para evaluar evolución clínica
3. **Trazabilidad** completa (RGPD/HIPAA)
4. **Integración con 3DGS** (Residual Gaussian Splatting)
5. **Rendimiento** con campos de millones de gaussianas

## Decisión

### 1. Tres soportes geométricos

Los atributos clínicos NO comparten soporte:

| Atributo | Soporte | Ubicación | Ejemplo |
|---|---|---|---|
| Densidad radiológica (σ) | Volumétrico | `GaussianPrimitive` | Por gaussiana |
| Color de superficie | Superficial | `GaussianPrimitive` | Solo cáscara 2D |
| pH y otros | Regional | `ClinicalAttributes` | Una zona/diente |

### 2. Campo gaussiano referenciado, no embebido

```python
gaussian_field_ref: str  # Hash/URI al almacén de 3dgs-engine
```

**Razón:** millones de objetos Pydantic serían ineficientes. Se referencia por hash, y `3dgs-engine` maneja los tensores.

### 3. Enfoque snapshot-céntrico

Cada `TwinSnapshot` es autocontenido:
- Tiene su propio `gaussian_field_ref`
- Incluye sus modalidades
- Contiene sus observaciones regionales

**Razón:** reversibilidad. Para regenerar datos de una fecha específica, basta con ese snapshot.

### 4. `PatientDigitalTwin` como secuencia temporal

No es un modelo 3D continuo, sino una línea temporal de estados:

```
snapshot_1 (2024-01) → snapshot_2 (2024-06) → snapshot_3 (2025-01)
```

Esto soporta directamente la Tarea 3: evaluar evolución clínica.

### 5. Trazabilidad granular

`Provenance` se adjunta a cada `RegionalObservation`, no al snapshot completo. Esto permite preguntar: "¿de dónde viene este pH del diente 16 del snapshot 2?"

### 6. Validación FDI

`FDICode` valida patrones ISO-FDI:
- Dientes permanentes: 11-48
- Dientes temporales: 51-85

## Alternativas consideradas

### Alternativa A: Modelo monolítico
Un solo modelo con todos los atributos mezclados.
- **Descartado:** no respeta la distinción de soportes geométricos.

### Alternativa B: Referencia por índices
Usar índices numéricos en vez de códigos FDI.
- **Descartado:** pierde semántica clínica y validación.

### Alternativa C: Campo gaussiano embebido
Almacenar todas las gaussianas en el modelo Pydantic.
- **Descartado:** ineficiente con millones de puntos.

## Consecuencias

### Positivas
- Claridad en la separación de responsabilidades
- Soporte nativo para series temporales
- Trazabilidad completa
- Rendimiento con campos masivos

### Negativas
- Complejidad inicial mayor
- Necesidad de sincronización entre `3dgs-engine` y `core-schemas`
- El código FDI requiere validación adicional

## Notas de implementación

- `GaussianPrimitive` documenta la unidad canónica para serialización de conjuntos pequeños
- Los arrays masivos viven como tensores en `3dgs-engine`
- Cada modelo usa `ConfigDict(extra="forbid")` para rechazar campos no previstos

## Referencias

- Lin et al., "Residual Gaussian Splatting for Ultra Sparse-View CBCT Reconstruction", arXiv:2604.27552v1 (2026)
- `docs/research/3dgs-clinical-extension.md`
