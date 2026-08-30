# Medir el tono de cada corona sobre fotografías clínicas

> **Estado (2026-08-27):** resultado **positivo y medido**. El artefacto que hacía
> incomparables los tonos entre piezas está identificado, corregido y declarado. Lo que
> sigue **sin** poder afirmarse es el nivel absoluto — para eso hace falta una referencia
> gris en el encuadre, y una serie clínica no la lleva.

Complementa `frontera-encia-desde-foto.md`: allí se mide *dónde* acaba la corona; aquí, de
qué color es la parte que sí es corona.

## 1 · El síntoma

Abriendo el contenedor de un caso real en el visor, la arcada salía con **un diente
naranja, otro blanco y otro rosa**. Es una boca sola: eso no puede ser.

Los tonos declarados en `clinical/observations.json`, por tercio medio:

| FDI | 16 | 15 | 14 | 13 | 12 | 11 | 21 | 22 | 23 | 24 | 25 | 26 | 27 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `L*` | 70,1 | 73,5 | 77,1 | 68,6 | 78,0 | 80,8 | 81,8 | 76,5 | 76,2 | 72,0 | 65,2 | 65,3 | 59,0 |
| `a*` | 10,1 | 8,5 | 7,2 | **17,5** | 7,4 | 2,9 | 2,9 | 6,4 | 6,1 | 6,3 | 7,9 | 8,9 | 12,7 |

`L*` recorre **22,7 puntos** y correlaciona **−0,86** con la distancia a la línea media.
Eso no describe unos dientes: describe una fuente de luz.

## 2 · La medida que podía refutarlo

Si la caída es del flash, la **encía contigua a cada corona** tiene que oscurecerse igual;
si el degradado fuera de los dientes, la encía saldría plana. Se muestreó una banda de 40 px
por encima del cuello de cada corona, excluyendo píxeles de cualquier otra corona:

| foto | `r(L*` diente`, L*` encía`)` | recorrido diente | recorrido encía |
|---|---|---|---|
| `c50eab52…` | **0,96** | 11,9 | 18,6 |
| `baeb31fa…` | **0,99** | 20,8 | 24,8 |

La encía se oscurece con el diente, y **recorre más que él**. Ese segundo número es el que
decide cómo hay que corregir.

## 3 · Por qué no se divide por la sonda

La tentación es normalizar cada corona dividiendo por su encía. Eso asume que la encía
recibe la misma luz que el diente y nada más, y es falso: la encía está retraída respecto al
plano vestibular y le entra sombra propia. Medido — dividir **invierte** el degradado:

| | observado | dividiendo | regresando |
|---|---|---|---|
| corona más oscura | `L*` 58,6 | `L*` **83,4** | `L*` 72,6 |
| corona más clara | `L*` 80,2 | `L*` 74,3 | `L*` 74,3 |

El molar del fondo acababa siendo el diente más claro de la boca.

Lo que se hace es descontar del diente **solo la parte que la sonda explica**: una
regresión robusta (Theil–Sen) de `log(diente)` contra `log(encía/referencia)` por canal, y
se resta `β · log(encía/referencia)`. `β = 1` sería dividir; `β = 0`, no tocar nada. Sobre
este caso sale **0,65 / 0,58 / 0,68**, estimado y no elegido.

## 4 · Resultado

| | antes | después |
|---|---|---|
| recorrido `L*` | 22,7 | **5,6** |
| recorrido `a*` | 14,6 | **5,9** |
| `r(L*`, posición en el arco`)` | −0,86 | **−0,35** |
| FDI 13 (el «rosa») | `a*` 17,5 | **`a*` 10,8** |

El 13 tenía además encía dentro de la muestra: la mediana aguanta «algún píxel colado en el
borde», no que la máscara se desborde por el cuello. Se rechazan los píxeles más próximos a
la encía que al diente **en `a*b*`** — separar por `L*` confundiría sombra con mucosa, que
es justo el artefacto que se acaba de corregir.

**Un control que no se le impuso:** tras corregir, el canino (13) es la pieza más cromática
de la arcada, que es lo que dice la anatomía. Antes lo era por contaminación; ahora lo es
por poco y por la razón correcta.

## 5 · Cuándo una sonda no es encía

La banda se toma por geometría —lo que queda por encima del cuello y no es corona de
nadie—, y ahí arriba no siempre hay mucosa: puede haber un separador, un labio, el espejo
intraoral o el fondo negro de la boca. Corregir una corona contra un separador blanco la
deja con un color inventado y con toda la pinta de estar medido.

El primer criterio fue de **dispersión**: rechazar las sondas que se apartaran más de 3 MAD
de las demás. Sobre este caso tiró exactamente las dos buenas — la encía del 13 y la del 22
son papila, más saturadas que la encía adherida (`a*` 36,1 y 31,2 frente a 25-30). Esas dos
piezas se quedaban con el artefacto puesto **y** con mucosa dentro de la muestra: el 13
volvía a declararse con `a*` 17,5, que es el diente «rosa» del que venía todo esto. Peor que
no filtrar nada.

El criterio bueno es **direccional**: la mucosa es más roja que el esmalte con cualquier
luz. Medido, las dieciséis sondas están entre **12,5 y 22,2** puntos de `a*` por encima de
su corona; un plástico blanco, un espejo o el fondo estarían en cero o por debajo. El umbral
se pone en 5, con la separación entera de margen.

## 6 · Lo que sigue sin poder afirmarse

El **nivel absoluto**. La corrección pone las piezas en la misma escala, no en una escala
conocida: sin una referencia gris en el encuadre no hay forma de decir «esto es un A2». El
contenedor lo declara pieza a pieza en `correccion_iluminacion`, y la nota de cada color se
compone de ese campo — **ausente no es cero**, es «no comparable con las demás».
