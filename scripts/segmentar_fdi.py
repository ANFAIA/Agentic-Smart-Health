#!/usr/bin/env python
"""segmentar_fdi.py — etiqueta cada diente de una arcada con su código FDI.

    ~/.venvs/dental-gpu/bin/python scripts/segmentar_fdi.py \
        --escaneo ARCADA.stl --arcada lower --salida DIR

⚠️ **No se ejecuta con `uv run`.** Necesita torch y `torch_geometric`, que viven en el
entorno dedicado `~/.venvs/dental-gpu` y no en el del proyecto. Esa separación es
deliberada: `packages/tooth-aggregation` está escrito **sin depender de torch** para que
la agregación punto→diente se instale y se testee en el workspace normal, y el *forward*
del modelo sea cosa del llamante. Este script es ese llamante.

Reutiliza `data/processed/a3-checkpoints/normales-ce-baseline.pt` SIN reentrenar, y no
el otro checkpoint de A3: está medido que `normales-ce-boundary-centroid` está
**degenerado en mandibular** (0,193 de acierto; mapea sistemáticamente códigos
inferiores a superiores).

## La pose, que es lo que hacía falta y no estaba

El modelo se entrenó sobre Teeth3DS+ con `NormalizeScale`, que centra y escala pero
**no rota**. Así que depende de la orientación absoluta con la que cada escáner escribe
su fichero, y dos escáneres no tienen por qué coincidir. Medido: el eje oclusal de
Teeth3DS+ va en **+z** y el de `histora` en **+y**, coseno **0,004** — perpendiculares.
Sin canonizar, el modelo etiquetaba una mandíbula con códigos de maxilar.

Canonizando la pose con `fusion_agents.marco` el orden anatómico sale perfecto
(ρ ±0,99 entre código FDI y posición a lo largo del arco).

⚠️ **La canonización no es gratis y falla cuando el marco es dudoso.** En la
autocomprobación sobre Teeth3DS+, una malla con razón de orientación 0,61 se hundió de
0,808 a 0,197 de acierto; las de razón 0,40-0,43 quedaron intactas. Por eso el script
imprime la razón y avisa por encima de 0,6: no es un adorno.

## Decodificación restringida

Se enmascaran los códigos de la arcada contraria antes de agregar. No es hacer trampa:
la arcada no es una hipótesis que el modelo deba resolver —viene en el fichero y en la
anatomía—, y dejarle emitirlos es darle libertad para fallar en algo que no estaba en
duda. Medido sobre `histora`: 23 instancias → 16, y desaparecen los cuadrantes
imposibles.

## Lo que NO valida esto

No hay etiquetas de los pacientes propios, así que la comprobación que se imprime es de
**consistencia, no de acierto**: que los códigos salgan ordenados a lo largo del arco.
Es necesario y no suficiente. El 0,932 de FDI por diente de la ficha del experimento es
sobre el test de Teeth3DS+ y **no se transfiere por decreto** a otro escáner, otro
paciente y otra patología.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import vtk
from torch.nn import Linear as Lin
from torch_geometric.data import Data
from torch_geometric.nn import (
    MLP,
    PointTransformerConv,
    fps,
    knn,
    knn_graph,
    knn_interpolate,
)
from torch_geometric.transforms import NormalizeScale
from torch_geometric.utils import scatter
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

RAIZ = Path(__file__).resolve().parent.parent
for _paquete in ("ingestion-agents", "fusion-agents", "core-schemas", "tooth-aggregation"):
    sys.path.insert(0, str(RAIZ / f"packages/{_paquete}/src"))

from fusion_agents.marco import a_marco_de, marco_arcada, marco_canonico  # noqa: E402
from ingestion_agents import ArtifactStore, MeshAgent  # noqa: E402
from tooth_aggregation import aggregate_teeth  # noqa: E402

vtk.vtkObject.GlobalWarningDisplayOff()

CHUNK, PASADAS = 2048, 3   # trozos del TAMAÑO DE ENTRENAMIENTO: misma distribución
KNN_INST, MIN_INST, MERGE_MULT = 8, 30, 12.0
RAZON_DUDOSA = 0.6
FDI_INFERIOR = list(range(48, 40, -1)) + list(range(31, 39))
FDI_SUPERIOR = list(range(18, 10, -1)) + list(range(21, 29))


# --------------------------------------------------------------------------- #
# El modelo. Copiado del notebook: los pesos exigen exactamente esta estructura.
# --------------------------------------------------------------------------- #
class TransformerBlock(torch.nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.lin_in = Lin(ic, ic)
        self.lin_out = Lin(oc, oc)
        self.pos_nn = MLP([3, 64, oc], norm=None, plain_last=False)
        self.attn_nn = MLP([oc, 64, oc], norm=None, plain_last=False)
        self.transformer = PointTransformerConv(
            ic, oc, pos_nn=self.pos_nn, attn_nn=self.attn_nn
        )

    def forward(self, x, pos, ei):
        x = self.lin_in(x).relu()
        x = self.transformer(x, pos, ei)
        return self.lin_out(x).relu()


class TransitionDown(torch.nn.Module):
    def __init__(self, ic, oc, ratio=0.25, k=16):
        super().__init__()
        self.k, self.ratio = k, ratio
        self.mlp = MLP([ic, oc], plain_last=False)

    def forward(self, x, pos, batch):
        idc = fps(pos, ratio=self.ratio, batch=batch)
        sb = batch[idc] if batch is not None else None
        idk = knn(pos, pos[idc], k=self.k, batch_x=batch, batch_y=sb)
        x = self.mlp(x)
        xo = scatter(x[idk[1]], idk[0], dim=0, dim_size=idc.size(0), reduce="max")
        return xo, pos[idc], sb


class TransitionUp(torch.nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.mlp_sub = MLP([ic, oc], plain_last=False)
        self.mlp = MLP([oc, oc], plain_last=False)

    def forward(self, x, x_sub, pos, pos_sub, batch=None, batch_sub=None):
        x_sub = self.mlp_sub(x_sub)
        xi = knn_interpolate(x_sub, pos_sub, pos, k=3, batch_x=batch_sub, batch_y=batch)
        return self.mlp(x) + xi


class Net(torch.nn.Module):
    def __init__(self, ic, oc, dim_model, k=16):
        super().__init__()
        self.k = k
        ic = max(ic, 1)
        self.mlp_input = MLP([ic, dim_model[0]], plain_last=False)
        self.transformer_input = TransformerBlock(dim_model[0], dim_model[0])
        self.transformers_up = torch.nn.ModuleList()
        self.transformers_down = torch.nn.ModuleList()
        self.transition_up = torch.nn.ModuleList()
        self.transition_down = torch.nn.ModuleList()
        for i in range(len(dim_model) - 1):
            self.transition_down.append(TransitionDown(dim_model[i], dim_model[i + 1], k=k))
            self.transformers_down.append(TransformerBlock(dim_model[i + 1], dim_model[i + 1]))
            self.transition_up.append(TransitionUp(dim_model[i + 1], dim_model[i]))
            self.transformers_up.append(TransformerBlock(dim_model[i], dim_model[i]))
        self.mlp_summit = MLP([dim_model[-1], dim_model[-1]], norm=None, plain_last=False)
        self.transformer_summit = TransformerBlock(dim_model[-1], dim_model[-1])
        self.mlp_output = MLP([dim_model[0], 64, oc], norm=None)

    def forward(self, x, pos, batch=None):
        if x is None:
            x = torch.ones((pos.shape[0], 1), device=pos.device)
        ox, op, ob = [], [], []
        x = self.mlp_input(x)
        ei = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_input(x, pos, ei)
        ox.append(x), op.append(pos), ob.append(batch)
        for i in range(len(self.transformers_down)):
            x, pos, batch = self.transition_down[i](x, pos, batch=batch)
            ei = knn_graph(pos, k=self.k, batch=batch)
            x = self.transformers_down[i](x, pos, ei)
            ox.append(x), op.append(pos), ob.append(batch)
        x = self.mlp_summit(x)
        ei = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_summit(x, pos, ei)
        for i in range(len(self.transformers_down)):
            x = self.transition_up[-i - 1](
                x=ox[-i - 2], x_sub=x, pos=op[-i - 2], pos_sub=op[-i - 1],
                batch_sub=ob[-i - 1], batch=ob[-i - 2],
            )
            ei = knn_graph(op[-i - 2], k=self.k, batch=ob[-i - 2])
            x = self.transformers_up[-i - 1](x, op[-i - 2], ei)
        return F.log_softmax(self.mlp_output(x), dim=-1)


# --------------------------------------------------------------------------- #
def normales_vtk(V: np.ndarray, caras: np.ndarray) -> np.ndarray:
    """Normales por vértice **como en el entrenamiento**: sin split, con consistencia.

    Que sea el mismo cálculo no es pedantería: la normal es la única entrada del modelo,
    así que otra convención —de signo, o de promediado— es un cambio de dominio gratuito
    encima del que ya trae el escáner.
    """
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(V, dtype=np.float32), deep=1))
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    celdas = vtk.vtkCellArray()
    for t in caras:
        celdas.InsertNextCell(3)
        for j in t:
            celdas.InsertCellPoint(int(j))
    poly.SetPolys(celdas)
    nf = vtk.vtkPolyDataNormals()
    nf.SetInputData(poly)
    nf.SetSplitting(0)
    nf.SetConsistency(1)
    nf.SetComputePointNormals(1)
    nf.SetComputeCellNormals(0)
    nf.Update()
    return vtk_to_numpy(nf.GetOutput().GetPointData().GetNormals()).astype(np.float32)


@torch.no_grad()
def logprob_denso(modelo, pos, x, n_clases: int, dev: str) -> np.ndarray:
    """Log-probabilidades por vértice, acumuladas en trozos del tamaño de entrenamiento.

    Un subconjunto aleatorio de 2048 puntos es exactamente lo que el modelo vio con
    `FixedPoints`, así que no hay desajuste de distribución. Pasar la malla entera de
    golpe sí lo tendría: el grafo kNN saldría de otra densidad.
    """
    modelo.eval()
    n = len(pos)
    acc = torch.zeros(n, n_clases, device=dev)
    cnt = torch.zeros(n, device=dev)
    por_lote = CHUNK * 8
    for _ in range(PASADAS):
        perm = torch.randperm(n, device=dev)
        pad = (-n) % CHUNK  # completa el último trozo: evita grafos diminutos
        if pad:
            perm = torch.cat([perm, perm[torch.randint(n, (pad,), device=dev)]])
        for s in range(0, len(perm), por_lote):
            sub = perm[s : s + por_lote]
            b = torch.arange(len(sub), device=dev) // CHUNK
            acc.index_add_(0, sub, modelo(x[sub], pos[sub], b))
            cnt.index_add_(0, sub, torch.ones(len(sub), device=dev))
    return (acc / cnt.clamp_min(1)[:, None]).cpu().numpy()


def restringe_a_arcada(logp: np.ndarray, codigos: list[int], arcada: str) -> np.ndarray:
    """Anula los códigos de la arcada contraria y renormaliza a log-probabilidad."""
    permitidos = set(FDI_INFERIOR if arcada == "lower" else FDI_SUPERIOR)
    mask = np.array([c == 0 or c in permitidos for c in codigos])
    m = logp.copy()
    m[:, ~mask] = -1e9
    return m - np.logaddexp.reduce(m, axis=1, keepdims=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--escaneo", type=Path, required=True)
    ap.add_argument("--arcada", choices=("lower", "upper"), required=True)
    ap.add_argument("--salida", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path,
                    default=RAIZ / "data/processed/a3-checkpoints/normales-ce-baseline.pt")
    ap.add_argument("--referencia", type=Path, default=None,
                    help="OBJ de Teeth3DS+ que define la pose esperada por el modelo. "
                         "Sin él no se canoniza, que es lo que rompe la inferencia.")
    ap.add_argument("--sin-restringir", action="store_true",
                    help="No enmascarar la arcada contraria (para diagnóstico).")
    args = ap.parse_args()
    args.salida.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    codigos = list(ck["codes"])
    n_clases, in_ch = int(ck["K"]), int(ck["in_ch"])
    modelo = Net(in_ch, n_clases, dim_model=[32, 64, 128, 256, 512], k=16).to(dev)
    modelo.load_state_dict(ck["state_dict"])
    print(f"modelo {ck['name']} · semilla {ck['seed']} · {n_clases} clases · {dev}")

    store = ArtifactStore(args.salida / "_store")
    arte = store.load(MeshAgent(store).ingest(args.escaneo).artifact_ref)
    V = arte["positions"].astype(np.float64)
    N = normales_vtk(arte["positions"].astype(np.float32), arte["faces"]).astype(np.float64)
    print(f"malla: {len(V):,} vertices · {len(arte['faces']):,} caras")

    if args.referencia is not None:
        lector = vtk.vtkOBJReader()
        lector.SetFileName(str(args.referencia))
        lector.Update()
        V_ref = vtk_to_numpy(lector.GetOutput().GetPoints().GetData()).astype(np.float64)
        origen = marco_canonico(V)[:2]
        destino = marco_canonico(V_ref)[:2]
        razon = marco_canonico(V)[2]
        V_ent, N_ent = a_marco_de(V, origen, destino, normales=N)
        print(f"pose canonizada · razon de orientacion {razon:.2f}"
              + ("  ⚠ DUDOSA: desconfia del resultado" if razon > RAZON_DUDOSA else ""))
    else:
        print("⚠ sin --referencia NO se canoniza la pose; si el escaner no escribe sus "
              "ejes como Teeth3DS+, la salida sera de la arcada equivocada.")
        V_ent, N_ent = V, N

    # A simple precision: los pesos del checkpoint son float32 y la malla viene en
    # float64. El error que da si no es «mat1 and mat2 must have the same dtype», que
    # salta en mitad del grafo y no dice de dónde viene el doble.
    pos = torch.from_numpy(np.ascontiguousarray(V_ent, dtype=np.float32))
    d = NormalizeScale()(Data(pos=pos))
    logp = logprob_denso(
        modelo, d.pos.to(dev),
        torch.from_numpy(np.ascontiguousarray(N_ent, dtype=np.float32)).to(dev),
        n_clases, dev,
    )
    if not args.sin_restringir:
        logp = restringe_a_arcada(logp, codigos, args.arcada)

    instancias = aggregate_teeth(
        V, logp, gum_class=0, k=KNN_INST, min_size=MIN_INST, merge_mult=MERGE_MULT,
        codes=dict(enumerate(codigos)),
    )
    _, _, P, _ = marco_arcada(V)
    filas = sorted(
        ({"fdi": int(t.fdi), "n": t.size, "x": float(P[t.vertices].mean(0)[0]),
          "confianza": float(t.confidence)} for t in instancias),
        key=lambda f: f["x"],
    )
    print(f"\n{len(instancias)} instancias\n{'x (mm)':>8} {'FDI':>5} {'pts':>7} {'logp':>7}")
    for f in filas:
        print(f"{f['x']:>8.1f} {f['fdi']:>5} {f['n']:>7,} {f['confianza']:>7.3f}")

    # Consistencia: sin etiquetas propias, lo único comprobable es que los códigos
    # salgan ORDENADOS por el arco, que es como está puesta una dentición.
    esperado = FDI_INFERIOR if args.arcada == "lower" else FDI_SUPERIOR
    grandes = [f["fdi"] for f in filas if f["n"] >= 200]
    orden = [esperado.index(c) for c in grandes if c in esperado]
    print(f"\nFDI por el arco: {grandes}")
    rho = float("nan")
    if len(orden) >= 4:
        from scipy.stats import spearmanr

        rho, p = spearmanr(range(len(orden)), orden)
        print(f"monotonia FDI vs posicion: rho {rho:+.2f} (p {p:.5f}) → "
              f"{'CONSISTENTE' if abs(rho) > 0.9 else 'NO consistente, desconfia'}")

    etiquetas = np.zeros(len(V), dtype=np.int16)
    for t in instancias:
        etiquetas[t.vertices] = int(t.fdi)
    np.save(args.salida / f"etiquetas_{args.arcada}.npy", etiquetas)
    np.save(args.salida / f"logprob_{args.arcada}.npy", logp.astype(np.float32))
    destino_json = args.salida / f"fdi_{args.arcada}.json"
    destino_json.write_text(
        json.dumps({"escaneo": args.escaneo.name, "arcada": args.arcada,
                    "modelo": ck["name"], "monotonia_rho": rho,
                    "instancias": filas}, indent=2),
        encoding="utf-8",
    )
    print(f"→ {destino_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
