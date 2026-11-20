"""La reserva de Teeth3DS+: que no cambie sin que se note y que no se mezcle con nada.

Sin red: la lista oficial y los zips de OSF se sustituyen por datos en memoria.
"""

from __future__ import annotations

import importlib.util
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

_RAIZ = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("fetch_teeth3ds_osf",
                                               _RAIZ / "scripts" / "fetch_teeth3ds_osf.py")
assert _spec and _spec.loader
osf = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = osf
_spec.loader.exec_module(osf)


def test_solo_cuentan_los_pacientes_con_las_dos_arcadas() -> None:
    texto = "AAA_upper\nAAA_lower\nBBB_upper\nCCC_lower\nCCC_upper\n"
    assert osf.pacientes_de_lista(texto) == ["AAA", "CCC"]


def test_la_reserva_es_determinista_y_no_depende_del_orden_de_entrada() -> None:
    ids = [f"P{k:03d}" for k in range(300)]
    a = osf.elige_reserva(ids)
    b = osf.elige_reserva(list(reversed(ids)))
    assert a == b == sorted(a)
    assert len(a) == osf.PACIENTES_RESERVA == len(set(a))
    assert osf.elige_reserva(ids, semilla=1) != a


def test_pedir_mas_de_los_que_hay_falla() -> None:
    with pytest.raises(ValueError):
        osf.elige_reserva(["A", "B"], n=3)


def test_la_lista_versionada_tiene_60_pacientes_sin_repetir() -> None:
    ids = osf.ids_reserva()
    assert len(ids) == osf.PACIENTES_RESERVA == len(set(ids))
    assert ids == sorted(ids)


def test_la_segunda_reserva_no_repite_ninguno_de_la_primera() -> None:
    privados = [f"P{k:03d}" for k in range(300)]
    primera = osf.elige_reserva(privados)
    segunda = osf.elige_reserva_2(privados, primera)
    assert len(segunda) == osf.PACIENTES_RESERVA == len(set(segunda))
    assert not set(segunda) & set(primera)
    assert segunda == osf.elige_reserva_2(list(reversed(privados)), primera)


def test_la_segunda_reserva_exige_que_la_primera_salga_de_la_misma_lista() -> None:
    with pytest.raises(ValueError):
        osf.elige_reserva_2(["A", "B"], ["Z"])


def test_las_dos_reservas_versionadas_son_disjuntas() -> None:
    segunda = osf.ids_reserva_2()
    assert len(segunda) == osf.PACIENTES_RESERVA == len(set(segunda))
    assert segunda == sorted(segunda)
    assert not set(segunda) & set(osf.ids_reserva())


def test_la_reserva_no_esta_en_los_datos_locales() -> None:
    """Los 300 locales son el public-training-set-1; la reserva sale del private-testing-set."""
    mallas = _RAIZ / "data/raw/teeth3ds/3D_scans_per_patient_obj_files"
    if not mallas.is_dir():
        pytest.skip("Teeth3DS+ no está descargado en este checkout")
    locales = {d.name for d in mallas.iterdir()}
    assert not set(osf.ids_reserva()) & locales
    assert not set(osf.ids_reserva_2()) & locales


def _zip_en_memoria(miembros: dict[str, bytes]) -> zipfile.ZipFile:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for nombre, datos in miembros.items():
            zf.writestr(nombre, datos)
    buf.seek(0)
    return zipfile.ZipFile(buf)


def test_el_indice_agrupa_por_paciente_e_ignora_lo_demas() -> None:
    zf = _zip_en_memoria({
        "upper/AAA/AAA_upper.obj": b"v 0 0 0\n", "upper/AAA/AAA_upper.json": b"{}",
        "lower/AAA/AAA_lower.obj": b"v 0 0 0\n", "README.txt": b"x",
        "upper/BBB/BBB_upper_landmarks.json": b"{}",
    })
    idx = osf.indice_de(zf, "data_part_9")
    assert set(idx) == {"AAA"}
    assert idx["AAA"]["AAA_upper.obj"] == ("data_part_9", "upper/AAA/AAA_upper.obj")
    assert len(idx["AAA"]) == 3


def test_extraer_escribe_los_bytes_y_devuelve_su_sha256(tmp_path: Path) -> None:
    import hashlib

    zf = _zip_en_memoria({"lower/AAA/AAA_lower.json": b'{"labels": [0, 31]}'})
    destino = tmp_path / "AAA" / "AAA_lower.json"
    sha = osf.extrae(zf, "lower/AAA/AAA_lower.json", destino)
    assert destino.read_bytes() == b'{"labels": [0, 31]}'
    assert sha == hashlib.sha256(destino.read_bytes()).hexdigest()


def test_un_paciente_incompleto_en_osf_no_se_baja_a_medias() -> None:
    idx = {"AAA": {"AAA_upper.obj": ["p", "x"], "AAA_upper.json": ["p", "y"]}}
    with pytest.raises(SystemExit):
        osf.ficheros_del_paciente(idx, "AAA")


def test_el_npz_conserva_vertices_normales_y_etiquetas(tmp_path: Path) -> None:
    pytest.importorskip("vtk")
    obj = tmp_path / "t.obj"
    obj.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0.2\nf 1 2 3\nf 2 4 3\n")
    etq = tmp_path / "t.json"
    etq.write_text('{"labels": [0, 11, 11, 0]}')
    n = osf.a_npz(obj, etq, tmp_path / "t.npz")
    z = np.load(tmp_path / "t.npz")
    assert n == 4 and z["pos"].shape == (4, 3) and z["nrm"].shape == (4, 3)
    assert z["labels"].tolist() == [0, 11, 11, 0]
    assert np.allclose(np.linalg.norm(z["nrm"], axis=1), 1.0, atol=1e-5)


def test_los_comentarios_de_la_lista_no_son_ids(tmp_path: Path) -> None:
    f = tmp_path / "l.txt"
    f.write_text("# Reserva de Teeth3DS+: 60 pacientes\nAAA\n\nBBB  # nota\n")
    assert osf.ids_de_fichero(f) == ["AAA", "BBB"]
