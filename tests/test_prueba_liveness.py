"""Tests unitarios de las funciones puras de `prueba_liveness.py`."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/unc"))

from prueba_liveness import (  # noqa: E402
    auc_rango,
    decidir,
    estadisticas_segmento,
    identidad_operador,
    matriz_confusion,
    matriz_por_segmento,
    matriz_sujeto_mostrado,
    metricas_desde_matriz,
    score_real,
    sujetos_mostrados,
    tabla_por_identidad,
    tabla_umbrales,
    voto_estable,
)


@pytest.mark.unit
class TestLivenessFunciones:
    def test_score_real_complementa_la_confianza_de_foto(self):
        assert score_real(True, 0.93) == pytest.approx(0.93)
        assert score_real(False, 0.80) == pytest.approx(0.20)
        assert score_real(False, 0.98) == pytest.approx(0.02)

    def test_decidir_replica_el_umbral_del_motor(self):
        assert decidir(True, 0.95, 0.90) == "real"
        assert decidir(True, 0.89, 0.90) == "foto"
        assert decidir(False, 1.00, 0.90) == "foto"

    def test_matriz_confusion_y_metricas(self):
        filas = [
            {"decision": "real", "gt_real": True},
            {"decision": "foto", "gt_real": True},
            {"decision": "real", "gt_real": False},
            {"decision": "foto", "gt_real": False},
        ]
        m = matriz_confusion(filas)
        assert m == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
        met = metricas_desde_matriz(m)
        assert met["precision"] == pytest.approx(0.5)
        assert met["recall"] == pytest.approx(0.5)
        assert met["f1"] == pytest.approx(0.5)
        assert met["accuracy"] == pytest.approx(0.5)

    def test_matriz_confusion_ignora_sin_cara_y_omitidos(self):
        filas = [
            {"decision": "sin_cara", "gt_real": True},
            {"decision": "omitido", "gt_real": False},
            {"decision": "foto", "gt_real": False},
        ]
        assert matriz_confusion(filas) == {"tp": 0, "fp": 0, "fn": 0, "tn": 1}

    def test_voto_estable(self):
        assert voto_estable(["real", "foto", "real", "real", "real"], 3) == ("real", 2)
        assert voto_estable(["real", "foto"], 3) == (None, None)
        assert voto_estable(["foto", "foto", "foto"], 3) == ("foto", 0)

    def test_auc_rango_perfecta_y_empate(self):
        assert auc_rango([0.9, 0.8], [0.2, 0.1]) == pytest.approx(1.0)
        assert auc_rango([0.5], [0.5]) == pytest.approx(0.5)
        assert auc_rango([], [0.1]) == 0.0

    def test_estadisticas_segmento(self):
        filas = [
            {"decision": "real", "score_real": 0.95, "segundo": 0.0,
             "id_reconocido": "dani", "sim": 0.4},
            {"decision": "real", "score_real": 0.90, "segundo": 0.1,
             "id_reconocido": "dani", "sim": 0.5},
            {"decision": "real", "score_real": 0.85, "segundo": 0.2,
             "id_reconocido": "dani", "sim": 0.6},
            {"decision": "foto", "score_real": 0.20, "segundo": 0.3,
             "id_reconocido": "", "sim": ""},
            {"decision": "sin_cara", "score_real": "", "segundo": 0.4,
             "id_reconocido": "", "sim": ""},
        ]
        s = estadisticas_segmento(filas, "real", 1, 0.0, 0.5, 3)
        assert s["frames"] == 5
        assert s["con_cara"] == 4
        assert s["sin_cara"] == 1
        assert s["frames_real"] == 3
        assert s["frames_foto"] == 1
        assert s["pct_frames_real"] == pytest.approx(75.0)
        assert s["score_mediana"] == pytest.approx(0.875)
        assert s["decision_mayoria"] == "real"
        assert s["s_voto_estable"] == pytest.approx(0.0)
        assert s["id_reconocido"] == "dani"
        assert s["id_casos"] == 3
        assert s["sim_media"] == pytest.approx(0.5)

    def test_matriz_por_segmento(self):
        segmentos = [
            {"gt_real": True, "decision_mayoria": "real"},
            {"gt_real": False, "decision_mayoria": "foto"},
            {"gt_real": False, "decision_mayoria": "real"},
        ]
        assert matriz_por_segmento(segmentos) == {"tp": 1, "fp": 1, "fn": 0, "tn": 1}

    def test_tabla_umbrales_usa_la_regla_del_motor(self):
        filas = [
            {"decision": "real", "gt_real": True, "is_real": True,
             "conf": 0.85, "score_real": 0.85},
            {"decision": "foto", "gt_real": False, "is_real": False,
             "conf": 0.99, "score_real": 0.01},
        ]
        tabla = {r["umbral"]: r for r in tabla_umbrales(filas)}
        assert tabla[0.90]["tp"] == 0
        assert tabla[0.90]["fn"] == 1
        assert tabla[0.90]["tn"] == 1
        assert tabla[0.75]["tp"] == 1
        assert tabla[0.75]["tn"] == 1

    def _filas_mixtas(self):
        return [
            {"segmento": 1, "etiqueta": "real", "gt_real": True,
             "decision": "real", "id_reconocido": "val", "score_real": 0.99,
             "segundo": 0.0},
            {"segmento": 1, "etiqueta": "real", "gt_real": True,
             "decision": "foto", "id_reconocido": "val", "score_real": 0.60,
             "segundo": 0.1},
            {"segmento": 1, "etiqueta": "real", "gt_real": True,
             "decision": "real", "id_reconocido": "Desconocido",
             "score_real": 0.98, "segundo": 0.2},
            {"segmento": 2, "etiqueta": "telefono", "gt_real": False,
             "decision": "foto", "id_reconocido": "kevin", "score_real": 0.0,
             "segundo": 1.0},
            {"segmento": 2, "etiqueta": "telefono", "gt_real": False,
             "decision": "real", "id_reconocido": "kevin", "score_real": 0.95,
             "segundo": 1.1},
            {"segmento": 2, "etiqueta": "telefono", "gt_real": False,
             "decision": "real", "id_reconocido": "val", "score_real": 0.99,
             "segundo": 1.2},
            {"segmento": 2, "etiqueta": "telefono", "gt_real": False,
             "decision": "foto", "id_reconocido": "val", "score_real": 0.70,
             "segundo": 1.3},
        ]

    def test_identidad_operador_y_sujeto_mostrado(self):
        filas = self._filas_mixtas()
        segmentos = [
            {"n_segmento": 1, "gt_real": True},
            {"n_segmento": 2, "gt_real": False},
        ]
        assert identidad_operador(filas) == "val"
        assert sujetos_mostrados(filas, segmentos, "val") == {2: "kevin"}

    def test_matriz_sujeto_mostrado_excluye_al_operador(self):
        filas = self._filas_mixtas()
        mostrados = {2: "kevin"}
        assert matriz_sujeto_mostrado(filas, mostrados, "val") == {
            "tp": 2, "fp": 1, "fn": 1, "tn": 1,
        }

    def test_tabla_por_identidad(self):
        tabla = {r["id_reconocido"]: r for r in tabla_por_identidad(self._filas_mixtas())}
        assert tabla["val"]["frames"] == 4
        assert tabla["val"]["frames_real"] == 2
        assert tabla["val"]["frames_foto"] == 2
        assert tabla["val"]["pct_foto"] == pytest.approx(50.0)
        assert tabla["kevin"]["frames"] == 2
        assert tabla["kevin"]["frames_foto"] == 1
        assert "—" in tabla
        assert tabla["—"]["frames_real"] == 1
