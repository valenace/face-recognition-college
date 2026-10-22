import pytest
import numpy as np
import pickle
import csv
from pathlib import Path
from datetime import datetime


@pytest.mark.unit
class TestCalcularIoU:
    def _iou(self, boxA, boxB):
        from scripts.asistencia_unificada import MotorAsistencia
        return MotorAsistencia._calcular_iou(boxA, boxB)

    def test_identico(self):
        box = [10, 10, 100, 100]
        assert self._iou(box, box) == pytest.approx(1.0)

    def test_sin_overlap(self):
        assert self._iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0

    def test_parcial(self):
        iou = self._iou([0, 0, 20, 20], [10, 10, 30, 30])
        assert 0.0 < iou < 1.0

    def test_contenido(self):
        iou = self._iou([0, 0, 100, 100], [25, 25, 75, 75])
        inner_area = 50 * 50
        outer_area = 100 * 100
        expected = inner_area / (outer_area + inner_area - inner_area)
        assert iou == pytest.approx(expected, abs=0.01)

    def test_touching_edges(self):
        assert self._iou([0, 0, 10, 10], [10, 0, 20, 10]) == 0.0

    def test_un_pixel_overlap(self):
        iou = self._iou([0, 0, 10, 10], [9, 0, 19, 10])
        assert iou > 0


@pytest.mark.unit
class TestBaseDatos:
    """El almacenamiento pasó de pickle a PostgreSQL + pgvector (core.db_config)."""

    def _registro(self):
        from scripts.registro_unificado import RegistroBiometrico
        return RegistroBiometrico.__new__(RegistroBiometrico)

    def test_cargar_db_delega_en_pgvector(self, monkeypatch):
        import core.db_config
        esperado = {"alexa": np.ones(512, dtype="float32")}
        monkeypatch.setattr(core.db_config, "cargar_centroides", lambda: esperado)
        assert self._registro()._cargar_db() == esperado

    def test_cargar_db_si_la_bd_falla_devuelve_vacio(self, monkeypatch):
        import core.db_config

        def boom():
            raise RuntimeError("sin conexión")

        monkeypatch.setattr(core.db_config, "cargar_centroides", boom)
        assert self._registro()._cargar_db() == {}

    def test_guardar_sin_clave_no_hace_nada(self, monkeypatch):
        import core.db_config
        llamadas = []
        monkeypatch.setattr(
            core.db_config, "guardar_centroide",
            lambda *a, **k: llamadas.append(a) or True,
        )
        reg = self._registro()
        reg.db_embeddings = {"u1": np.zeros(512, dtype="float32")}
        reg._guardar_db()
        assert llamadas == []

    def test_guardar_persiste_solo_la_clave(self, monkeypatch):
        import core.db_config
        llamadas = []

        def falso(clave, vector, **kwargs):
            llamadas.append((clave, kwargs))
            return True

        monkeypatch.setattr(core.db_config, "guardar_centroide", falso)
        reg = self._registro()
        reg.db_embeddings = {"u1": np.ones(512, dtype="float32")}
        reg._guardar_db("u1")
        assert len(llamadas) == 1
        assert llamadas[0][0] == "u1"
        assert llamadas[0][1]["crear_usuario"] is False


@pytest.mark.unit
class TestCSVAsistencia:
    def _crear_motor(self, db_path, csv_path):
        from scripts.asistencia_unificada import MotorAsistencia
        db = {"user1": np.random.randn(512).astype("float32")}
        db["user1"] /= np.linalg.norm(db["user1"])
        with open(db_path, "wb") as f:
            pickle.dump(db, f)
        motor = MotorAsistencia.__new__(MotorAsistencia)
        motor.db_path = Path(db_path)
        motor.csv_path = Path(csv_path)
        motor.asistencia_hoy = set()
        motor.nombres_lista = list(db.keys())
        return motor

    def test_preparar_csv_nuevo(self, tmp_path):
        db_path = tmp_path / "db.pkl"
        csv_path = tmp_path / "asistencia.csv"
        motor = self._crear_motor(db_path, csv_path)
        motor._preparar_archivo_csv()
        assert csv_path.exists()
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == ["Identificador", "Fecha", "Hora", "Similitud"]

    def test_registrar_primera_vez(self, tmp_path):
        db_path = tmp_path / "db.pkl"
        csv_path = tmp_path / "asistencia.csv"
        motor = self._crear_motor(db_path, csv_path)
        motor._preparar_archivo_csv()
        motor._registrar_asistencia("user1", 0.85)
        assert "user1" in motor.asistencia_hoy
        with open(csv_path, "r") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 2

    def test_registrar_duplicado(self, tmp_path):
        db_path = tmp_path / "db.pkl"
        csv_path = tmp_path / "asistencia.csv"
        motor = self._crear_motor(db_path, csv_path)
        motor._preparar_archivo_csv()
        motor._registrar_asistencia("user1", 0.85)
        motor._registrar_asistencia("user1", 0.90)
        with open(csv_path, "r") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 2

    def test_registrar_desconocido(self, tmp_path):
        db_path = tmp_path / "db.pkl"
        csv_path = tmp_path / "asistencia.csv"
        motor = self._crear_motor(db_path, csv_path)
        motor._preparar_archivo_csv()
        motor._registrar_asistencia("Desconocido", 0.5)
        with open(csv_path, "r") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 1

    def test_cargar_registros_hoy(self, tmp_path):
        db_path = tmp_path / "db.pkl"
        csv_path = tmp_path / "asistencia.csv"
        motor = self._crear_motor(db_path, csv_path)
        motor._preparar_archivo_csv()
        hoy = datetime.now().strftime("%Y-%m-%d")
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["user_existente", hoy, "09:00:00", "0.8500"])
        motor2 = self._crear_motor(db_path, csv_path)
        motor2._preparar_archivo_csv()
        assert "user_existente" in motor2.asistencia_hoy

    def test_ignorar_registros_ayer(self, tmp_path):
        db_path = tmp_path / "db.pkl"
        csv_path = tmp_path / "asistencia.csv"
        motor = self._crear_motor(db_path, csv_path)
        motor._preparar_archivo_csv()
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["user_ayer", "2020-01-01", "09:00:00", "0.8500"])
        motor2 = self._crear_motor(db_path, csv_path)
        motor2._preparar_archivo_csv()
        assert "user_ayer" not in motor2.asistencia_hoy


@pytest.mark.unit
class TestVotacionTemporal:
    def test_votos_suficientes_anclan(self):
        votos = {}
        candidato = "user1"
        votos_requeridos = 3
        for _ in range(votos_requeridos):
            votos[candidato] = votos.get(candidato, 0) + 1
        assert votos[candidato] >= votos_requeridos

    def test_votos_insuficientes_no_anclan(self):
        votos = {}
        candidato = "user1"
        votos_requeridos = 3
        for _ in range(2):
            votos[candidato] = votos.get(candidato, 0) + 1
        assert votos[candidato] < votos_requeridos

    def test_votos_mixtos(self):
        votos = {}
        votos_requeridos = 3
        for c in ["user1", "user2", "user1", "user2", "user1"]:
            votos[c] = votos.get(c, 0) + 1
        assert votos["user1"] >= votos_requeridos
        assert votos["user2"] < votos_requeridos


@pytest.mark.unit
class TestConflictosIdentidad:
    """Dos pistas simultáneas no pueden compartir la misma identidad."""

    def _motor(self):
        from scripts.asistencia_unificada_optimizada import MotorAsistencia

        motor = MotorAsistencia.__new__(MotorAsistencia)
        motor.margen_conflicto = 0.05
        motor.frames_veto = 90
        motor.frame_idx = 100
        motor.identidades_ancladas = {
            70: {"nombre": "dani", "similitud": 0.62, "frame": 0},
            117: {"nombre": "dani", "similitud": 0.32, "frame": 0},
        }
        motor.votos_identidad = {}
        motor.vetos_identidad = {}
        return motor

    def test_degrada_la_pista_con_menor_similitud(self):
        motor = self._motor()
        decisiones = [
            {"track_id": 70, "nombre": "dani", "estado": "ANCLADO", "similitud": 0.62},
            {"track_id": 117, "nombre": "dani", "estado": "ANCLADO", "similitud": 0.32},
        ]
        motor._resolver_conflictos(decisiones)
        assert decisiones[0]["nombre"] == "dani"
        assert decisiones[1]["nombre"] == "Desconocido"
        assert decisiones[1]["estado"] == "DESCONOCIDO"
        assert decisiones[1]["motivo"].startswith("conflicto_con_pista_70")
        assert 117 not in motor.identidades_ancladas
        assert 117 in motor.vetos_identidad

    def test_no_degrada_si_la_diferencia_es_pequena(self):
        motor = self._motor()
        decisiones = [
            {"track_id": 70, "nombre": "dani", "estado": "ANCLADO", "similitud": 0.40},
            {"track_id": 117, "nombre": "dani", "estado": "ANCLADO", "similitud": 0.38},
        ]
        motor._resolver_conflictos(decisiones)
        assert decisiones[0]["nombre"] == "dani"
        assert decisiones[1]["nombre"] == "dani"
        assert decisiones[1]["motivo"].startswith("identidad_ambigua")

    def test_no_toca_pistas_con_nombres_distintos(self):
        motor = self._motor()
        decisiones = [
            {"track_id": 70, "nombre": "dani", "estado": "ANCLADO", "similitud": 0.60},
            {"track_id": 90, "nombre": "alexa", "estado": "ANCLADO", "similitud": 0.35},
        ]
        motor._resolver_conflictos(decisiones)
        assert [d["nombre"] for d in decisiones] == ["dani", "alexa"]


@pytest.mark.unit
class TestBandaRevision:
    """Las identidades con similitud 0.30-0.40 se marcan para revisión."""

    def _motor(self):
        from scripts.asistencia_unificada_optimizada import MotorAsistencia

        motor = MotorAsistencia.__new__(MotorAsistencia)
        motor.umbral_revision = 0.40
        return motor

    def test_marca_solo_las_confirmaciones_de_baja_confianza(self):
        decisiones = [
            {"estado": "CONFIRMADO", "similitud": 0.35},
            {"estado": "ANCLADO", "similitud": 0.55},
            {"estado": "DESCONOCIDO", "similitud": 0.20},
            {"estado": "CENTRATE", "similitud": None},
        ]
        self._motor()._marcar_revision(decisiones)
        assert [d["revision"] for d in decisiones] == [True, False, False, False]

    def test_limite_exacto(self):
        decisiones = [
            {"estado": "CONFIRMADO", "similitud": 0.40},
            {"estado": "CONFIRMADO", "similitud": 0.399},
        ]
        self._motor()._marcar_revision(decisiones)
        assert [d["revision"] for d in decisiones] == [False, True]
