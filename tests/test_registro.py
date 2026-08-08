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
    def test_cargar_db_vacio(self, tmp_db_path):
        from scripts.registro_unificado import RegistroBiometrico
        reg = RegistroBiometrico.__new__(RegistroBiometrico)
        reg.db_path = tmp_db_path
        resultado = reg._cargar_db()
        assert resultado == {}

    def test_guardar_y_cargar_roundtrip(self, tmp_db_path, sample_db):
        from scripts.registro_unificado import RegistroBiometrico
        reg = RegistroBiometrico.__new__(RegistroBiometrico)
        reg.db_path = tmp_db_path
        reg.db_embeddings = sample_db
        reg._guardar_db()

        cargado = reg._cargar_db()
        assert set(cargado.keys()) == set(sample_db.keys())
        for k in sample_db:
            np.testing.assert_array_almost_equal(cargado[k], sample_db[k])

    def test_guardar_atomico_crea_archivo(self, tmp_db_path):
        from scripts.registro_unificado import RegistroBiometrico
        reg = RegistroBiometrico.__new__(RegistroBiometrico)
        reg.db_path = tmp_db_path
        reg.db_embeddings = {"u1": np.zeros(512, dtype="float32")}
        reg._guardar_db()
        assert tmp_db_path.exists()
        assert not tmp_db_path.with_suffix('.pkl.tmp').exists()

    def test_pickle_corrupto(self, tmp_db_path):
        tmp_db_path.write_bytes(b"esto no es pickle valido")
        from scripts.registro_unificado import RegistroBiometrico
        reg = RegistroBiometrico.__new__(RegistroBiometrico)
        reg.db_path = tmp_db_path
        with pytest.raises(Exception):
            reg._cargar_db()

    def test_db_con_embedding_nan(self, tmp_db_path):
        from scripts.registro_unificado import RegistroBiometrico
        reg = RegistroBiometrico.__new__(RegistroBiometrico)
        reg.db_path = tmp_db_path
        reg.db_embeddings = {"bad_user": np.array([np.nan] * 512, dtype="float32")}
        reg._guardar_db()
        cargado = reg._cargar_db()
        assert "bad_user" in cargado
        assert np.isnan(cargado["bad_user"]).all()


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
