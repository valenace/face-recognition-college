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
class TestAsistenciaCSVOpcional:
    """El motor optimizado solo escribe CSV cuando se le pide.

    Los flujos web y el modo `--sesion` escriben en RegistroAsistencia; el CSV
    queda para los experimentos.
    """

    def _motor(self, csv_path, registrar_csv):
        from scripts.asistencia_unificada_optimizada import MotorAsistencia

        motor = MotorAsistencia.__new__(MotorAsistencia)
        motor.csv_path = Path(csv_path)
        motor.asistencia_hoy = set()
        motor.fecha_actual = datetime.now().strftime("%Y-%m-%d")
        motor.registrar_csv = registrar_csv
        return motor

    def test_modo_bd_no_crea_csv(self, tmp_path):
        csv_path = tmp_path / "asistencia.csv"
        motor = self._motor(csv_path, registrar_csv=False)
        motor._registrar_asistencia("user1", 0.85)
        assert not csv_path.exists()
        assert motor.asistencia_hoy == set()

    def test_modo_csv_escribe(self, tmp_path):
        csv_path = tmp_path / "asistencia.csv"
        motor = self._motor(csv_path, registrar_csv=True)
        motor._preparar_archivo_csv()
        motor._registrar_asistencia("user1", 0.85)
        assert "user1" in motor.asistencia_hoy
        with open(csv_path, "r") as f:
            assert len(list(csv.reader(f))) == 2


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


@pytest.mark.unit
class TestSeguimientoIntrusos:
    """Un desconocido de un frame no es intruso si la pista es un registrado."""

    def _desconocido(self, track_id, sim, motivo=None, candidato="dani", bbox=(100, 100, 150, 150)):
        return {
            "track_id": track_id,
            "estado": "DESCONOCIDO",
            "similitud": sim,
            "candidato": candidato,
            "motivo": motivo,
            "bbox": bbox,
        }

    def _confirmado(self, track_id, nombre="dani"):
        return {"track_id": track_id, "estado": "CONFIRMADO", "similitud": 0.5, "nombre": nombre}

    def test_pista_persistente_bajo_umbral_crea_intruso(self):
        from core.intrusos import SeguimientoIntrusos

        seg = SeguimientoIntrusos(frames=3)
        crear = []
        for _ in range(3):
            crear, _ = seg.evaluar([self._desconocido(7, 0.10)])
        assert [d["track_id"] for d in crear] == [7]

    def test_pista_que_supera_el_umbral_no_es_intruso(self):
        from core.intrusos import SeguimientoIntrusos

        seg = SeguimientoIntrusos(frames=3, umbral=0.24)
        crear = []
        for sim in (0.28, 0.29, 0.30):
            crear, _ = seg.evaluar([self._desconocido(7, sim)])
        assert crear == []

    def test_fila_creada_se_descarta_si_la_pista_mejora(self):
        from core.intrusos import SeguimientoIntrusos

        seg = SeguimientoIntrusos(frames=2, umbral=0.24)
        crear = []
        for _ in range(2):
            crear, _ = seg.evaluar([self._desconocido(7, 0.10)])
        assert crear
        seg.marcar_registrado(7, 99)
        _, descartar = seg.evaluar([self._desconocido(7, 0.60)])
        assert descartar == [99]

    def test_pista_que_se_acerco_a_un_registrado_no_es_intruso(self):
        """El promedio es acumulativo: si la cara mejora, la pista se descarta."""
        from core.intrusos import SeguimientoIntrusos

        seg = SeguimientoIntrusos(frames=3, umbral=0.24, umbral_mejor=0.26)
        crear = []
        for _ in range(3):
            crear, _ = seg.evaluar([self._desconocido(7, 0.10)])
        assert crear
        seg.marcar_registrado(7, 77)
        # El promedio sigue bajo (0.15), pero la mejor similitud efectiva ya
        # demuestra que es un estudiante registrado.
        _, descartar = seg.evaluar([self._desconocido(7, 0.30)])
        assert descartar == [77]

    def test_conflicto_de_identidad_no_es_intruso(self):
        from core.intrusos import SeguimientoIntrusos

        seg = SeguimientoIntrusos(frames=2)
        crear = []
        for _ in range(5):
            crear, _ = seg.evaluar(
                [self._desconocido(7, 0.10, "conflicto_con_pista_2")]
            )
        assert crear == []

    def test_confirmacion_descarta_la_fila(self):
        from core.intrusos import SeguimientoIntrusos

        seg = SeguimientoIntrusos(frames=2)
        for _ in range(2):
            crear, _ = seg.evaluar([self._desconocido(7, 0.10)])
        assert crear
        seg.marcar_registrado(7, 42)
        _, descartar = seg.evaluar([self._confirmado(7)])
        assert descartar == [42]

    def test_pocos_frames_no_crean_intruso(self):
        from core.intrusos import SeguimientoIntrusos

        seg = SeguimientoIntrusos(frames=3)
        crear = []
        for _ in range(2):
            crear, _ = seg.evaluar([self._desconocido(7, 0.10)])
        assert crear == []

    def test_candidato_confirmado_en_otra_pista_no_descarta_desconocido(self):
        from core.intrusos import SeguimientoIntrusos

        seg = SeguimientoIntrusos(frames=3, umbral=0.24)
        crear = []
        # La pista 7 tiene similitud 0.28 con 'dani', pero 'dani' ya está confirmada en la pista 1
        for _ in range(3):
            crear, _ = seg.evaluar([
                self._confirmado(1, nombre="dani"),
                self._desconocido(7, 0.28, candidato="dani"),
            ])
        assert [d["track_id"] for d in crear] == [7]

    def test_deduplicacion_espacial_reutiliza_asiento(self):
        from core.intrusos import SeguimientoIntrusos

        seg = SeguimientoIntrusos(frames=2)
        bbox1 = (100, 100, 140, 140)
        for _ in range(2):
            crear, _ = seg.evaluar([self._desconocido(10, 0.10, bbox=bbox1)])
        assert [d["track_id"] for d in crear] == [10]
        seg.marcar_registrado(10, registro_id=501, bbox=bbox1)

        # La pista 10 desaparece unos frames...
        for _ in range(5):
            seg.evaluar([])

        # Aparece la pista 25 en el mismo asiento (bbox cercano)
        bbox2 = (105, 102, 145, 142)
        crear2 = []
        for _ in range(2):
            crear2, _ = seg.evaluar([self._desconocido(25, 0.10, bbox=bbox2)])

        assert len(crear2) == 1
        assert crear2[0]["track_id"] == 25
        assert crear2[0]["reutilizar_registro_id"] == 501


@pytest.mark.unit
class TestAlias:
    """El alias debe ser único: nombre para unc_*, username para el resto."""

    def test_prefijo_unc(self):
        from core.db_config import alias_de

        assert alias_de("unc_alexa", "Alexa") == "alexa"

    def test_username_sin_prefijo(self):
        from core.db_config import alias_de

        assert alias_de("20230001", "Juan") == "20230001"


@pytest.mark.unit
class TestRecargaCentroides:
    """El motor puede reconstruir el índice FAISS sin reiniciarse."""

    def test_reconstruye_el_indice_y_limpia_el_estado(self, monkeypatch):
        import numpy as np

        import core.db_config as db_config
        from scripts.asistencia_unificada_optimizada import MotorAsistencia

        motor = MotorAsistencia.__new__(MotorAsistencia)
        motor.usar_bd = True
        motor.identidades_ancladas = {"x": 1}
        motor.votos_identidad = {"x": 1}
        motor.votos_fraude = {"x": 1}
        motor.vetos_identidad = {"x": 1}

        monkeypatch.setattr(
            db_config,
            "cargar_centroides",
            lambda: {"alexa": np.ones(512), "bob": np.ones(512)},
        )

        assert motor.recargar_centroides() is True
        assert motor.indice_faiss.ntotal == 2
        assert motor.nombres_lista == ["alexa", "bob"]
        assert motor.identidades_ancladas == {}
        assert motor.votos_identidad == {}
        assert motor.votos_fraude == {}
        assert motor.vetos_identidad == {}

    def test_registro_biometrico_recarga_su_cache(self, monkeypatch):
        from scripts.registro_unificado import RegistroBiometrico

        registro = RegistroBiometrico.__new__(RegistroBiometrico)
        monkeypatch.setattr(
            registro, "_cargar_db", lambda: {"alexa": [1.0] * 512}
        )

        assert registro.recargar_centroides() is True
        assert registro.db_embeddings == {"alexa": [1.0] * 512}
