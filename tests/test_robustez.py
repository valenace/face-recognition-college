import pytest
import numpy as np
import cv2
import pickle
import tempfile
import os
from pathlib import Path


@pytest.mark.unit
class TestEdgeCasesUtils:
    def test_clahe_imagen_gris_puro(self):
        from core.utils_facial import aplicar_clahe
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        resultado = aplicar_clahe(img)
        assert resultado is not None
        assert resultado.shape == img.shape

    def test_clahe_imagen_un_canal_falla_graceful(self):
        from core.utils_facial import aplicar_clahe
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        resultado = aplicar_clahe(img)
        assert resultado is not None

    def test_blur_imagen_muy_pequena(self):
        from core.utils_facial import es_imagen_borrosa
        img = np.random.randint(0, 256, (5, 5, 3), dtype=np.uint8)
        resultado = es_imagen_borrosa(img, umbral=80)
        assert isinstance(resultado, (bool, np.bool_))

    def test_alinear_con_coordenadas_negativas(self):
        from core.utils_facial import alinear_rostro
        frame = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        landmarks = np.array([
            [-10.0, 50.0],
            [10.0, 50.0],
            [0.0, 70.0],
            [-5.0, 90.0],
            [5.0, 90.0],
        ], dtype=np.float32)
        resultado = alinear_rostro(frame, landmarks)
        assert resultado is None or resultado.shape == (224, 224, 3)


@pytest.mark.unit
class TestRobustezRegistro:
    def test_enrolar_headless_video_inexistente(self, tmp_path):
        from scripts.registro_unificado import RegistroBiometrico
        from unittest.mock import patch
        with patch.object(RegistroBiometrico, '__init__', lambda self, **kw: None):
            reg = RegistroBiometrico()
            reg.db_path = tmp_path / "db.pkl"
            reg.max_muestras = 15
            reg.umbral_blur = 60
            reg.umbral_brillo = 20
            reg.umbral_liveness = 0.90
            from uniface import create_detector, create_recognizer
            from uniface.spoofing import create_spoofer
            reg.detector = create_detector('retinaface')
            reg.recognizer = create_recognizer('arcface')
            reg.spoofer = create_spoofer()
            reg.db_embeddings = {}
            resultado = reg.enrolar_usuario_headless("test", "/no/existe/video.mp4")
        assert resultado["exito"] is False
        assert resultado["muestras"] == 0
        assert "No se pudo abrir" in resultado["mensaje"]

    def test_enrolar_headless_sin_sobrescribir(self, tmp_path):
        from scripts.registro_unificado import RegistroBiometrico
        from unittest.mock import patch
        with patch.object(RegistroBiometrico, '__init__', lambda self, **kw: None):
            reg = RegistroBiometrico()
            reg.db_path = tmp_path / "db.pkl"
            reg.db_embeddings = {"user_existente": np.zeros(512, dtype="float32")}
            resultado = reg.enrolar_usuario_headless("user_existente", "dummy.mp4", sobrescribir=False)
        assert resultado["exito"] is False
        assert "ya existe" in resultado["mensaje"]

    def test_enrolar_headless_sobrescribir_flag(self, tmp_path):
        from scripts.registro_unificado import RegistroBiometrico
        from unittest.mock import patch
        with patch.object(RegistroBiometrico, '__init__', lambda self, **kw: None):
            reg = RegistroBiometrico()
            reg.db_path = tmp_path / "db.pkl"
            reg.db_embeddings = {"user_existente": np.zeros(512, dtype="float32")}
            reg.max_muestras = 15
            resultado = reg.enrolar_usuario_headless("user_existente", "/no/existe.mp4", sobrescribir=True)
        assert resultado["exito"] is False
        assert "No se pudo abrir" in resultado["mensaje"]


@pytest.mark.unit
class TestRobustezTracker:
    def test_bytetrack_sin_detecciones(self):
        import supervision as sv
        tracker = sv.ByteTrack()
        detections = sv.Detections(
            xyxy=np.empty((0, 4)),
            confidence=np.empty((0,)),
        )
        tracked = tracker.update_with_detections(detections=detections)
        assert len(tracked) == 0

    def test_bytetrack_con_deteccion_simple(self):
        import supervision as sv
        tracker = sv.ByteTrack()
        detections = sv.Detections(
            xyxy=np.array([[100, 100, 200, 200]]),
            confidence=np.array([0.95]),
        )
        tracked = tracker.update_with_detections(detections=detections)
        assert len(tracked) >= 1


@pytest.mark.unit
class TestRobustezFAISS:
    def test_busqueda_en_indice_vacio(self):
        import faiss
        index = faiss.IndexFlatIP(512)
        vec = np.random.randn(1, 512).astype("float32")
        vec /= np.linalg.norm(vec)
        dist, idx = index.search(vec, 1)
        assert idx[0][0] == -1

    def test_busqueda_con_nan(self):
        import faiss
        index = faiss.IndexFlatIP(512)
        vec = np.random.randn(512).astype("float32")
        vec /= np.linalg.norm(vec)
        index.add(np.array([vec]))
        query_nan = np.full((1, 512), np.nan, dtype="float32")
        dist, idx = index.search(query_nan, 1)
        assert dist[0][0] != dist[0][0] or idx[0][0] == -1


@pytest.mark.integration
@pytest.mark.slow
class TestRobustezConModelos:
    def test_detector_imagen_muy_pequena(self, detector):
        img = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        rostros = detector.detect(img)
        assert isinstance(rostros, list)
        assert len(rostros) == 0

    def test_detector_imagen_4k(self, detector):
        img = np.random.randint(80, 180, (2160, 3840, 3), dtype=np.uint8)
        rostros = detector.detect(img)
        assert isinstance(rostros, list)

    def test_evaluar_calidad_pipeline_completo(self, detector, spoofer, frame_with_face):
        rostros = detector.detect(frame_with_face)
        if not rostros:
            pytest.skip("No se detecto rostro en frame")
        rostro = rostros[0]
        x1, y1, x2, y2 = map(int, rostro.bbox)
        assert (x2 - x1) >= 40 or (y2 - y1) >= 40

    def test_video_corrupto_no_crashea(self, tmp_path):
        from scripts.registro_unificado import RegistroBiometrico
        from unittest.mock import patch
        video_corrupto = tmp_path / "corrupto.mp4"
        video_corrupto.write_bytes(b"\x00\x01\x02\x03basura" * 100)
        with patch.object(RegistroBiometrico, '__init__', lambda self, **kw: None):
            reg = RegistroBiometrico()
            reg.db_path = tmp_path / "db.pkl"
            reg.max_muestras = 15
            reg.db_embeddings = {}
            from uniface import create_detector, create_recognizer
            from uniface.spoofing import create_spoofer
            reg.detector = create_detector('retinaface')
            reg.recognizer = create_recognizer('arcface')
            reg.spoofer = create_spoofer()
            reg.umbral_blur = 60
            reg.umbral_brillo = 20
            reg.umbral_liveness = 0.90
            resultado = reg.enrolar_usuario_headless("test", str(video_corrupto))
        assert resultado["exito"] is False


@pytest.mark.integration
@pytest.mark.slow
class TestPipelineRegistroVideoReal:
    def test_enrolar_headless_con_video_lino(self, video_lino_path, tmp_path):
        from scripts.registro_unificado import RegistroBiometrico
        from unittest.mock import patch
        with patch.object(RegistroBiometrico, '__init__', lambda self, **kw: None):
            reg = RegistroBiometrico()
            reg.db_path = tmp_path / "db.pkl"
            reg.max_muestras = 5
            reg.umbral_blur = 60
            reg.umbral_brillo = 20
            reg.umbral_liveness = 0.90
            reg.db_embeddings = {}
            from uniface import create_detector, create_recognizer
            from uniface.spoofing import create_spoofer
            reg.detector = create_detector('retinaface')
            reg.recognizer = create_recognizer('arcface')
            reg.spoofer = create_spoofer()
            resultado = reg.enrolar_usuario_headless("Lino", video_lino_path)

        assert resultado["exito"] is True
        assert resultado["muestras"] > 0
        assert resultado["embedding"] is not None
        assert len(resultado["embedding"]) == 512

    def test_centroide_normalizado_tras_enrolamiento(self, video_lino_path, tmp_path):
        from scripts.registro_unificado import RegistroBiometrico
        from unittest.mock import patch
        with patch.object(RegistroBiometrico, '__init__', lambda self, **kw: None):
            reg = RegistroBiometrico()
            reg.db_path = tmp_path / "db.pkl"
            reg.max_muestras = 3
            reg.umbral_blur = 60
            reg.umbral_brillo = 20
            reg.umbral_liveness = 0.90
            reg.db_embeddings = {}
            from uniface import create_detector, create_recognizer
            from uniface.spoofing import create_spoofer
            reg.detector = create_detector('retinaface')
            reg.recognizer = create_recognizer('arcface')
            reg.spoofer = create_spoofer()
            reg.enrolar_usuario_headless("Lino", video_lino_path)

        if "Lino" in reg.db_embeddings:
            emb = reg.db_embeddings["Lino"]
            norma = np.linalg.norm(emb)
            assert norma == pytest.approx(1.0, abs=1e-4)


@pytest.mark.integration
@pytest.mark.slow
class TestPipelineReconocimientoCompleto:
    def test_enrolar_y_reconocer(self, video_lino_path, tmp_path,
                                  detector, recognizer, spoofer):
        import faiss
        from scripts.registro_unificado import RegistroBiometrico
        from unittest.mock import patch

        with patch.object(RegistroBiometrico, '__init__', lambda self, **kw: None):
            reg = RegistroBiometrico()
            reg.db_path = tmp_path / "db.pkl"
            reg.max_muestras = 5
            reg.umbral_blur = 60
            reg.umbral_brillo = 20
            reg.umbral_liveness = 0.90
            reg.db_embeddings = {}
            reg.detector = detector
            reg.recognizer = recognizer
            reg.spoofer = spoofer
            resultado = reg.enrolar_usuario_headless("Lino", video_lino_path)

        if not resultado["exito"]:
            pytest.skip("No se pudo enrolar a Lino con el video")

        cap = cv2.VideoCapture(video_lino_path)
        found = False
        for _ in range(30):
            ret, frame = cap.read()
            if not ret:
                break
            rostros = detector.detect(frame)
            if rostros:
                from core.utils_facial import aplicar_clahe
                f_eq = aplicar_clahe(frame)
                emb = recognizer.get_normalized_embedding(f_eq, rostros[0].landmarks)
                emb_lino = np.array(resultado["embedding"], dtype="float32")
                sim = np.dot(emb, emb_lino)
                if sim > 0.35:
                    found = True
                    break
        cap.release()
        assert found, "No se reconocio a Lino en su propio video"
