import pytest
import numpy as np
import cv2


@pytest.mark.integration
@pytest.mark.slow
class TestDetectorRetinaFace:
    def test_detecta_rostro_jorge(self, detector, sample_face_jorge):
        rostros = detector.detect(sample_face_jorge)
        assert len(rostros) >= 1

    def test_detecta_rostro_lino(self, detector, sample_face_lino):
        rostros = detector.detect(sample_face_lino)
        assert len(rostros) >= 1

    def test_detecta_rostro_valu(self, detector, sample_face_valu):
        rostros = detector.detect(sample_face_valu)
        assert len(rostros) >= 1

    def test_sin_rostro_en_ruido(self, detector, synthetic_noise):
        rostros = detector.detect(synthetic_noise)
        assert len(rostros) == 0

    def test_sin_rostro_en_negro(self, detector, synthetic_blank):
        rostros = detector.detect(synthetic_blank)
        assert len(rostros) == 0

    def test_bbox_formato(self, detector, sample_face_jorge):
        rostros = detector.detect(sample_face_jorge)
        if rostros:
            r = rostros[0]
            assert r.bbox.shape == (4,)
            x1, y1, x2, y2 = r.bbox
            assert x2 > x1
            assert y2 > y1

    def test_landmarks_formato(self, detector, sample_face_jorge):
        rostros = detector.detect(sample_face_jorge)
        if rostros:
            r = rostros[0]
            assert r.landmarks is not None
            assert r.landmarks.shape == (5, 2)

    def test_confianza_rango(self, detector, sample_face_jorge):
        rostros = detector.detect(sample_face_jorge)
        if rostros:
            for r in rostros:
                assert 0.0 <= r.confidence <= 1.0

    def test_detecta_en_frame_grande(self, detector, frame_with_face):
        rostros = detector.detect(frame_with_face)
        assert len(rostros) >= 1
        r = rostros[0]
        x1, y1, x2, y2 = map(int, r.bbox)
        assert (x2 - x1) > 60
        assert (y2 - y1) > 60


@pytest.mark.integration
@pytest.mark.slow
class TestRecognizerArcFace:
    def test_embedding_dim_512(self, recognizer, detector, sample_face_lino):
        rostros = detector.detect(sample_face_lino)
        if not rostros:
            pytest.skip("No se detecto rostro en imagen de Lino")
        emb = recognizer.get_normalized_embedding(sample_face_lino, rostros[0].landmarks)
        assert emb.shape == (512,)

    def test_embedding_normalizado(self, recognizer, detector, sample_face_lino):
        rostros = detector.detect(sample_face_lino)
        if not rostros:
            pytest.skip("No se detecto rostro en imagen de Lino")
        emb = recognizer.get_normalized_embedding(sample_face_lino, rostros[0].landmarks)
        norma = np.linalg.norm(emb)
        assert norma == pytest.approx(1.0, abs=1e-5)

    def test_misma_persona_alta_similitud(self, recognizer, detector):
        img1 = cv2.imread("data/dataset_pro/Lino313/video_lino_0.jpg")
        img2 = cv2.imread("data/dataset_pro/Lino313/video_lino_5.jpg")
        r1 = detector.detect(img1)
        r2 = detector.detect(img2)
        if not r1 or not r2:
            pytest.skip("No se detectaron rostros en ambas imagenes")
        emb1 = recognizer.get_normalized_embedding(img1, r1[0].landmarks)
        emb2 = recognizer.get_normalized_embedding(img2, r2[0].landmarks)
        sim = np.dot(emb1, emb2)
        assert sim > 0.3, f"Similitud misma persona muy baja: {sim:.4f}"

    def test_diferente_persona_baja_similitud(self, recognizer, detector,
                                               sample_face_jorge, sample_face_lino):
        r_j = detector.detect(sample_face_jorge)
        r_l = detector.detect(sample_face_lino)
        if not r_j or not r_l:
            pytest.skip("No se detectaron rostros")
        emb_j = recognizer.get_normalized_embedding(sample_face_jorge, r_j[0].landmarks)
        emb_l = recognizer.get_normalized_embedding(sample_face_lino, r_l[0].landmarks)
        sim = np.dot(emb_j, emb_l)
        assert sim < 0.5, f"Similitud diferente persona muy alta: {sim:.4f}"

    def test_estabilidad_mismo_crop(self, recognizer, detector, sample_face_lino):
        rostros = detector.detect(sample_face_lino)
        if not rostros:
            pytest.skip("No se detecto rostro")
        emb1 = recognizer.get_normalized_embedding(sample_face_lino, rostros[0].landmarks)
        emb2 = recognizer.get_normalized_embedding(sample_face_lino, rostros[0].landmarks)
        np.testing.assert_array_almost_equal(emb1, emb2, decimal=5)

    def test_embedding_dtype_float32(self, recognizer, detector, sample_face_lino):
        rostros = detector.detect(sample_face_lino)
        if not rostros:
            pytest.skip("No se detecto rostro")
        emb = recognizer.get_normalized_embedding(sample_face_lino, rostros[0].landmarks)
        assert emb.dtype == np.float32


@pytest.mark.integration
@pytest.mark.slow
class TestSpooferMiniFASNet:
    def test_retorna_resultado(self, spoofer, detector, frame_with_face):
        rostros = detector.detect(frame_with_face)
        if not rostros:
            pytest.skip("No se detecto rostro en frame")
        resultado = spoofer.predict(frame_with_face, rostros[0].bbox)
        assert hasattr(resultado, "is_real")
        assert hasattr(resultado, "confidence")

    def test_confianza_rango(self, spoofer, detector, frame_with_face):
        rostros = detector.detect(frame_with_face)
        if not rostros:
            pytest.skip("No se detecto rostro en frame")
        resultado = spoofer.predict(frame_with_face, rostros[0].bbox)
        assert 0.0 <= resultado.confidence <= 1.0

    def test_is_real_es_bool(self, spoofer, detector, frame_with_face):
        rostros = detector.detect(frame_with_face)
        if not rostros:
            pytest.skip("No se detecto rostro en frame")
        resultado = spoofer.predict(frame_with_face, rostros[0].bbox)
        assert isinstance(resultado.is_real, bool)

    def test_resultado_es_inmutable(self, spoofer, detector, frame_with_face):
        rostros = detector.detect(frame_with_face)
        if not rostros:
            pytest.skip("No se detecto rostro en frame")
        resultado = spoofer.predict(frame_with_face, rostros[0].bbox)
        with pytest.raises(AttributeError):
            resultado.is_real = False
