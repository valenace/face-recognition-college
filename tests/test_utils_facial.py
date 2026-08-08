import pytest
import numpy as np
import cv2
from core.utils_facial import aplicar_clahe, es_imagen_borrosa, alinear_rostro


@pytest.mark.unit
class TestAplicarClahe:
    def test_mejora_contraste(self, synthetic_face):
        resultado = aplicar_clahe(synthetic_face)
        lab_orig = cv2.cvtColor(synthetic_face, cv2.COLOR_BGR2LAB)
        lab_res = cv2.cvtColor(resultado, cv2.COLOR_BGR2LAB)
        var_orig = np.var(lab_orig[:, :, 0])
        var_res = np.var(lab_res[:, :, 0])
        assert var_res >= var_orig

    def test_con_none(self):
        assert aplicar_clahe(None) is None

    def test_con_imagen_vacia(self):
        img = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        resultado = aplicar_clahe(img)
        assert resultado.size == 0

    def test_preserva_dimensiones(self, synthetic_face):
        resultado = aplicar_clahe(synthetic_face)
        assert resultado.shape == synthetic_face.shape

    def test_preserva_dtype(self, synthetic_face):
        resultado = aplicar_clahe(synthetic_face)
        assert resultado.dtype == synthetic_face.dtype

    def test_imagen_gris_puro(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        resultado = aplicar_clahe(img)
        assert resultado is not None
        assert resultado.shape == (100, 100, 3)

    def test_imagen_muy_oscura(self, synthetic_dark):
        resultado = aplicar_clahe(synthetic_dark)
        assert resultado is not None

    def test_imagen_muy_clara(self, synthetic_bright):
        resultado = aplicar_clahe(synthetic_bright)
        assert resultado is not None

    def test_idempotente_approx(self, synthetic_face):
        r1 = aplicar_clahe(synthetic_face)
        r2 = aplicar_clahe(r1)
        diff = np.mean(np.abs(r1.astype(float) - r2.astype(float)))
        assert diff < 25


@pytest.mark.unit
class TestEsImagenBorrosa:
    def test_detecta_borrosa(self, synthetic_blurry):
        assert es_imagen_borrosa(synthetic_blurry, umbral=80) == True

    def test_detecta_nitida(self, synthetic_sharp):
        assert es_imagen_borrosa(synthetic_sharp, umbral=80) == False

    def test_con_none(self):
        assert es_imagen_borrosa(None) is True

    def test_con_imagen_vacia(self):
        img = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        assert es_imagen_borrosa(img) is True

    def test_umbral_alto_detecta_mas(self):
        img_semi_blur = np.random.randint(80, 180, (224, 224, 3), dtype=np.uint8)
        img_semi_blur = cv2.GaussianBlur(img_semi_blur, (11, 11), 5)
        resultado_alto = es_imagen_borrosa(img_semi_blur, umbral=500)
        resultado_bajo = es_imagen_borrosa(img_semi_blur, umbral=10)
        assert bool(resultado_alto) or not bool(resultado_bajo)

    def test_umbral_cero(self, synthetic_sharp):
        assert es_imagen_borrosa(synthetic_sharp, umbral=0) == False

    def test_imagen_negra(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        resultado = es_imagen_borrosa(img, umbral=80)
        assert isinstance(resultado, (bool, np.bool_))


@pytest.mark.unit
class TestAlinearRostro:
    def test_con_landmarks_validos(self, synthetic_face, fake_landmarks_5pts):
        resultado = alinear_rostro(synthetic_face, fake_landmarks_5pts)
        assert resultado is not None
        assert resultado.shape == (224, 224, 3)

    def test_con_landmarks_none(self, synthetic_face):
        assert alinear_rostro(synthetic_face, None) is None

    def test_con_landmarks_insuficientes(self, synthetic_face):
        landmarks = np.array([[10.0, 10.0]], dtype=np.float32)
        assert alinear_rostro(synthetic_face, landmarks) is None

    def test_ojos_muy_cercanos(self, synthetic_face):
        landmarks = np.array([
            [100.0, 100.0],
            [102.0, 100.0],
            [101.0, 110.0],
            [98.0, 120.0],
            [104.0, 120.0],
        ], dtype=np.float32)
        assert alinear_rostro(synthetic_face, landmarks) is None

    def test_landmarks_duplicados(self, synthetic_face):
        landmarks = np.array([
            [50.0, 50.0],
            [50.0, 50.0],
            [50.0, 50.0],
            [50.0, 50.0],
            [50.0, 50.0],
        ], dtype=np.float32)
        assert alinear_rostro(synthetic_face, landmarks) is None

    def test_preserva_dtype(self, synthetic_face, fake_landmarks_5pts):
        resultado = alinear_rostro(synthetic_face, fake_landmarks_5pts)
        if resultado is not None:
            assert resultado.dtype == np.uint8

    def test_frame_pequeno(self):
        frame = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        landmarks = np.array([
            [15.0, 15.0],
            [35.0, 15.0],
            [25.0, 25.0],
            [18.0, 35.0],
            [32.0, 35.0],
        ], dtype=np.float32)
        resultado = alinear_rostro(frame, landmarks)
        assert resultado is None or resultado.shape == (224, 224, 3)

    def test_landmarks_en_borde(self):
        frame = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        landmarks = np.array([
            [5.0, 5.0],
            [25.0, 5.0],
            [15.0, 15.0],
            [8.0, 25.0],
            [22.0, 25.0],
        ], dtype=np.float32)
        resultado = alinear_rostro(frame, landmarks)
        assert resultado is None or resultado.shape == (224, 224, 3)

    def test_ojos_con_angulo_inclinado(self):
        frame = np.random.randint(80, 180, (400, 400, 3), dtype=np.uint8)
        landmarks = np.array([
            [150.0, 180.0],
            [250.0, 160.0],
            [200.0, 210.0],
            [170.0, 250.0],
            [230.0, 240.0],
        ], dtype=np.float32)
        resultado = alinear_rostro(frame, landmarks)
        assert resultado is not None
        assert resultado.shape == (224, 224, 3)
