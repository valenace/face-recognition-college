import cv2
import numpy as np


def aplicar_clahe(imagen):
    if imagen is None or imagen.size == 0:
        return imagen
    try:
        lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_ecualizado = clahe.apply(l)
        lab_mejorado = cv2.merge((l_ecualizado, a, b))
        return cv2.cvtColor(lab_mejorado, cv2.COLOR_LAB2BGR)
    except Exception:
        return imagen


def es_imagen_borrosa(imagen, umbral=80):
    if imagen is None or imagen.size == 0:
        return True
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score < umbral


def alinear_rostro(frame, landmarks_5pts, target_size=(224, 224)):
    if landmarks_5pts is None or len(landmarks_5pts) < 2:
        return None

    ojo_izq = landmarks_5pts[0]
    ojo_der = landmarks_5pts[1]

    dx = float(ojo_der[0] - ojo_izq[0])
    dy = float(ojo_der[1] - ojo_izq[1])

    angulo = np.degrees(np.arctan2(dy, dx))
    centro_ojos = (
        int((ojo_izq[0] + ojo_der[0]) / 2),
        int((ojo_izq[1] + ojo_der[1]) / 2)
    )

    dist_ojos = np.sqrt(dx**2 + dy**2)
    if dist_ojos < 10:
        return None

    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D(centro_ojos, angulo, 1.0)
    frame_rotado = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_CUBIC)

    ancho_crop = int(dist_ojos * 4.5)
    alto_crop = int(dist_ojos * 5.5)

    start_x = max(0, int(centro_ojos[0] - ancho_crop // 2))
    start_y = max(0, int(centro_ojos[1] - alto_crop * 0.4))
    end_x = min(w, start_x + ancho_crop)
    end_y = min(h, start_y + alto_crop)

    rostro_alineado = frame_rotado[start_y:end_y, start_x:end_x]

    if rostro_alineado.size > 0 and rostro_alineado.shape[0] > 50:
        try:
            return cv2.resize(rostro_alineado, target_size)
        except Exception:
            return None
    return None
