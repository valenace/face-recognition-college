"""Utilidades de dibujo para los videos de demostración."""

import unicodedata

import cv2
import numpy as np

FUENTE = cv2.FONT_HERSHEY_SIMPLEX


def sin_acentos(texto):
    """OpenCV solo dibuja ASCII: convierte tildes y eñes."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c)
    )


def texto(frame, cadena, origen, escala=0.5, color=(240, 240, 240), grosor=1):
    """Texto simple, sin sombra (estilo por defecto de OpenCV)."""
    cv2.putText(frame, cadena, origen, FUENTE, escala, color, grosor, cv2.LINE_AA)


def check(frame, x, y, tam, color=(80, 220, 120), grosor=2):
    """Marca de verificación dibujada con líneas (sin caracteres especiales)."""
    puntos = np.array([
        [x + int(tam * 0.12), y + int(tam * 0.55)],
        [x + int(tam * 0.42), y + int(tam * 0.85)],
        [x + int(tam * 0.90), y + int(tam * 0.20)],
    ], dtype=np.int32)
    cv2.polylines(frame, [puntos], False, color, grosor, cv2.LINE_AA)


def caja(frame, x, y, ancho, alto, color, grosor=1):
    cv2.rectangle(frame, (x, y), (x + ancho, y + alto), color, grosor)


def rellenar(frame, x, y, ancho, alto, color):
    cv2.rectangle(frame, (x, y), (x + ancho, y + alto), color, -1)


def franja(frame, x, y, ancho, alto, color, alpha=0.85):
    """Rectángulo semitransparente."""
    capa = frame.copy()
    cv2.rectangle(capa, (x, y), (x + ancho, y + alto), color, -1)
    cv2.addWeighted(capa, alpha, frame, 1 - alpha, 0, frame)


def crear_writer(ruta, fps, ancho, alto):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    return cv2.VideoWriter(
        str(ruta), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ancho, alto)
    )


def formato_tiempo(segundos):
    segundos = max(0, int(segundos))
    return f"{segundos // 60:02d}:{segundos % 60:02d}"
