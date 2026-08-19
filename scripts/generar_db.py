"""
generar_db.py — Procesamiento Masivo del Dataset (CV College v2)

Lee todas las carpetas dentro de DATASET_PATH (frames extraídos por
registro_video.py), genera el embedding centroide de cada persona
y los guarda en database_embeddings.pkl.

Uso interactivo:
  python scripts/generar_db.py

Importable:
  from scripts.generar_db import generar_db
  db = generar_db(dataset_path=..., db_file=...)
"""

import cv2
import os
import pickle
import numpy as np
import sys

# Inyección al PYTHONPATH para encontrar la carpeta 'core'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uniface import create_detector, create_recognizer
from core.utils_facial import aplicar_clahe

# ============================================================
# ⚙️ CONFIGURACIÓN
# ============================================================
DATASET_PATH = "data/dataset_cv/frames"
DB_FILE = "data/database_embeddings.pkl"


def generar_db(dataset_path=DATASET_PATH, db_file=DB_FILE):
    """Recorre las subcarpetas de frames y genera database_embeddings.pkl."""

    print("=" * 60)
    print("  GENERACIÓN DE BASE DE DATOS BIOMÉTRICA — CV College v2")
    print("=" * 60)

    if not os.path.exists(dataset_path):
        print(f"Error: No existe la carpeta '{dataset_path}'")
        return None

    print("\n[INFO] Cargando motores CV College (ONNX Runtime)...")
    detector = create_detector("retinaface")
    recognizer = create_recognizer("arcface")
    print("[OK] Motores listos.\n")

    db_embeddings = {}

    # ── RECORRIDO DEL DATASET ──────────────────────────────
    carpetas_personas = sorted(os.listdir(dataset_path))
    print(f"[INFO] Se encontraron {len(carpetas_personas)} personas potenciales.\n")

    for persona in carpetas_personas:
        ruta_persona = os.path.join(dataset_path, persona)
        if not os.path.isdir(ruta_persona):
            continue

        print(f"Procesando: {persona}...")
        vectores_temporales = []

        archivos_imagen = sorted(
            [
                f
                for f in os.listdir(ruta_persona)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )

        for i, archivo in enumerate(archivos_imagen):
            ruta_imagen = os.path.join(ruta_persona, archivo)

            img = cv2.imread(ruta_imagen)
            if img is None:
                continue

            img_limpia = aplicar_clahe(img)
            rostros = detector.detect(img_limpia)

            if rostros:
                rostro = rostros[0]
                vector = recognizer.get_normalized_embedding(img_limpia, rostro.landmarks)
                vectores_temporales.append(vector)

        # 🧠 CENTROIDE
        if vectores_temporales:
            centroide = np.mean(vectores_temporales, axis=0)
            centroide_final = centroide / np.linalg.norm(centroide)

            db_embeddings[persona] = centroide_final
            print(
                f"   Éxito! {len(vectores_temporales)} fotos procesadas. Centroide generado."
            )
        else:
            print(f"   Advertencia: No se encontraron rostros válidos para {persona}.")

    # 💾 GUARDAR
    with open(db_file, "wb") as f:
        pickle.dump(db_embeddings, f)

    print("\n" + "=" * 60)
    print(f"  [OK] Base de datos guardada en: {db_file}")
    print(f"  Total de identidades en el 'cerebro': {len(db_embeddings)}")
    for n in db_embeddings:
        print(f"    → {n}")
    print("=" * 60)

    return db_embeddings


if __name__ == "__main__":
    generar_db()