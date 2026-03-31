"""
generar_db.py — Procesamiento Masivo del Dataset (UniFace v2)
"""

import cv2
import os
import pickle
import numpy as np
from uniface import create_detector, create_recognizer
from utils_facial import aplicar_clahe

# ============================================================
# ⚙️ CONFIGURACIÓN
# ============================================================
DATASET_PATH = "dataset_pro"
DB_FILE = "database_embeddings.pkl"

# ============================================================
# 🚀 INICIALIZACIÓN
# ============================================================
print("=" * 60)
print("  GENERACIÓN DE BASE DE DATOS BIOMÉTRICA — UniFace v2")
print("=" * 60)

if not os.path.exists(DATASET_PATH):
    print(f"❌ Error: No existe la carpeta '{DATASET_PATH}'")
    exit()

print("\n[INFO] Cargando motores UniFace (ONNX Runtime)...")
detector = create_detector('retinaface')
recognizer = create_recognizer('arcface')
print("[OK] Motores listos.\n")

db_embeddings = {}

# ============================================================
# 📂 RECORRIDO DEL DATASET
# ============================================================
carpetas_personas = sorted(os.listdir(DATASET_PATH))
print(f"[INFO] Se encontraron {len(carpetas_personas)} personas potenciales.\n")

for persona in carpetas_personas:
    ruta_persona = os.path.join(DATASET_PATH, persona)
    if not os.path.isdir(ruta_persona):
        continue
        
    print(f"👉 Procesando: {persona}...")
    vectores_temporales = []
    
    archivos_imagen = sorted([f for f in os.listdir(ruta_persona) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
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
    
    # ============================================================
    # 🧠 CENTROIDE
    # ============================================================
    if vectores_temporales:
        centroide = np.mean(vectores_temporales, axis=0)
        centroide_final = centroide / np.linalg.norm(centroide)
        
        db_embeddings[persona] = centroide_final
        print(f"   ✅ Éxito! {len(vectores_temporales)} fotos procesadas. Centroide generado.")
    else:
        print(f"   ⚠️ Advertencia: No se encontraron rostros válidos para {persona}.")

# ============================================================
# 💾 GUARDAR BASE DE DATOS
# ============================================================
with open(DB_FILE, "wb") as f:
    pickle.dump(db_embeddings, f)
    
print("\n" + "=" * 60)
print(f"  [OK] Base de datos guardada en: {DB_FILE}")
print(f"  Total de identidades en el 'cerebro': {len(db_embeddings)}")
for n in db_embeddings:
    print(f"    → {n}")
print("=" * 60)
print("\n✨ Ahora ya puedes ejecutar: python3 reconocimiento.py")
