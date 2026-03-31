"""
registro.py — Sistema de Registro Biométrico con UniFace v2
"""

import cv2
import pickle
import numpy as np
import tempfile
from pathlib import Path
from uniface import create_detector, create_recognizer
from utils_facial import aplicar_clahe, es_imagen_borrosa, alinear_rostro

# ============================================================
# ⚙️ CONFIGURACIÓN
# ============================================================
DATASET_PATH = Path("dataset_pro")
DB_FILE = Path("database_embeddings.pkl")

MAX_FOTOS = 50              
FRAMES_ENTRE_FOTOS = 8      
UMBRAL_CALIDAD_BLUR = 60    
UMBRAL_BRILLO_MIN = 20

# ============================================================
# 🚀 INICIALIZACIÓN DE MOTORES
# ============================================================
print("=" * 60)
print("  SISTEMA DE REGISTRO BIOMÉTRICO — UniFace v2 (ONNX)")
print("=" * 60)

print("\n[INFO] Cargando modelos de UniFace...")
print("  → RetinaFace (Detector de rostros)")
print("  → ArcFace (Extractor de embeddings 512-D)")
detector = create_detector('retinaface')
recognizer = create_recognizer('arcface')
print("[OK] Motores cargados exitosamente.\n")


# ============================================================
# 📋 DATOS DEL ESTUDIANTE
# ============================================================
nombre = input("Ingrese el nombre de la persona a registrar: ").strip()
if not nombre:
    print("❌ Error: El nombre no puede estar vacío.")
    exit()

path_estudiante = DATASET_PATH / nombre
path_estudiante.mkdir(parents=True, exist_ok=True)
print(f"📂 Directorio de trabajo: {path_estudiante}")


# ============================================================
# 📷 FASE 1: CAPTURA INTERACTIVA
# ============================================================
WINDOW_NAME = "REGISTRO BIOMETRICO - UniFace v2"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: No se pudo acceder a la cámara.")
    exit()

count = 0
frames_counter = 0

print("\n--- INSTRUCCIONES ---")
print("  1. Mantén presionada la tecla 's'.")
print("  2. Gira tu cabeza lentamente dibujando un círculo.")
print("  3. Presiona 'q' para salir en cualquier momento.")
print("---------------------\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Advertencia: Pérdida de señal de la cámara.")
            break

        display = frame.copy()
        frames_counter += 1

        rostros = detector.detect(frame)

        rostro_listo = None
        mensaje_estado = "Buscando rostro..."
        color_mensaje = (0, 165, 255)

        if rostros:
            rostro = rostros[0]
            landmarks = rostro.landmarks

            if landmarks is not None and len(landmarks) >= 2:
                rostro_alineado = alinear_rostro(frame, landmarks)

                if rostro_alineado is not None:
                    if es_imagen_borrosa(rostro_alineado, umbral=UMBRAL_CALIDAD_BLUR):
                        mensaje_estado = "FOTO BORROSA - Quietud por favor"
                        color_mensaje = (0, 0, 255)
                        rostro_listo = None
                    else:
                        gray = cv2.cvtColor(rostro_alineado, cv2.COLOR_BGR2GRAY)
                        brillo = np.mean(gray)

                        if brillo < UMBRAL_BRILLO_MIN:
                            mensaje_estado = "MUY OSCURO - Busca mejor luz"
                            color_mensaje = (0, 0, 255)
                            rostro_listo = None
                        else:
                            mensaje_estado = "CALIDAD ÓPTIMA - Presiona 's'"
                            color_mensaje = (0, 255, 0)
                            rostro_listo = rostro_alineado

                    h_crop, w_crop = rostro_alineado.shape[:2]
                    if 10 + h_crop < display.shape[0] and 10 + w_crop < display.shape[1]:
                        display[10:10 + h_crop, 10:10 + w_crop] = rostro_alineado
                        cv2.rectangle(display, (10, 10), (10 + w_crop, 10 + h_crop), color_mensaje, 2)

                x1, y1, x2, y2 = map(int, rostro.bbox)
                cv2.rectangle(display, (x1, y1), (x2, y2), color_mensaje, 2)

        if count < MAX_FOTOS:
            progreso = int((count / MAX_FOTOS) * 100)
            instruccion = f"Gira la cabeza en circulos ({progreso}%)"
        else:
            instruccion = "¡REGISTRO COMPLETADO! (Presiona 'q')"
            color_mensaje = (0, 255, 255)

        h_frame = display.shape[0]
        
        ancho_barra = int((count / MAX_FOTOS) * display.shape[1])
        cv2.rectangle(display, (0, h_frame - 15), (ancho_barra, h_frame), (0, 255, 0), -1)

        cv2.putText(display, f"FOTOS: {count}/{MAX_FOTOS}", (20, h_frame - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, instruccion, (20, h_frame - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display, mensaje_estado, (150, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_mensaje, 2)

        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and rostro_listo is not None and count < MAX_FOTOS:
            if frames_counter % FRAMES_ENTRE_FOTOS == 0:
                filename = path_estudiante / f"{nombre}_{count}.jpg"
                cv2.imwrite(str(filename), rostro_listo)
                count += 1
                print(f"  ✅ Foto {count}/{MAX_FOTOS} guardada. ({instruccion})")

        if key == ord('q'):
            break

finally:
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print(f"\n--- CAPTURA FINALIZADA: {count} fotos ---")

if count < 10:
    print("⚠️ ADVERTENCIA: Muy pocas fotos. Se recomienda repetir el registro.\n")

# ============================================================
# 🧠 FASE 2: GENERACIÓN DE EMBEDDINGS Y BASE DE DATOS
# ============================================================
print("=" * 60)
print("  FASE 2: GENERANDO EMBEDDINGS (ArcFace 512-D)")
print("=" * 60)

db_embeddings = {}
if DB_FILE.exists():
    with open(DB_FILE, "rb") as f:
        db_embeddings = pickle.load(f)
    print(f"[INFO] Base de datos existente cargada: {len(db_embeddings)} personas.")

vectores_estudiante = []

extensiones_validas = {'.jpg', '.jpeg', '.png'}
archivos_validos = [p for p in path_estudiante.iterdir() if p.suffix.lower() in extensiones_validas]

for i, ruta_imagen in enumerate(sorted(archivos_validos)):
    img = cv2.imread(str(ruta_imagen))
    
    if img is None or img.size == 0:
        print(f"⚠️ Advertencia: No se pudo leer o está corrupta -> {ruta_imagen.name}")
        continue

    img_limpia = aplicar_clahe(img)
    rostros = detector.detect(img_limpia)

    if rostros:
        vector = recognizer.get_normalized_embedding(img_limpia, rostros[0].landmarks)
        vectores_estudiante.append(vector)

    if (i + 1) % 10 == 0:
        print(f"  → Procesadas {i + 1}/{len(archivos_validos)} fotos...")

if vectores_estudiante:
    centroide = np.mean(vectores_estudiante, axis=0)
    centroide_normalizado = centroide / np.linalg.norm(centroide)
    db_embeddings[nombre] = centroide_normalizado
    
    ruta_temporal = DB_FILE.with_suffix('.pkl.tmp')
    try:
        with open(ruta_temporal, "wb") as f:
            pickle.dump(db_embeddings, f)
        
        ruta_temporal.replace(DB_FILE)
        print(f"\n  ✅ Base de datos protegida y actualizada con éxito.")
    except Exception as e:
        print(f"\n  ❌ Error crítico al guardar la base de datos: {e}")
        if ruta_temporal.exists():
            ruta_temporal.unlink()
else:
    print(f"\n  ❌ No se pudieron extraer embeddings para {nombre}.")

with open(DB_FILE, "wb") as f:
    pickle.dump(db_embeddings, f)

print(f"\n{'=' * 60}")
print(f"  BASE DE DATOS ACTUALIZADA: {DB_FILE}")
print(f"  Total de personas registradas: {len(db_embeddings)}")
for n in db_embeddings:
    print(f"    → {n}")
print(f"{'=' * 60}")
print("\n✨ ¡Registro completado! Ya puedes ejecutar reconocimiento.py")
