import cv2
import pickle
import numpy as np
from pathlib import Path
from uniface import create_detector, create_recognizer
from core.utils_facial import aplicar_clahe, es_imagen_borrosa, alinear_rostro

DATASET_PATH = Path("data/dataset_pro")
DB_FILE = Path("data/database_embeddings.pkl")

MAX_FOTOS = 50
FRAMES_ENTRE_FOTOS = 8
UMBRAL_CALIDAD_BLUR = 60
UMBRAL_BRILLO_MIN = 20

print("Cargando modelos...")
detector = create_detector('retinaface')
recognizer = create_recognizer('arcface')

nombre = input("Nombre de la persona a registrar: ").strip()
if not nombre:
    print("Error: el nombre no puede estar vacio.")
    exit()

ruta_video = input("Ruta del archivo de video: ").strip()
if not Path(ruta_video).exists():
    print("Error: el archivo de video no existe.")
    exit()

path_estudiante = DATASET_PATH / nombre
path_estudiante.mkdir(parents=True, exist_ok=True)
print(f"Directorio: {path_estudiante}")

WINDOW_NAME = "Registro desde Video"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

cap = cv2.VideoCapture(ruta_video)
if not cap.isOpened():
    print("Error: no se pudo abrir el video.")
    exit()

count = 0
frames_counter = 0

print("Extrayendo frames... 'q' para cancelar.\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fin del video.")
            break

        frames_counter += 1
        if frames_counter % FRAMES_ENTRE_FOTOS != 0:
            continue

        display = frame.copy()
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
                        mensaje_estado = "BORROSA - descartando"
                        color_mensaje = (0, 0, 255)
                    else:
                        gray = cv2.cvtColor(rostro_alineado, cv2.COLOR_BGR2GRAY)
                        brillo = np.mean(gray)
                        if brillo < UMBRAL_BRILLO_MIN:
                            mensaje_estado = "MUY OSCURO - descartando"
                            color_mensaje = (0, 0, 255)
                        else:
                            mensaje_estado = "OK - capturando"
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
            instruccion = f"Extrayendo... ({progreso}%)"
        else:
            instruccion = "Extraccion completada"
            color_mensaje = (0, 255, 255)

        h_frame = display.shape[0]
        ancho_barra = int((count / MAX_FOTOS) * display.shape[1])
        cv2.rectangle(display, (0, h_frame - 15), (ancho_barra, h_frame), (0, 255, 0), -1)
        cv2.putText(display, f"Fotos: {count}/{MAX_FOTOS}", (20, h_frame - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, instruccion, (20, h_frame - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display, mensaje_estado, (150, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_mensaje, 2)

        cv2.imshow(WINDOW_NAME, display)

        if rostro_listo is not None and count < MAX_FOTOS:
            filename = path_estudiante / f"{nombre}_{count}.jpg"
            cv2.imwrite(str(filename), rostro_listo)
            count += 1
            print(f"  Foto {count}/{MAX_FOTOS}")

        if count >= MAX_FOTOS:
            print("Maximo de fotos alcanzado.")
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Cancelado por el usuario.")
            break

finally:
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print(f"\nExtraccion finalizada: {count} fotos")

if count < 10:
    print("Advertencia: pocas fotos. La calidad del reconocimiento puede ser baja.")

# --- Generacion de embeddings ---
print("\nGenerando embeddings...")

db_embeddings = {}
if DB_FILE.exists():
    with open(DB_FILE, "rb") as f:
        db_embeddings = pickle.load(f)
    print(f"Base de datos cargada: {len(db_embeddings)} personas")

vectores_estudiante = []
extensiones_validas = {'.jpg', '.jpeg', '.png'}
archivos_validos = [p for p in path_estudiante.iterdir() if p.suffix.lower() in extensiones_validas]

for i, ruta_imagen in enumerate(sorted(archivos_validos)):
    img = cv2.imread(str(ruta_imagen))
    if img is None or img.size == 0:
        print(f"No se pudo leer: {ruta_imagen.name}")
        continue

    img_limpia = aplicar_clahe(img)
    rostros = detector.detect(img_limpia)
    if rostros:
        vector = recognizer.get_normalized_embedding(img_limpia, rostros[0].landmarks)
        vectores_estudiante.append(vector)

    if (i + 1) % 10 == 0:
        print(f"  {i + 1}/{len(archivos_validos)} fotos procesadas...")

if vectores_estudiante:
    centroide = np.mean(vectores_estudiante, axis=0)
    centroide_normalizado = centroide / np.linalg.norm(centroide)
    db_embeddings[nombre] = centroide_normalizado

    ruta_temporal = DB_FILE.with_suffix('.pkl.tmp')
    try:
        with open(ruta_temporal, "wb") as f:
            pickle.dump(db_embeddings, f)
        ruta_temporal.replace(DB_FILE)
        print("Base de datos actualizada.")
    except Exception as e:
        print(f"Error al guardar: {e}")
        if ruta_temporal.exists():
            ruta_temporal.unlink()
else:
    print(f"No se pudieron extraer embeddings para {nombre}.")

print(f"\nTotal registradas: {len(db_embeddings)}")
for n in db_embeddings:
    print(f"  {n}")