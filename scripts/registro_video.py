import cv2
import numpy as np
from pathlib import Path
import sys
import os

# Inyección al PYTHONPATH para encontrar la carpeta 'core'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uniface import create_detector, create_recognizer
from uniface.spoofing import create_spoofer
from core.utils_facial import aplicar_clahe, es_imagen_borrosa, alinear_rostro, embedding_facial

RAIZ = Path(__file__).resolve().parents[1]
DATASET_PATH = RAIZ / "data/dataset_cv/frames"
DB_FILE = RAIZ / "data/database_embeddings.pkl"

MAX_FOTOS = 50
MIN_FOTOS = 15
CANDIDATOS_OBJETIVO = 300
UMBRAL_CALIDAD_BLUR = 60
UMBRAL_BRILLO_MIN = 20
UMBRAL_LIVENESS_MIN = 0.90
UMBRAL_POSE_REPETIDA = 0.95


def _mejor_rostro(rostros):
    return max(rostros, key=lambda r: (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1]))


def registrar_desde_video(
    nombre,
    ruta_video,
    db_file=DB_FILE,
    dataset_path=DATASET_PATH,
    check_liveness=False,
    mostrar=True,
):
    """
    Extrae hasta MAX_FOTOS frames alineados (una sola cara por frame, muestreo uniforme
    en todo el video), genera el centroide normalizado y lo guarda en la DB.

    Returns:
        dict: la DB de embeddings resultante, o None si falló o hay menos de MIN_FOTOS.
    """
    ruta_video = Path(ruta_video)
    if not ruta_video.exists():
        print("Error: el archivo de video no existe.")
        return None

    cap = cv2.VideoCapture(str(ruta_video))
    if not cap.isOpened():
        print("Error: no se pudo abrir el video.")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step_value = max(1, total_frames // CANDIDATOS_OBJETIVO) if total_frames > 0 else 1

    print("Cargando modelos...")
    detector = create_detector("retinaface")
    recognizer = create_recognizer("arcface")
    spoofer = create_spoofer() if check_liveness else None

    path_estudiante = Path(dataset_path) / nombre
    path_estudiante.mkdir(parents=True, exist_ok=True)
    for jpg_previo in path_estudiante.glob("*.jpg"):
        jpg_previo.unlink()
    print(f"Directorio: {path_estudiante} (limpiado)")
    print(f"Paso de muestreo: {step_value} frames | Liveness: {check_liveness}")

    window_name = "Registro desde Video"
    if mostrar:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    count = 0
    frames_counter = 0
    vectores_cap = []

    try:
        while count < MAX_FOTOS:
            ret, frame = cap.read()
            if not ret:
                print("Fin del video.")
                break

            frames_counter += 1
            if frames_counter % step_value != 0:
                continue

            rostros = detector.detect(frame)
            if len(rostros) != 1:
                continue

            rostro = rostros[0]
            landmarks = rostro.landmarks
            if landmarks is None or len(landmarks) < 2:
                continue

            rostro_alineado = alinear_rostro(frame, landmarks)
            if rostro_alineado is None:
                continue
            if es_imagen_borrosa(rostro_alineado, umbral=UMBRAL_CALIDAD_BLUR):
                continue

            gray = cv2.cvtColor(rostro_alineado, cv2.COLOR_BGR2GRAY)
            if np.mean(gray) < UMBRAL_BRILLO_MIN:
                continue

            if check_liveness:
                try:
                    resultado_spoof = spoofer.predict(frame, rostro.bbox)
                    if not resultado_spoof.is_real or resultado_spoof.confidence < UMBRAL_LIVENESS_MIN:
                        continue
                except Exception:
                    continue

            rostros_crop = detector.detect(rostro_alineado)
            if not rostros_crop:
                continue
            vec = embedding_facial(recognizer, rostro_alineado, _mejor_rostro(rostros_crop).landmarks)

            max_sim_pose = max((float(np.dot(vec, v)) for v in vectores_cap), default=0.0)
            if max_sim_pose > UMBRAL_POSE_REPETIDA:
                continue

            filename = path_estudiante / f"{nombre}_{count}.jpg"
            cv2.imwrite(str(filename), rostro_alineado)
            vectores_cap.append(vec)
            count += 1
            print(f"  Foto {count}/{MAX_FOTOS} (sim max con prev: {max_sim_pose:.3f})")

            if mostrar:
                display = frame.copy()
                x1, y1, x2, y2 = map(int, rostro.bbox)
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display, f"Fotos: {count}/{MAX_FOTOS}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow(window_name, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Cancelado por el usuario.")
                    break
    finally:
        cap.release()
        if mostrar:
            cv2.destroyAllWindows()
        print(f"\nExtraccion finalizada: {count} fotos")

    if count < MIN_FOTOS:
        print(f"Error: solo {count} fotos válidas (mínimo {MIN_FOTOS}). No se guarda en la DB.")
        return None

    print("\nGenerando embeddings...")
    from core.db_config import cargar_centroides, guardar_centroide

    existentes = cargar_centroides()
    print(f"Base vectorial cargada: {len(existentes)} personas")

    if nombre in existentes:
        print(f"Aviso: '{nombre}' ya existía en la base y será reemplazado.")

    vectores_estudiante = []
    for ruta_imagen in sorted(path_estudiante.glob("*.jpg")):
        img = cv2.imread(str(ruta_imagen))
        if img is None or img.size == 0:
            print(f"No se pudo leer: {ruta_imagen.name}")
            continue
        rostros = detector.detect(img)
        if not rostros:
            continue
        vectores_estudiante.append(
            embedding_facial(recognizer, img, _mejor_rostro(rostros).landmarks)
        )

    if len(vectores_estudiante) < MIN_FOTOS:
        print(f"Error: solo {len(vectores_estudiante)} embeddings válidos (mínimo {MIN_FOTOS}).")
        return None

    centroide = np.mean(vectores_estudiante, axis=0)
    centroide = centroide / np.linalg.norm(centroide)

    print("\nGuardando en la base vectorial (pgvector)...")
    guardado = guardar_centroide(
        nombre,
        centroide,
        modelo="arcface",
        n_muestras=len(vectores_estudiante),
        fuente="VIDEO",
        crear_usuario=True,
    )
    if not guardado:
        print("Error: no se pudo guardar el embedding en la base.")
        return None

    total = cargar_centroides()
    print(f"\nTotal registradas: {len(total)}")
    for n in total:
        print(f"  {n}")

    return total


if __name__ == "__main__":
    nombre = input("Nombre de la persona a registrar: ").strip()
    if not nombre:
        print("Error: el nombre no puede estar vacio.")
        exit()

    ruta_video = input("Ruta del archivo de video: ").strip()
    registrar_desde_video(nombre, ruta_video)
