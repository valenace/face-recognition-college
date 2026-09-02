import cv2
import pickle
import numpy as np
from pathlib import Path
import sys
import os

# Inyección al PYTHONPATH para encontrar la carpeta 'core'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uniface import create_detector, create_recognizer
from uniface.spoofing import create_spoofer
from core.utils_facial import aplicar_clahe, es_imagen_borrosa, alinear_rostro

DATASET_PATH = Path("data/dataset_cv/frames")
DB_FILE = Path("data/database_embeddings.pkl")

MAX_FOTOS = 50
FRAMES_ENTRE_FOTOS = 8
UMBRAL_CALIDAD_BLUR = 60
UMBRAL_BRILLO_MIN = 20
UMBRAL_LIVENESS_MIN = 0.90


def registrar_desde_video(nombre, ruta_video, db_file=DB_FILE, dataset_path=DATASET_PATH, check_liveness=False, step_value=None):
    """
    Extrae hasta MAX_FOTOS frames alineados del video, los guarda como JPG,
    genera el embedding centroide y lo guarda en la DB.

    Args:
        nombre: ID de la persona (ej: "jorge")
        ruta_video: ruta al MP4
        db_file: ruta al .pkl de embeddings (default: data/database_embeddings.pkl)
        dataset_path: carpeta base donde guardar los JPGs (default: data/dataset_cv/frames)
        check_liveness: Si True, realiza validaciones de anti-spoofing
        step_value: paso de frames. Si es None, se calcula dinámicamente

    Returns:
        dict: la DB de embeddings resultante (o None si falló)
    """
    if not Path(ruta_video).exists():
        print("Error: el archivo de video no existe.")
        return None

    print("Cargando modelos...")
    detector = create_detector("retinaface")
    recognizer = create_recognizer("arcface")
    spoofer = create_spoofer() if check_liveness else None

    path_estudiante = Path(dataset_path) / nombre
    path_estudiante.mkdir(parents=True, exist_ok=True)
    print(f"Directorio: {path_estudiante}")

    window_name = "Registro desde Video"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        print("Error: no se pudo abrir el video.")
        return None

    # Calcular paso dinámico
    if step_value is None:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            step_value = max(1, min(8, total_frames // 100))
        else:
            step_value = FRAMES_ENTRE_FOTOS
    print(f"Paso de muestreo de frames ajustado a: {step_value}. Liveness: {check_liveness}")

    count = 0
    frames_counter = 0
    vectores_cap = []

    print("Extrayendo frames... 'q' para cancelar.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Fin del video.")
                break

            frames_counter += 1
            if frames_counter % step_value != 0:
                continue

            display = frame.copy()
            rostros = detector.detect(frame)

            rostro_listo = None
            vec_temp = None
            max_sim_pose = 0.0
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
                                if check_liveness:
                                    # Anti-spoofing check
                                    try:
                                        resultado_spoof = spoofer.predict(frame, rostro.bbox)
                                        if not resultado_spoof.is_real or resultado_spoof.confidence < UMBRAL_LIVENESS_MIN:
                                            mensaje_estado = f"SPOOF ({resultado_spoof.confidence:.2f}) - descartando"
                                            color_mensaje = (0, 0, 255)
                                        else:
                                            mensaje_estado = "OK - capturando"
                                            color_mensaje = (0, 255, 0)
                                            rostro_listo = rostro_alineado
                                    except Exception:
                                        mensaje_estado = "ERROR LIVENESS - descartando"
                                        color_mensaje = (0, 0, 255)
                                else:
                                    mensaje_estado = "OK - capturando"
                                    color_mensaje = (0, 255, 0)
                                    rostro_listo = rostro_alineado

                                # Filtro de Diversidad de Poses
                                if rostro_listo is not None:
                                    img_limpia = aplicar_clahe(rostro_listo)
                                    rostros_crop = detector.detect(img_limpia)
                                    if rostros_crop:
                                        vec = recognizer.get_normalized_embedding(img_limpia, rostros_crop[0].landmarks)
                                        
                                        # Calcular similitud máxima con los ya guardados
                                        for v_prev in vectores_cap:
                                            sim = np.dot(vec.flatten(), v_prev.flatten())
                                            if sim > max_sim_pose:
                                                max_sim_pose = sim
                                        
                                        if max_sim_pose > 0.95:
                                            mensaje_estado = f"POSE REPETIDA ({max_sim_pose:.2f}) - descartando"
                                            color_mensaje = (0, 0, 255)
                                            rostro_listo = None
                                            vec_temp = None
                                        else:
                                            vec_temp = vec
                                    else:
                                        rostro_listo = None
                                        vec_temp = None

                        h_crop, w_crop = rostro_alineado.shape[:2]
                        if (
                            10 + h_crop < display.shape[0]
                            and 10 + w_crop < display.shape[1]
                        ):
                            display[10 : 10 + h_crop, 10 : 10 + w_crop] = rostro_alineado
                            cv2.rectangle(
                                display,
                                (10, 10),
                                (10 + w_crop, 10 + h_crop),
                                color_mensaje,
                                2,
                            )

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
            cv2.rectangle(
                display, (0, h_frame - 15), (ancho_barra, h_frame), (0, 255, 0), -1
            )
            cv2.putText(
                display,
                f"Fotos: {count}/{MAX_FOTOS}",
                (20, h_frame - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                display,
                instruccion,
                (20, h_frame - 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                display,
                mensaje_estado,
                (150, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color_mensaje,
                2,
            )

            cv2.imshow(window_name, display)

            if rostro_listo is not None and count < MAX_FOTOS:
                filename = path_estudiante / f"{nombre}_{count}.jpg"
                cv2.imwrite(str(filename), rostro_listo)
                vectores_cap.append(vec_temp)
                count += 1
                print(f"  Foto {count}/{MAX_FOTOS} (sim max con prev: {max_sim_pose:.3f})")

            if count >= MAX_FOTOS:
                print("Maximo de fotos alcanzado.")
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
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
    if Path(db_file).exists():
        with open(db_file, "rb") as f:
            db_embeddings = pickle.load(f)
        print(f"Base de datos cargada: {len(db_embeddings)} personas")

    vectores_estudiante = []
    extensiones_validas = {".jpg", ".jpeg", ".png"}
    archivos_validos = [
        p for p in path_estudiante.iterdir() if p.suffix.lower() in extensiones_validas
    ]

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

        ruta_temporal = Path(db_file).with_suffix(".pkl.tmp")
        try:
            with open(ruta_temporal, "wb") as f:
                pickle.dump(db_embeddings, f)
            ruta_temporal.replace(db_file)
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

    return db_embeddings


if __name__ == "__main__":
    nombre = input("Nombre de la persona a registrar: ").strip()
    if not nombre:
        print("Error: el nombre no puede estar vacio.")
        exit()

    ruta_video = input("Ruta del archivo de video: ").strip()
    
    liveness_input = input("¿Activar validación de liveness (anti-spoofing)? (s/N): ").strip().lower()
    check_liveness = liveness_input == 's'
    
    registrar_desde_video(nombre, ruta_video, check_liveness=check_liveness)