import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Inyección al PYTHONPATH para encontrar 'core' y 'scripts'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uniface import create_detector, create_recognizer
from uniface.spoofing import create_spoofer
from core.utils_facial import aplicar_clahe, es_imagen_borrosa, alinear_rostro

def depurar_registro(nombre, ruta_video, db_file="data/database_embeddings.pkl", dataset_path="data/dataset_cv/frames"):
    print("\n[INFO] Inicializando Depurador de Registro...")
    print("Cargando modelos biométricos...")
    detector = create_detector("retinaface")
    recognizer = create_recognizer("arcface")
    spoofer = create_spoofer()

    if not Path(ruta_video).exists():
        print(f"❌ Error: El video '{ruta_video}' no existe.")
        return

    path_estudiante = Path(dataset_path) / nombre
    path_estudiante.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Directorio de salida: {path_estudiante}")

    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        print("❌ Error al abrir el video.")
        return

    # Calcular paso dinámico por defecto
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step_value = max(1, min(8, total_frames // 100)) if total_frames > 0 else 5

    frame_idx = 0
    count = 0
    modo_automatico = False
    check_liveness = False # Comienza en False para poder alternarlo con 'L' y ver el impacto

    window_name = "Depurador de Enrolamiento Facial"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n" + "="*60)
    print(" CONTROLES DE LA VENTANA:")
    print("  - [Espacio]: Avanzar al siguiente frame (Modo Paso a Paso)")
    print("  - [A]: Alternar Modo Automático (Ejecución continua)")
    print("  - [S]: Saltar 10 frames")
    print("  - [L]: Alternar Control de Liveness / Anti-spoofing (ON/OFF)")
    print("  - [Q]: Salir y guardar embeddings")
    print("="*60 + "\n")
    print(f"Paso de muestreo dinámico calculado: {step_value} frames (evalúa ~100 frames del video).")

    vectores_extraidos = []

    try:
        while True:
            # Pausa interactiva si no estamos en modo automático
            if not modo_automatico:
                print(f"\n[PAUSA] Frame #{frame_idx} | Fotos: {count} | Liveness: {'ON' if check_liveness else 'OFF'}. Presiona [Espacio] para procesar...")
                while True:
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord(' '):
                        break
                    elif key == ord('a'):
                        modo_automatico = True
                        print("▶️ Modo automático activado.")
                        break
                    elif key == ord('l'):
                        check_liveness = not check_liveness
                        print(f"🔒 Liveness cambiado a: {'ON' if check_liveness else 'OFF'}")
                        print(f"[PAUSA] Frame #{frame_idx} | Fotos: {count} | Liveness: {'ON' if check_liveness else 'OFF'}. Presiona [Espacio]...")
                    elif key == ord('s'):
                        # Saltar 10 frames
                        for _ in range(10):
                            cap.read()
                            frame_idx += 1
                        print(f"⏭️ Saltados 10 frames. Ahora en #{frame_idx}")
                        break
                    elif key == ord('q'):
                        print("⏹️ Salida solicitada por el usuario.")
                        return
            
            ret, frame = cap.read()
            if not ret:
                print("🏁 Fin del video.")
                break

            frame_idx += 1
            if frame_idx % step_value != 0:
                continue

            display = frame.copy()
            rostros = detector.detect(frame)

            rostro_listo = None
            mensaje_estado = "Buscando rostro..."
            color_mensaje = (0, 165, 255) # Naranja
            detalles_consola = []

            if rostros:
                rostro = rostros[0]
                landmarks = rostro.landmarks
                x1, y1, x2, y2 = map(int, rostro.bbox)
                w_face, h_face = x2 - x1, y2 - y1
                detalles_consola.append(f"Face Bbox: {w_face}x{h_face}px")

                # QA 1: Bbox Size
                if w_face < 60 or h_face < 60:
                    mensaje_estado = "MUY LEJOS"
                    color_mensaje = (0, 165, 255)
                # QA 2: Landmarks
                elif landmarks is None or len(landmarks) < 5:
                    mensaje_estado = "SIN Landmarks"
                    color_mensaje = (0, 165, 255)
                else:
                    # QA 3: Alineación
                    rostro_alineado = alinear_rostro(frame, landmarks)
                    if rostro_alineado is None:
                        mensaje_estado = "FALLO ALINEACION"
                        color_mensaje = (0, 0, 255)
                    else:
                        # QA 4: Borrosidad
                        gray = cv2.cvtColor(rostro_alineado, cv2.COLOR_BGR2GRAY)
                        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                        detalles_consola.append(f"Blur: {blur_score:.1f} (Umbral > 60)")

                        if blur_score < 60:
                            mensaje_estado = "BORROSO"
                            color_mensaje = (0, 0, 255)
                        else:
                            # QA 5: Brillo/Iluminación
                            brillo = np.mean(gray)
                            detalles_consola.append(f"Brillo: {brillo:.1f} (Umbral > 20)")

                            if brillo < 20:
                                mensaje_estado = "MUY OSCURO"
                                color_mensaje = (0, 0, 255)
                            else:
                                # QA 6: Liveness Check (Opcional)
                                if check_liveness:
                                    try:
                                        resultado_spoof = spoofer.predict(frame, rostro.bbox)
                                        detalles_consola.append(f"Liveness: {'REAL' if resultado_spoof.is_real else 'SPOOF'} (conf: {resultado_spoof.confidence:.4f})")
                                        if not resultado_spoof.is_real or resultado_spoof.confidence < 0.90:
                                            mensaje_estado = f"SPOOF ({resultado_spoof.confidence:.2f})"
                                            color_mensaje = (0, 0, 255)
                                        else:
                                            mensaje_estado = "OK - CAPTURANDO"
                                            color_mensaje = (0, 255, 0)
                                            rostro_listo = rostro_alineado
                                    except Exception as e:
                                        mensaje_estado = "ERROR LIVENESS"
                                        color_mensaje = (0, 0, 255)
                                else:
                                    detalles_consola.append("Liveness: Omitido")
                                    mensaje_estado = "OK - CAPTURANDO"
                                    color_mensaje = (0, 255, 0)
                                    rostro_listo = rostro_alineado

                        # Render del rostro alineado en la esquina superior izquierda
                        h_crop, w_crop = rostro_alineado.shape[:2]
                        if 10 + h_crop < display.shape[0] and 10 + w_crop < display.shape[1]:
                            display[10 : 10 + h_crop, 10 : 10 + w_crop] = rostro_alineado
                            cv2.rectangle(display, (10, 10), (10 + w_crop, 10 + h_crop), color_mensaje, 2)

                cv2.rectangle(display, (x1, y1), (x2, y2), color_mensaje, 2)
            else:
                detalles_consola.append("Ningún rostro detectado.")

            # Imprimir parámetros de diagnóstico detallados en consola
            print(f"  Frame #{frame_idx} | Estado: {mensaje_estado} | " + " | ".join(detalles_consola))

            # UI labels
            h_f = display.shape[0]
            cv2.putText(display, f"Fotos: {count}", (20, h_f - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display, f"Liveness: {'ON' if check_liveness else 'OFF'} (Tecla L para alternar)", (20, h_f - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, mensaje_estado, (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_mensaje, 2)

            cv2.imshow(window_name, display)

            # Escribir frame si pasó el QA y no hemos excedido el máximo
            if rostro_listo is not None and count < 50:
                filename = path_estudiante / f"{nombre}_{count}.jpg"
                cv2.imwrite(str(filename), rostro_listo)
                
                # Extraer embedding
                img_limpia = aplicar_clahe(rostro_listo)
                rostros_crop = detector.detect(img_limpia)
                if rostros_crop:
                    vec = recognizer.get_normalized_embedding(img_limpia, rostros_crop[0].landmarks)
                    vectores_extraidos.append(vec)
                    count += 1
                    print(f"    💾 Guardada Foto #{count} -> {filename.name}")

            if count >= 50:
                print("🏁 Máximo de 50 fotos alcanzado.")
                break

            if modo_automatico:
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    modo_automatico = False
                    print("⏸️ Pausado por el usuario.")
                elif key == ord('l'):
                    check_liveness = not check_liveness
                    print(f"🔒 Liveness cambiado a: {'ON' if check_liveness else 'OFF'}")
                elif key == ord('q'):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n🏁 Extracción finalizada: {count} fotos guardadas.")

    # Generación y guardado final de embeddings promedio
    if vectores_extraidos:
        import pickle
        db_embeddings = {}
        if Path(db_file).exists():
            with open(db_file, "rb") as f:
                db_embeddings = pickle.load(f)
        
        centroide = np.mean(vectores_extraidos, axis=0)
        centroide_normalizado = centroide / np.linalg.norm(centroide)
        db_embeddings[nombre] = centroide_normalizado
        
        with open(db_file, "wb") as f:
            pickle.dump(db_embeddings, f)
        print(f"✅ Base de datos biométrica actualizada con '{nombre}' usando {len(vectores_extraidos)} vectores de calidad.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("❌ Error: Argumentos incorrectos.")
        print("Uso: python tests/run_registro_paso_a_paso.py <nombre_usuario> <ruta_video>")
        print("Ejemplo: python tests/run_registro_paso_a_paso.py ronald data/dataset_cv/videos/ronald_delgado_video/ronald_delgado.mp4")
        sys.exit(1)
    
    nombre = sys.argv[1]
    ruta_video = sys.argv[2]
    depurar_registro(nombre, ruta_video)
