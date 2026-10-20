import sys
import os
import cv2
import numpy as np
import time
from pathlib import Path

# Inyección al PYTHONPATH para encontrar 'core' y 'scripts'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.asistencia_unificada_optimizada import MotorAsistencia
from core.utils_facial import aplicar_clahe

class DepuradorAsistencia(MotorAsistencia):
    def iniciar_depuracion(self, origen_video):
        cap = cv2.VideoCapture(origen_video)
        if not cap.isOpened():
            print(f"❌ Error: No se pudo abrir {origen_video}")
            return

        cv2.namedWindow("Depurador de Asistencia", cv2.WINDOW_NORMAL)
        frame_idx = 0
        modo_automatico = False
        
        print("\n" + "="*60)
        # Controles
        print(" CONTROLES DE LA VENTANA:")
        print("  - [Espacio]: Avanzar al siguiente frame (Modo Paso a Paso)")
        print("  - [A]: Alternar Modo Automático (Ejecución continua)")
        print("  - [S]: Saltar 10 frames")
        print("  - [Q]: Salir")
        print("="*60 + "\n")

        try:
            while True:
                if not modo_automatico:
                    print(f"\n[PAUSA] Frame #{frame_idx}. Presiona [Espacio] para procesar...")
                    while True:
                        key = cv2.waitKey(0) & 0xFF
                        if key == ord(' '):
                            break
                        elif key == ord('a'):
                            modo_automatico = True
                            print("▶️ Modo automático activado.")
                            break
                        elif key == ord('s'):
                            # Saltar 10 frames
                            for _ in range(10):
                                cap.read()
                                frame_idx += 1
                            print(f"⏭️ Saltados 10 frames. Ahora en #{frame_idx}")
                            break
                        elif key == ord('q'):
                            return

                ret, frame = cap.read()
                if not ret:
                    print("🏁 Fin del video.")
                    break
                
                frame_idx += 1
                f_ecualizado = None
                
                # Ejecutar detección
                rostros_raw = self.detector.detect(frame)
                detections_list = []
                rostros_data = []

                for r in rostros_raw:
                    x1, y1, x2, y2 = map(int, r.bbox)
                    if (x2 - x1) > self.tamano_minimo_rostro:
                        detections_list.append([x1, y1, x2, y2, r.confidence])
                        rostros_data.append(r)

                print(f"\n--- Frame #{frame_idx} | Rostros detectados > {self.tamano_minimo_rostro}px: {len(detections_list)} ---")

                if len(detections_list) > 0:
                    import supervision as sv
                    detections = sv.Detections(
                        xyxy=np.array([d[:4] for d in detections_list]),
                        confidence=np.array([d[4] for d in detections_list]),
                    )
                    tracked = self.tracker.update_with_detections(detections=detections)

                    for t in tracked:
                        x1, y1, x2, y2 = map(int, t[0])
                        t_id = t[4]
                        
                        h_frame, w_frame = frame.shape[:2]
                        margen = 15
                        if x1 < margen or y1 < margen or x2 > (w_frame - margen) or y2 > (h_frame - margen):
                            print(f"  ⚠️ ID {t_id}: Rostro muy al borde (Céntrate). Saltando reconocimiento.")
                            self._dibujar_ui(frame, x1, y1, x2, y2, "Centrate", (0, 165, 255), t_id)
                            continue

                        # Búsqueda por IoU
                        mejor_iou, r_asociado = 0, None
                        for r in rostros_data:
                            iou = self._calcular_iou([x1, y1, x2, y2], list(map(int, r.bbox)))
                            if iou > mejor_iou:
                                mejor_iou, r_asociado = iou, r

                        if mejor_iou < 0.5 or r_asociado is None:
                            continue

                        if t_id in self.identidades_ancladas:
                            nombre = self.identidades_ancladas[t_id]
                            print(f"  👤 ID {t_id}: [ANCLADO] -> {nombre}")
                            self._dibujar_ui(frame, x1, y1, x2, y2, nombre, (0, 255, 0), t_id)
                        else:
                            # 1. Reconocimiento ArcFace Primero
                            if f_ecualizado is None:
                                f_ecualizado = aplicar_clahe(frame)
                            vec = self.recognizer.get_normalized_embedding(f_ecualizado, r_asociado.landmarks)
                            
                            dist, idx = self.indice_faiss.search(np.array([vec.flatten()]).astype("float32"), 1)
                            max_sim = dist[0][0]
                            
                            if max_sim > self.umbral_similitud:
                                candidato = self.nombres_lista[idx[0][0]]

                                # 2. Liveness Check (Solo para reconocidos con ancho >= 80px)
                                ancho_rostro = x2 - x1
                                if ancho_rostro < 80:
                                    is_real = True
                                    print(f"  🔍 ID {t_id}: [LIVENESS] Omitido por distancia (ancho: {ancho_rostro}px < 80px)")
                                else:
                                    try:
                                        res_spoof = self.spoofer.predict(frame, r_asociado.bbox)
                                        is_real = res_spoof.is_real and res_spoof.confidence >= self.umbral_liveness
                                        print(f"  🔍 ID {t_id}: [LIVENESS] Real={res_spoof.is_real} (conf: {res_spoof.confidence:.2f}) -> {'PASÓ' if is_real else 'FRAUDE'}")
                                    except Exception:
                                        is_real = False
                                        print(f"  🔍 ID {t_id}: [LIVENESS] Error en validación")

                                if not is_real:
                                    self._dibujar_ui(frame, x1, y1, x2, y2, "FRAUDE", (0, 0, 255), t_id)
                                    continue

                                # Votos
                                if t_id not in self.votos_identidad:
                                    self.votos_identidad[t_id] = {}
                                self.votos_identidad[t_id][candidato] = self.votos_identidad[t_id].get(candidato, 0) + 1
                                votos_actuales = self.votos_identidad[t_id][candidato]
                                
                                print(f"  🎯 ID {t_id}: [MATCH] Candidato='{candidato}' | Similitud={max_sim:.3f} | Votos={votos_actuales}/{self.votos_requeridos}")

                                if votos_actuales >= self.votos_requeridos:
                                    self.identidades_ancladas[t_id] = candidato
                                    print(f"     ✅ ID {t_id} ANCLADO oficialmente a '{candidato}'")
                                    self._registrar_asistencia(candidato, max_sim)
                                    self._dibujar_ui(frame, x1, y1, x2, y2, candidato, (0, 255, 0), t_id)
                                else:
                                    self._dibujar_ui(frame, x1, y1, x2, y2, f"Verificando {votos_actuales}/{self.votos_requeridos}", (0, 165, 255), t_id)
                            else:
                                print(f"  ❓ ID {t_id}: [DESCONOCIDO] | Max Sim={max_sim:.3f} (Umbral={self.umbral_similitud})")
                                self._dibujar_ui(frame, x1, y1, x2, y2, "Desconocido", (0, 0, 255), t_id)

                cv2.imshow("Depurador de Asistencia", frame)
                
                # En modo automático, procesamos continuamente escuchando brevemente
                if modo_automatico:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(' '):
                        modo_automatico = False
                        print("⏸️ Pausado por el usuario.")
                    elif key == ord('q'):
                        break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\n🏁 Depuración finalizada.")

if __name__ == "__main__":
    db_file = "data/database_embeddings.pkl"
    video_salon = "data/salon/20260806_105645.mp4"
    
    # Permitir argumentos
    if len(sys.argv) > 1:
        video_salon = sys.argv[1]

    if not Path(video_salon).exists():
        print(f"❌ Error: El video '{video_salon}' no existe.")
        sys.exit(1)
        
    depurador = DepuradorAsistencia(db_path=db_file)
    depurador.iniciar_depuracion(video_salon)
