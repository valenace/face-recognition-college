import sys
import os
import cv2
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Inyección al PYTHONPATH para encontrar 'core' y 'scripts'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.registro_unificado import RegistroBiometrico
from scripts.asistencia_unificada_optimizada import MotorAsistencia
from core.utils_facial import aplicar_clahe

def calcular_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def ejecutar_experimento_salon():
    video_angel = "data/dataset_cv/videos/angel_video/angel.mp4"
    video_salon = "data/salon/20260806_105645.mp4"
    db_temp = "data/database_temp.pkl"
    
    print("=== INICIANDO EXPERIMENTO DE SENSIBILIDAD EN EL AULA ===")
    
    # ── FASE A: DESCUBRIR GROUND TRUTH DE ANGEL EN EL SALÓN ─────────────────
    print("\n[FASE A] Descubriendo la ubicación de Angel en el salón usando referencia S=25...")
    if os.path.exists(db_temp):
        os.remove(db_temp)
    
    reg_ref = RegistroBiometrico(db_path=db_temp, max_muestras=25)
    reg_ref.enrolar_usuario_headless("angel", video_angel, sobrescribir=True)
    motor_ref = MotorAsistencia(db_path=db_temp)
    
    cap = cv2.VideoCapture(video_salon)
    ground_truth_angel = {}
    frame_idx = 0
    paso_frames = 15 # Evaluar 1 frame cada 15 para velocidad
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % paso_frames == 0:
            rostros = motor_ref.detector.detect(frame)
            f_ecualizado = aplicar_clahe(frame)
            
            for r in rostros:
                vec = motor_ref.recognizer.get_normalized_embedding(f_ecualizado, r.landmarks)
                dist, idx = motor_ref.indice_faiss.search(np.array([vec.flatten()]).astype("float32"), 1)
                max_sim = dist[0][0]
                
                # Criterio de coincidencia para ground truth
                if max_sim > 0.35 and motor_ref.nombres_lista[idx[0][0]] == "angel":
                    ground_truth_angel[frame_idx] = r.bbox
                    break # Solo hay un Angel en escena
        
        frame_idx += 1
    
    cap.release()
    print(f"[OK] Ground Truth obtenido: Angel localizado en {len(ground_truth_angel)} frames de prueba.\n")
    
    # ── FASE B: BARRIDO DE MUESTRAS S ──────────────────────────────────────
    rango_muestras = [1, 3, 5, 10, 15, 20, 25, 30]
    precisiones = []
    
    for S in rango_muestras:
        print(f"📊 Evaluando con S = {S} muestras...")
        
        if os.path.exists(db_temp):
            os.remove(db_temp)
            
        reg = RegistroBiometrico(db_path=db_temp, max_muestras=S)
        reg.enrolar_usuario_headless("angel", video_angel, sobrescribir=True)
        motor = MotorAsistencia(db_path=db_temp)
        
        cap = cv2.VideoCapture(video_salon)
        curr_frame_idx = 0
        reconocidos = 0
        totales_evaluados = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Solo procesar si el frame pertenece al Ground Truth
            if curr_frame_idx in ground_truth_angel:
                bbox_ref = ground_truth_angel[curr_frame_idx]
                rostros = motor.detector.detect(frame)
                f_ecualizado = aplicar_clahe(frame)
                
                # Buscar el rostro correspondiente al bbox de referencia (IoU > 0.5)
                mejor_iou = 0
                rostro_angel = None
                for r in rostros:
                    iou = calcular_iou(bbox_ref, r.bbox)
                    if iou > mejor_iou:
                        mejor_iou = iou
                        rostro_angel = r
                        
                if mejor_iou > 0.5 and rostro_angel is not None:
                    totales_evaluados += 1
                    vec = motor.recognizer.get_normalized_embedding(f_ecualizado, rostro_angel.landmarks)
                    dist, idx = motor.indice_faiss.search(np.array([vec.flatten()]).astype("float32"), 1)
                    max_sim = dist[0][0]
                    
                    if max_sim > motor.umbral_similitud:
                        candidato = motor.nombres_lista[idx[0][0]]
                        if candidato == "angel":
                            reconocidos += 1
            
            curr_frame_idx += 1
            
        cap.release()
        
        precision = (reconocidos / totales_evaluados * 100) if totales_evaluados > 0 else 0.0
        precisiones.append(precision)
        print(f"  🎯 Resultado S={S} | Eval: {totales_evaluados} | Reconocidos: {reconocidos} | Precisión: {precision:.2f}%\n")
        
    # Limpiar
    if os.path.exists(db_temp):
        os.remove(db_temp)
        
    # ── FASE C: GRAFICAR RESULTADOS ────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    plt.plot(rango_muestras, precisiones, marker='o', linestyle='-', color='#2ca02c', linewidth=2.5, markersize=8)
    plt.title("Precisión de Reconocimiento en el Aula (Distancia y Sombras) vs Muestras", fontsize=12, fontweight='bold', pad=15)
    plt.xlabel("Cantidad de Imágenes (Muestras) utilizadas para el Registro (S)", fontsize=11, labelpad=10)
    plt.ylabel("Precisión de Reconocimiento (%)", fontsize=11, labelpad=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rango_muestras)
    plt.ylim(0, 105)
    
    for x, y in zip(rango_muestras, precisiones):
        plt.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, fontweight='semibold')
        
    os.makedirs("media", exist_ok=True)
    grafica_path = "media/grafica_precision_salon.png"
    plt.savefig(grafica_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"🎉 ¡Experimento de aula terminado! Gráfica guardada en: {grafica_path}")

if __name__ == "__main__":
    ejecutar_experimento_salon()
