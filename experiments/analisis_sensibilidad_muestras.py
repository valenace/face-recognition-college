import sys
import os
import cv2
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Configurar para generación de imágenes sin GUI (Headless)
import matplotlib.pyplot as plt
from pathlib import Path

# Inyección al PYTHONPATH para encontrar 'core' y 'scripts'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.registro_unificado import RegistroBiometrico
from scripts.asistencia_unificada_optimizada import MotorAsistencia
from core.utils_facial import aplicar_clahe

def ejecutar_experimento():
    video_path = "data/dataset_cv/videos/angel_video/angel.mp4"
    db_temp = "data/database_temp.pkl"
    
    # Definir valores de muestras a evaluar
    rango_muestras = [1, 3, 5, 10, 15, 20, 25, 30]
    precisiones = []
    
    print("=== INICIANDO EXPERIMENTO DE SENSIBILIDAD BIOMÉTRICA ===")
    print(f"Video fuente: {video_path}\n")

    for S in rango_muestras:
        print(f"📊 Evaluando con S = {S} muestras de enrolamiento...")
        
        # 1. Enrolamiento Headless limitado a S muestras
        if os.path.exists(db_temp):
            os.remove(db_temp)
            
        reg = RegistroBiometrico(db_path=db_temp, max_muestras=S)
        res_enrol = reg.enrolar_usuario_headless("angel", video_path, sobrescribir=True)
        
        if not res_enrol["exito"]:
            print(f"  ❌ Falló el enrolamiento para S={S}: {res_enrol['mensaje']}")
            precisiones.append(0.0)
            continue
            
        print(f"  ✅ Enrolado exitosamente con {res_enrol['muestras']} muestras.")
        
        # 2. Inferencia de prueba en el video completo
        motor = MotorAsistencia(db_path=db_temp)
        
        cap = cv2.VideoCapture(video_path)
        detectados_totales = 0
        reconocidos_correctos = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            rostros = motor.detector.detect(frame)
            if not rostros:
                continue
                
            # Tomar el rostro principal
            rostro = max(rostros, key=lambda r: (r.bbox[2]-r.bbox[0]) * (r.bbox[3]-r.bbox[1]))
            detectados_totales += 1
            
            # Reconocimiento ArcFace
            f_ecualizado = aplicar_clahe(frame)
            vec = motor.recognizer.get_normalized_embedding(f_ecualizado, rostro.landmarks)
            
            dist, idx = motor.indice_faiss.search(np.array([vec.flatten()]).astype("float32"), 1)
            max_sim = dist[0][0]
            
            if max_sim > motor.umbral_similitud:
                candidato = motor.nombres_lista[idx[0][0]]
                if candidato == "angel":
                    reconocidos_correctos += 1
                    
        cap.release()
        
        # Calcular precisión
        precision = (reconocidos_correctos / detectados_totales * 100) if detectados_totales > 0 else 0.0
        precisiones.append(precision)
        print(f"  🎯 Resultado S={S} | Detecciones: {detectados_totales} | Reconocidos: {reconocidos_correctos} | Precisión: {precision:.2f}%\n")
        
    # Limpiar base de datos temporal
    if os.path.exists(db_temp):
        os.remove(db_temp)

    # 3. Graficar resultados con Matplotlib
    plt.figure(figsize=(10, 6))
    plt.plot(rango_muestras, precisiones, marker='o', linestyle='-', color='#1f77b4', linewidth=2.5, markersize=8)
    plt.title("Precisión de Reconocimiento en función del Número de Muestras de Enrolamiento", fontsize=12, fontweight='bold', pad=15)
    plt.xlabel("Cantidad de Imágenes (Muestras) utilizadas para el Registro (S)", fontsize=11, labelpad=10)
    plt.ylabel("Precisión de Reconocimiento (%)", fontsize=11, labelpad=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rango_muestras)
    plt.ylim(0, 105)
    
    # Anotar valores de precisión en los puntos
    for x, y in zip(rango_muestras, precisiones):
        plt.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, fontweight='semibold')
        
    os.makedirs("media", exist_ok=True)
    grafica_path = "media/grafica_precision_muestras.png"
    plt.savefig(grafica_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"🎉 ¡Experimento terminado! Gráfica guardada en: {grafica_path}")

if __name__ == "__main__":
    ejecutar_experimento()
