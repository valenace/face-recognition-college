import sys
import os
import pickle
from pathlib import Path


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.asistencia_unificada import MotorAsistencia

def main():
    db_file = Path("data/database_embeddings.pkl")
    # Videos disponibles: 20260806_105629.mp4, 20260806_105645.mp4, 20260806_105847.mp4, 20260806_105921.mp4
    video_salon = Path("data/salon/20260806_105645.mp4")
    
    if len(sys.argv) > 1:
        video_salon = Path(sys.argv[1])
        
    if not video_salon.exists():
        print(f"❌ Error: El video del salón '{video_salon}' no existe.")
        return

    print("\n" + "=" * 60)
    print(" INFERENCIA DE ASISTENCIA EN EL AULA (ROBUSTO)")
    print("=" * 60)
    print(f"Evaluando video del salón: {video_salon}")
    print("Solo 'angel' será identificado. El resto de rostros serán 'Desconocido'.")
    print("Presiona 'q' en la ventana de video para salir.")
    print("-" * 60)
    
    try:
        motor = MotorAsistencia(db_path=str(db_file))
        motor.iniciar_vigilancia(origen_video=str(video_salon))
    except Exception as e:
        print(f"❌ Error durante la inferencia: {e}")

if __name__ == "__main__":
    main()
