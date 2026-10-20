import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.asistencia_unificada import MotorAsistencia

def main():
    db_file = Path("data/database_embeddings.pkl")
    if not db_file.exists():
        print("❌ Error: La base de datos biométrica no existe en 'data/database_embeddings.pkl'.")
        print("Ejecute primero: `./venv/bin/python3 scripts/register_all_students.py`")
        return
        
    video_path = "data/dataset_cv/videos/angel_video/angel.mp4"
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        
    if not Path(video_path).exists():
        print(f"❌ Error: El video '{video_path}' no existe.")
        return

    print("\n" + "=" * 60)
    print(" INFERENCIA DE ASISTENCIA ROBUSTA (VISUALIZACIÓN)")
    print("=" * 60)
    print(f"Procesando video: {video_path}")
    print("Se abrirá la ventana 'Control de Asistencia'. Presiona 'q' para salir.")
    print("-" * 60)
    
    try:
        # Se instancia el motor robusto de asistencia
        motor = MotorAsistencia(db_path=str(db_file))
        motor.iniciar_vigilancia(origen_video=str(video_path))
    except Exception as e:
        print(f"❌ Error al iniciar el motor de asistencia: {e}")

if __name__ == "__main__":
    main()
