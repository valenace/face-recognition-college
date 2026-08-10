import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.registro_video import registrar_desde_video

def main():
    # If not enough arguments, prompt interactively
    if len(sys.argv) < 3:
        print("=== Lanzador de Registro Biométrico (1 Video) ===")
        print("Uso alternativo por argumentos:")
        print("  python scripts/run_single_video_reg.py <nombre_estudiante> <ruta_video_mp4>")
        print("-" * 50)
        
        nombre = input("Ingrese el nombre/ID a registrar (ej. 'angel'): ").strip()
        if not nombre:
            print("Error: Nombre vacío.")
            return
            
        ruta = input("Ingrese la ruta del archivo de video (.mp4): ").strip()
        if not Path(ruta).exists():
            print(f"Error: El archivo '{ruta}' no existe.")
            return
    else:
        nombre = sys.argv[1]
        ruta = sys.argv[2]

    print(f"\n[INFO] Iniciando registro biométrico...")
    print(f"       Nombre: {nombre}")
    print(f"       Video:  {ruta}")
    print("-" * 50)
    
    registrar_desde_video(nombre, ruta)

if __name__ == "__main__":
    main()
