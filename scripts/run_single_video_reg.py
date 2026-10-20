import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.registro_video import registrar_desde_video, DB_FILE, DATASET_PATH


def main():
    parser = argparse.ArgumentParser(description="Registro biométrico desde un video.")
    parser.add_argument("nombre", nargs="?", help="Nombre/ID del estudiante")
    parser.add_argument("video", nargs="?", help="Ruta del MP4")
    parser.add_argument("--db", default=str(DB_FILE), help="Ruta del .pkl de embeddings")
    parser.add_argument("--dataset", default=str(DATASET_PATH), help="Carpeta base de frames")
    parser.add_argument("--liveness", action="store_true", help="Activar anti-spoofing")
    parser.add_argument("--sin-ventana", action="store_true", help="No mostrar ventana (lote)")
    args = parser.parse_args()

    if not args.nombre or not args.video:
        print("=== Lanzador de Registro Biométrico (1 Video) ===")
        print("Uso alternativo por argumentos:")
        print("  python scripts/run_single_video_reg.py <nombre> <video.mp4> [--db ...] [--dataset ...]")
        print("-" * 50)
        args.nombre = input("Ingrese el nombre/ID a registrar (ej. 'angel'): ").strip()
        if not args.nombre:
            print("Error: Nombre vacío.")
            return
        args.video = input("Ingrese la ruta del archivo de video (.mp4): ").strip()

    if not Path(args.video).exists():
        print(f"Error: El archivo '{args.video}' no existe.")
        return

    print(f"\n[INFO] Iniciando registro biométrico...")
    print(f"       Nombre: {args.nombre}")
    print(f"       Video:  {args.video}")
    print(f"       DB:     {args.db}")
    print("-" * 50)

    registrar_desde_video(
        args.nombre,
        args.video,
        db_file=args.db,
        dataset_path=args.dataset,
        check_liveness=args.liveness,
        mostrar=not args.sin_ventana,
    )


if __name__ == "__main__":
    main()
