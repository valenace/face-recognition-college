import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.registro_unificado import RegistroBiometrico

def main():
    print("=== REGISTRANDO TODOS LOS ESTUDIANTES (MODO HEADLESS) ===")
    print("Cargando el pipeline robusto de registro biométrico con anti-spoofing...")
    
    # Instanciamos el registro biométrico oficial
    reg = RegistroBiometrico(db_path="data/database_embeddings.pkl", max_muestras=15)
    
    # Asegurar que la base de datos comience vacía en memoria
    reg.db_embeddings = {}
    
    videos_dir = Path("data/dataset_cv/videos")
    if not videos_dir.exists():
        print(f"❌ Error: No existe la carpeta {videos_dir}")
        return
        
    # Listar y registrar personas desde sus videos
    for folder in sorted(videos_dir.iterdir()):
        if not folder.is_dir():
            continue
        nombre = folder.name
        if nombre.endswith("_video"):
            nombre = nombre[:-6]  # Quitar sufijo '_video'
            
        videos = list(folder.glob("*.mp4"))
        if not videos:
            print(f"[WARN] No se encontró video en {folder.name}")
            continue
            
        video_path = videos[0]
        print(f"\n[PROCESANDO] Registrando a '{nombre}' usando {video_path.name}...")
        
        # Ejecuta el enrolamiento de forma rápida sin abrir ventanas OpenCV (headless)
        resultado = reg.enrolar_usuario_headless(nombre, str(video_path), sobrescribir=True)
        if resultado["exito"]:
            print(f"✅ Éxito: '{nombre}' registrado con {resultado['muestras']} muestras.")
        else:
            print(f"❌ Falló: {resultado['mensaje']}")
            
    print("\n" + "=" * 60)
    print(" BASE DE DATOS CREADA Y ACTUALIZADA CON ÉXITO")
    print("=" * 60)
    print(f"Estudiantes en la base biométrica: {list(reg.db_embeddings.keys())}")
    print("=" * 60)

if __name__ == "__main__":
    main()
