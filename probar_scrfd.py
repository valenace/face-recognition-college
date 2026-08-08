#!/usr/bin/env python3
"""
probar_scrfd.py — Lanzador y Comparador de SCRFD vs RetinaFace (CV College v2)
================================================================================
Este script interactivo te permite probar y comparar fácilmente los flujos
de registro y reconocimiento facial usando ambos detectores.
"""

import sys
import os
import subprocess

def verificar_dependencias():
    print("[...] Verificando dependencias en el entorno actual...")
    librerias_faltantes = []
    
    # Verificar OpenCV
    try:
        import cv2
        version = getattr(cv2, '__version__', 'instalado')
        print(f"  [OK] OpenCV ({version})")
    except ImportError:
        librerias_faltantes.append("opencv-python-headless")
        
    # Verificar UniFace
    try:
        import uniface
        print("  [OK] UniFace (IA)")
    except ImportError:
        librerias_faltantes.append("uniface")
        
    # Verificar Supervision
    try:
        import supervision as sv
        print("  [OK] Supervision (Tracking)")
    except ImportError:
        librerias_faltantes.append("supervision")
        
    # Verificar FAISS
    try:
        import faiss
        print("  [OK] FAISS (Búsqueda Vectorial)")
    except ImportError:
        librerias_faltantes.append("faiss-cpu")

    if librerias_faltantes:
        print("\n[!] ADVERTENCIA: Faltan las siguientes librerías en este entorno de Python:")
        for lib in librerias_faltantes:
            print(f"  - {lib}")
        print("\nPor favor, ejecuta el siguiente comando en tu terminal para instalarlas:")
        print(f"  pip install {' '.join(librerias_faltantes)}")
        print("\nO activa tu entorno virtual correcto antes de continuar.")
        return False
    
    print("[OK] Todas las dependencias están instaladas.\n")
    return True

def ejecutar_script(script_path):
    if not os.path.exists(script_path):
        print(f"[ERROR] No se encontró el script en: {script_path}")
        return
    
    print(f"\n[INFO] Ejecutando: python3 {script_path} ...")
    try:
        # Ejecuta el script compartiendo la entrada/salida estándar para que sea interactivo
        subprocess.run([sys.executable, script_path], check=True)
    except KeyboardInterrupt:
        print("\n[INFO] Ejecución cancelada por el usuario.")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] El script finalizó con errores (código {e.returncode}).")

def menu_principal():
    print("=" * 65)
    print("      PROBADOR DE FLUJOS BIOMÉTRICOS: SCRFD vs RETINAFACE")
    print("=" * 65)
    
    verificar_dependencias()
    
    while True:
        print("\n--- MENÚ PRINCIPAL ---")
        print("1. [SCRFD] Enrolamiento y Registro (registro_unificado_scrfd.py)")
        print("2. [SCRFD] Asistencia en Vivo (asistencia_unificada_scrfd.py)")
        print("-" * 65)
        print("3. [Original] Enrolamiento y Registro (registro_unificado.py)")
        print("4. [Original] Asistencia en Vivo (asistencia_unificada.py)")
        print("-" * 65)
        print("5. Salir")
        
        opcion = input("\nSeleccione una opción (1-5): ").strip()
        
        if opcion == "1":
            ejecutar_script("scripts/registro_unificado_scrfd.py")
        elif opcion == "2":
            ejecutar_script("scripts/asistencia_unificada_scrfd.py")
        elif opcion == "3":
            ejecutar_script("scripts/registro_unificado.py")
        elif opcion == "4":
            ejecutar_script("scripts/asistencia_unificada.py")
        elif opcion == "5":
            print("\n¡Gracias por probar el sistema!")
            break
        else:
            print("[ERROR] Opción no válida. Intente de nuevo.")

if __name__ == "__main__":
    menu_principal()
