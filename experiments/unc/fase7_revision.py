"""
Fase 7: prepara el material de revisión manual de una corrida del video 3.

Lee la carpeta de la corrida (la última prueba_*_video3 por defecto) y arma:
  revision/<persona>.jpg        una fila por (parte, pista): la mejor captura de esa pista
  revision/desconocidos_N.jpg   desconocidos con sim >= 0.25, ordenados por similitud
  revision/revision_personas.csv  tabla persona x parte x pista con la similitud máxima

El objetivo es verificar identidad con pocas imágenes: en vez de cientos de capturas,
una por pista, que es la unidad que un revisor puede etiquetar.

Uso:
  venv/bin/python experiments/unc/fase7_revision.py [--corrida=ruta]
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
DEMO = RAIZ / "experiments/unc/demo"
ANCHO = 150
COLS = 8
MAX_POR_HOJA = 48


def ultima_corrida():
    carpetas = sorted(DEMO.glob("prueba_*_video3"))
    return carpetas[-1] if carpetas else None


def leer_similitud(nombre_archivo):
    """Extrae la similitud del nombre: ..._sim0.42.jpg o ..._anc.jpg."""
    for parte in nombre_archivo.replace(".jpg", "").split("_"):
        if parte.startswith("sim"):
            try:
                return float(parte[3:])
            except ValueError:
                return -1.0
    return -1.0


def leer_parte_track(nombre_archivo):
    parte = track = 0
    for trozo in nombre_archivo.replace(".jpg", "").split("_"):
        if trozo.startswith("p") and trozo[1:].isdigit():
            parte = int(trozo[1:])
        if trozo.startswith("t") and trozo[1:].isdigit():
            track = int(trozo[1:])
    return parte, track


def anotar(img, texto, ancho=ANCHO, color=(0, 0, 255)):
    img = cv2.resize(img, (ancho, ancho))
    cv2.putText(img, texto, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1)
    return img


def hoja(imgs, destino, ancho=ANCHO, cols=COLS):
    if not imgs:
        return False
    imgs = list(imgs)
    while len(imgs) % cols:
        imgs.append(np.full((ancho, ancho, 3), 255, np.uint8))
    filas = [np.hstack(imgs[i:i + cols]) for i in range(0, len(imgs), cols)]
    cv2.imwrite(str(destino), np.vstack(filas))
    return True


def main():
    corrida = ultima_corrida()
    for a in sys.argv[1:]:
        if a.startswith("--corrida="):
            corrida = Path(a.split("=", 1)[1])
    if corrida is None or not corrida.is_dir():
        print("No encontré una corrida *_video3.")
        return

    revision = corrida / "revision"
    revision.mkdir(exist_ok=True)
    print(f"Corrida: {corrida}")

    # Una entrada por (persona, parte, pista): la captura con mayor similitud
    mejores = {}
    for ruta in (corrida / "estudiantes").glob("*/*.jpg"):
        persona = ruta.parent.name
        parte, track = leer_parte_track(ruta.name)
        sim = leer_similitud(ruta.name)
        clave = (persona, parte, track)
        if clave not in mejores or sim > mejores[clave][0]:
            mejores[clave] = (sim, ruta)

    filas = []
    personas = sorted({p for p, _, _ in mejores})
    for persona in personas:
        claves = sorted([k for k in mejores if k[0] == persona], key=lambda k: (k[1], k[2]))
        imgs = []
        for _, parte, track in claves:
            sim, ruta = mejores[(persona, parte, track)]
            img = cv2.imread(str(ruta))
            if img is None:
                continue
            imgs.append(anotar(img, f"p{parte} t{track} s{sim:.2f}" if sim >= 0 else f"p{parte} t{track} anc"))
            filas.append([persona, parte, track, round(sim, 4), ruta.name])
        if hoja(imgs, revision / f"{persona}.jpg"):
            print(f"  {persona:<8} {len(claves):3d} pistas -> revision/{persona}.jpg")

    with open(revision / "revision_personas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["persona", "parte", "pista", "sim_max", "captura"])
        w.writerows(filas)

    dudas = sorted((corrida / "desconocidos").glob("*.jpg"))
    print(f"\nDesconocidos guardados: {len(dudas)}")
    for n, inicio in enumerate(range(0, len(dudas), MAX_POR_HOJA), 1):
        imgs = []
        for ruta in dudas[inicio:inicio + MAX_POR_HOJA]:
            img = cv2.imread(str(ruta))
            if img is None:
                continue
            imgs.append(anotar(img, ruta.stem.replace("_top2-", " / ")[:26], ancho=ANCHO))
        if hoja(imgs, revision / f"desconocidos_{n:02d}.jpg"):
            print(f"  revisión desconocidos {n:02d}: {len(imgs)} recortes")

    print(f"\nMaterial en: {revision}")


if __name__ == "__main__":
    main()
