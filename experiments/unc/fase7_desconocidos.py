"""
Fase 7: material de revisión de los desconocidos, sin repeticiones.

Toma los recortes guardados en desconocidos/ (similitud >= 0.25) y deja uno por
(parte, segundo, pista), ordenados por similitud. También arma una hoja de
referencia con las fotos de registro de los 9 estudiantes.

Salidas en revision/:
  desconocidos_dedup_NN.jpg   recortes únicos, de mayor a menor similitud
  registro_todos.jpg          referencia: 2 fotos por estudiante
  desconocidos_dedup.csv      tabla parte, segundo, pista, sim, top2, archivo

Uso:
  venv/bin/python experiments/unc/fase7_desconocidos.py [--corrida=ruta]
"""

import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
DEMO = RAIZ / "experiments/unc/demo"
PATRON = re.compile(r"p(\d+)_s([0-9.]+)_t(\d+)_sim([0-9.]+)_top2-([a-zA-Z0-9]+)")
ANCHO = 150
COLS = 8
POR_HOJA = 48


def ultima_corrida():
    carpetas = sorted(DEMO.glob("prueba_*_video3"))
    return carpetas[-1] if carpetas else None


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

    # Hoja de referencia: 2 fotos por estudiante
    imgs = []
    for persona in sorted(p.name for p in (RAIZ / "data/dataset_cv/frames").iterdir() if p.is_dir()):
        fotos = sorted((RAIZ / "data/dataset_cv/frames" / persona).glob("*.jpg"))
        if not fotos:
            continue
        elegidas = [fotos[0], fotos[len(fotos) // 2]]
        for ruta in elegidas:
            img = cv2.imread(str(ruta))
            if img is None:
                continue
            img = cv2.resize(img, (ANCHO, ANCHO))
            cv2.putText(img, persona, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            imgs.append(img)
    if hoja(imgs, revision / "registro_todos.jpg"):
        print(f"Referencia: {len(imgs)} fotos -> revision/registro_todos.jpg")

    # Desconocidos sin repetición por (parte, segundo, pista)
    unicos = {}
    for ruta in sorted((corrida / "desconocidos").glob("*.jpg")):
        m = PATRON.match(ruta.stem)
        if not m:
            continue
        parte, segundo, track, sim, top2 = int(m[1]), float(m[2]), int(m[3]), float(m[4]), m[5]
        clave = (parte, round(segundo), track)
        if clave not in unicos or sim > unicos[clave]["sim"]:
            unicos[clave] = {"parte": parte, "segundo": segundo, "pista": track,
                             "sim": sim, "top2": top2, "archivo": ruta.name, "ruta": ruta}

    filas = sorted(unicos.values(), key=lambda f: -f["sim"])
    print(f"Recortes guardados: {len(list((corrida / 'desconocidos').glob('*.jpg')))} "
          f"| únicos por (parte, segundo, pista): {len(filas)}")

    for n, inicio in enumerate(range(0, len(filas), POR_HOJA), 1):
        imgs = []
        for fila in filas[inicio:inicio + POR_HOJA]:
            img = cv2.imread(str(fila["ruta"]))
            if img is None:
                continue
            img = cv2.resize(img, (ANCHO, ANCHO))
            cv2.putText(img, f"p{fila['parte']} s{fila['segundo']:.0f} t{fila['pista']} {fila['sim']:.2f}",
                        (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 255), 1)
            imgs.append(img)
        hoja(imgs, revision / f"desconocidos_dedup_{n:02d}.jpg")
        print(f"  hoja {n:02d}: {len(imgs)} recortes")

    with open(revision / "desconocidos_dedup.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["parte", "segundo", "pista", "sim", "top2", "archivo"])
        for fila in filas:
            w.writerow([fila["parte"], fila["segundo"], fila["pista"], fila["sim"],
                        fila["top2"], fila["archivo"]])

    por_top2 = {}
    for fila in filas:
        por_top2.setdefault(fila["top2"], []).append(fila["sim"])
    print("\nSegunda opción más frecuente en los desconocidos (top-2):")
    for nombre, sims in sorted(por_top2.items(), key=lambda kv: -len(kv[1])):
        print(f"  {nombre:<10} {len(sims):4d} recortes | sim media {np.mean(sims):.3f} "
              f"| max {np.max(sims):.3f}")


if __name__ == "__main__":
    main()
