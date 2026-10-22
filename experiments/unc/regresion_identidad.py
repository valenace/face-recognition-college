"""
Regresión de identidad: comprueba que el motor sigue acertando las pistas etiquetadas.

Lee `regresion_pistas.csv` (video, tramo, pista y la identidad verdadera) y procesa
cada tramo con el motor actual. Compara la identidad que asigna a cada pista con la
esperada y resume aciertos, errores y falsos positivos.

Uso:
    venv/bin/python experiments/unc/regresion_identidad.py
    venv/bin/python experiments/unc/regresion_identidad.py --max-fallos=0

Sale con código 1 si hay más fallos que `--max-fallos` (por defecto 0), para que se
pueda usar como puerta antes de aceptar un cambio en el motor.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2

RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))

from scripts.asistencia_unificada_optimizada import MotorAsistencia

FIXTURE = Path(__file__).parent / "regresion_pistas.csv"
CSV_TEMPORAL = Path("/tmp/opencode/regresion_asistencia.csv")


def main():
    max_fallos = 0
    for arg in sys.argv[1:]:
        if arg.startswith("--max-fallos="):
            max_fallos = int(arg.split("=", 1)[1])

    if not FIXTURE.exists():
        print(f"No existe el fixture: {FIXTURE}")
        return 1

    filas = list(csv.DictReader(FIXTURE.open(encoding="utf-8")))
    grupos = defaultdict(list)
    for fila in filas:
        grupos[(fila["archivo"], float(fila["desde"]), float(fila["hasta"]))].append(fila)

    resultados = []
    for (archivo, desde, hasta), esperadas in grupos.items():
        ruta = RAIZ / archivo
        if not ruta.exists():
            print(f"  ✗ no existe el video {archivo}")
            continue

        print(f"\nProcesando {archivo} ({desde:.0f}–{hasta:.0f} s)...")
        motor = MotorAsistencia(csv_path=str(CSV_TEMPORAL))
        motor.tamano_minimo_rostro = 20

        cap = cv2.VideoCapture(str(ruta))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(desde * fps))
        limite = min(int(hasta * fps), int(total)) if total else int(hasta * fps)

        pistas = {}
        for _ in range(max(0, limite - int(desde * fps))):
            ok, frame = cap.read()
            if not ok:
                break
            for d in motor.procesar_frame(frame):
                if d["estado"] in ("CONFIRMADO", "ANCLADO"):
                    pistas[d["track_id"]] = d["nombre"]
        cap.release()

        for fila in esperadas:
            esperado = fila["identidad"]
            obtenido = pistas.get(int(fila["pista"]))
            resultados.append({
                "archivo": archivo,
                "pista": int(fila["pista"]),
                "esperado": esperado,
                "obtenido": obtenido or "—",
                "nota": fila["nota"],
                "baseline": fila.get("baseline", "0"),
            })

    aciertos = errores = falsos = perdidos = conocidos = mejoras = 0
    print(f"\n{'video':<28} {'pista':>5} {'esperado':>12} {'obtenido':>12} {'resultado':>11}")
    for r in resultados:
        baseline = r.get("baseline") == "1"
        if r["esperado"] == "DESCONOCIDO":
            if r["obtenido"] == "—":
                estado = "OK"; aciertos += 1
            elif baseline:
                estado = "CONOCIDO"; conocidos += 1
            else:
                estado = "FALSO POS."; falsos += 1
        elif r["obtenido"] == r["esperado"]:
            estado = "OK"; aciertos += 1
        elif r["obtenido"] == "—":
            estado = "PERDIDA"; perdidos += 1
        else:
            estado = "ERROR"; errores += 1
        if baseline and estado == "OK":
            mejoras += 1
        print(f"{Path(r['archivo']).name:<28} {r['pista']:5d} {r['esperado']:>12} "
              f"{r['obtenido']:>12} {estado:>11}")

    fallos = errores + falsos + perdidos
    print(f"\nAciertos: {aciertos} | errores: {errores} | falsos positivos: {falsos} "
          f"| perdidas: {perdidos} | conocidos (baseline): {conocidos} | total: {len(resultados)}")
    if mejoras:
        print(f"El límite conocido mejoró en {mejoras} caso(s): actualiza el baseline.")
    print(f"Umbral de aceptación: {max_fallos} fallo(s)")

    if fallos > max_fallos:
        print("RESULTADO: FALLA")
        return 1
    print("RESULTADO: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
