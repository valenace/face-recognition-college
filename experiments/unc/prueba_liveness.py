"""
Prueba de liveness controlada por operador (experiments/unc/prueba_liveness.py)

El operador presenta rostros ante la cámara y anota con el teclado qué está
presentando: cara real, foto en papel, foto en teléfono, etc. Cada frame se
evalúa con el MISMO modelo de anti-spoofing que usa el motor en producción
(MiniFASNet vía `uniface.spoofing.create_spoofer`) y, si hay identidad
cargada, se reconoce también a quién pertenece el rostro (pgvector + ArcFace).

Con eso se generan por corrida:

  liveness_frames.csv      1 fila por frame: etiqueta GT, score, decisión, id
  liveness_segmentos.csv   1 fila por segmento (escenario) con su resumen
  umbrales.csv             sensibilidad en 0.50 / 0.75 / 0.90
  umbrales_sujeto.csv      lo mismo, restringido a operador vs sujeto mostrado
  matriz_confusion.csv     por frame, por escenario y por sujeto mostrado
  por_identidad.csv        decisión por identidad detectada
  informe_liveness.md      informe listo para citar, con capturas auditables

También acepta `--video` / `--carpeta` para re-evaluar grabaciones ya hechas
(etiqueta GT fija con `--etiqueta`), útil para validar el script sin cámara
o para reprocesar capturas.

Uso (webcam):
    venv/bin/python experiments/unc/prueba_liveness.py
    venv/bin/python experiments/unc/prueba_liveness.py --fuente 0 --umbral 0.90

Teclas durante la corrida:
    1 marca segmento REAL (tu cara)
    2 marca segmento PAPEL (foto impresa)
    3 marca segmento TELEFONO (foto que muestra el teléfono)
    4 marca segmento OTRA (otro ataque)
    e cierra el segmento actual y queda en pausa
    q termina la corrida y escribe el informe

Uso (archivo grabado, sin teclado):
    venv/bin/python experiments/unc/prueba_liveness.py --video r.mp4 --etiqueta real --sin-ventana
    venv/bin/python experiments/unc/prueba_liveness.py --carpeta frames/ --etiqueta telefono --sin-ventana

Reanálisis de una corrida ya guardada (no repite la prueba):
    venv/bin/python experiments/unc/prueba_liveness.py --reinforme <carpeta_de_la_corrida>
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from uniface import create_detector, create_recognizer
from uniface.spoofing import create_spoofer

from core.db_config import cargar_centroides, alias_de

RAIZ = Path(__file__).resolve().parents[2]
SALIDA_DEFAULT = RAIZ / "experiments/unc/salida_liveness"
UMBRAL_DEFAULT = 0.90
TAMANO_MIN_DEFAULT = 60
VOTOS_DEFAULT = 3
UMBRALES_TABLA = [0.50, 0.75, 0.90]
CAPTURA_CADA_S = 1.0

TECLAS_ETIQUETAS = {
    "1": "real",
    "2": "papel",
    "3": "telefono",
    "4": "otra",
}
ETIQUETAS_VALIDAS = set(TECLAS_ETIQUETAS.values())
EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Funciones puras (testeadas) ─────────────────────────────────────
def score_real(is_real: bool, conf: float) -> float:
    """Score continuo 'realidad': 1 = cara en vivo, 0 = foto.

    MiniFASNet devuelve en `confidence` la probabilidad de la clase ganadora
    (no de 'real'): si la etiqueta fue 'foto' hay que complementar.
    """
    return float(conf) if is_real else 1.0 - float(conf)


def decidir(is_real: bool, conf: float, umbral: float) -> str:
    """Replica la decisión del motor: real si el modelo confirma y confía."""
    if is_real and float(conf) >= umbral:
        return "real"
    return "foto"


def matriz_confusion(filas: list[dict]) -> dict:
    """Matriz por frame con positivo = cara real.

    TP reales aceptados · FN reales rechazados · FP ataques aceptados ·
    TN ataques rechazados. Ignora frames sin cara/omitidos.
    """
    tp = fp = fn = tn = 0
    for f in filas:
        if f["decision"] not in ("real", "foto"):
            continue
        if f["gt_real"]:
            if f["decision"] == "real":
                tp += 1
            else:
                fn += 1
        else:
            if f["decision"] == "real":
                fp += 1
            else:
                tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def metricas_desde_matriz(m: dict) -> dict:
    tp, fp, fn, tn = m["tp"], m["fp"], m["fn"], m["tn"]
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )
    return {
        "n": n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / n if n else 0.0,
        "falsos_aceptados": fp,
        "falsos_rechazados": fn,
    }


def matriz_por_segmento(segmentos: list[dict]) -> dict:
    """1 voto por escenario: decisión por mayoría de frames de cada segmento."""
    tp = fp = fn = tn = 0
    for s in segmentos:
        if s["decision_mayoria"] not in ("real", "foto"):
            continue
        if s["gt_real"]:
            if s["decision_mayoria"] == "real":
                tp += 1
            else:
                fn += 1
        else:
            if s["decision_mayoria"] == "real":
                fp += 1
            else:
                tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def identidad_operador(filas: list[dict]) -> str:
    """ID más frecuente en los segmentos marcados como real.

    Es la identidad de quien opera la cámara (p. ej. `val`). Si no hay
    segmentos reales, cae a la identidad más frecuente de toda la corrida.
    """
    ids = Counter(
        f["id_reconocido"] for f in filas
        if f["etiqueta"] == "real" and f["decision"] in ("real", "foto")
        and f["id_reconocido"] not in ("", None, "Desconocido")
    )
    if ids:
        return ids.most_common(1)[0][0]
    ids = Counter(
        f["id_reconocido"] for f in filas
        if f["decision"] in ("real", "foto")
        and f["id_reconocido"] not in ("", None, "Desconocido")
    )
    return ids.most_common(1)[0][0] if ids else ""


def sujetos_mostrados(filas: list[dict], segmentos: list[dict], operador: str) -> dict:
    """Identidad dominante no-operador de cada segmento de ataque.

    Es el estudiante cuya foto se mostró (p. ej. `kevin` en el teléfono).
    Devuelve {n_segmento: identidad}.
    """
    if not operador:
        return {}
    resultado = {}
    for s in segmentos:
        if s["gt_real"]:
            continue
        n = s["n_segmento"]
        ids = Counter(
            f["id_reconocido"] for f in filas
            if f["segmento"] == n and f["decision"] in ("real", "foto")
            and f["id_reconocido"] not in ("", None, "Desconocido", operador)
        )
        if ids:
            resultado[n] = ids.most_common(1)[0][0]
    return resultado


def matriz_sujeto_mostrado(filas: list[dict], mostrados: dict,
                           operador: str) -> dict:
    """Matriz limpia: operador real vs foto del estudiante mostrado.

    - Positivos: frames de segmentos `real` (la cara del operador).
    - Negativos: frames de ataques cuya identidad detectada es el sujeto
      mostrado, no el operador que quedó en cuadro.
    """
    tp = fp = fn = tn = 0
    for f in filas_sujeto_mostrado(filas, mostrados, operador):
        if f["gt_real"]:
            if f["decision"] == "real":
                tp += 1
            else:
                fn += 1
        else:
            if f["decision"] == "real":
                fp += 1
            else:
                tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def filas_sujeto_mostrado(filas: list[dict], mostrados: dict,
                          operador: str) -> list[dict]:
    """Subconjunto limpio: operador en `real` + sujeto mostrado en ataques."""
    salida = []
    for f in filas:
        if f["decision"] not in ("real", "foto"):
            continue
        if f["etiqueta"] == "real":
            if f["id_reconocido"] in ("", "Desconocido", operador):
                salida.append(f)
        elif mostrados.get(f["segmento"]) == f["id_reconocido"]:
            salida.append(f)
    return salida


def tabla_por_identidad(filas: list[dict]) -> list[dict]:
    """Resumen por identidad detectada (vacío/Desconocido se agrupa como '—')."""
    grupos = defaultdict(lambda: {
        "frames": 0, "real": 0, "foto": 0, "scores": [], "segmentos": set(),
    })
    for f in filas:
        if f["decision"] not in ("real", "foto"):
            continue
        ident = f["id_reconocido"] or "—"
        if ident == "Desconocido":
            ident = "—"
        g = grupos[ident]
        g["frames"] += 1
        g["real" if f["decision"] == "real" else "foto"] += 1
        if f["score_real"] != "":
            g["scores"].append(float(f["score_real"]))
        g["segmentos"].add(str(f["segmento"]))
    salida = []
    for ident, g in sorted(grupos.items(), key=lambda kv: -kv[1]["frames"]):
        salida.append({
            "id_reconocido": ident,
            "frames": g["frames"],
            "frames_real": g["real"],
            "frames_foto": g["foto"],
            "pct_foto": round(100 * g["foto"] / g["frames"], 1),
            "score_mediana": round(float(np.median(g["scores"])), 3) if g["scores"] else "",
            "score_media": round(float(np.mean(g["scores"])), 3) if g["scores"] else "",
            "segmentos": " ".join(sorted(g["segmentos"])),
        })
    return salida


def voto_estable(decisiones: list[str], k: int = VOTOS_DEFAULT):
    """Índice del 1.er frame con k decisiones iguales consecutivas.

    Devuelve (decision, indice) o (None, None).
    """
    for i in range(len(decisiones) - k + 1):
        if len(set(decisiones[i:i + k])) == 1:
            return decisiones[i], i
    return None, None


def auc_rango(pos: list[float], neg: list[float]) -> float:
    """AUC exacta por pares (Mann-Whitney con empates a 0.5)."""
    if not pos or not neg:
        return 0.0
    mayor = igual = 0
    for p in pos:
        for n in neg:
            if p > n:
                mayor += 1
            elif p == n:
                igual += 1
    return (mayor + 0.5 * igual) / (len(pos) * len(neg))


def estadisticas_segmento(filas: list[dict], etiqueta: str, n_segmento: int,
                          desde: float, hasta: float, votos: int) -> dict:
    """Resumen de un escenario: counts, score y decisión estable."""
    evaluadas = [f for f in filas if f["decision"] in ("real", "foto")]
    decisiones = [f["decision"] for f in evaluadas]
    scores = [f["score_real"] for f in evaluadas]
    decision, idx = voto_estable(decisiones, votos)
    ids = [
        (f["id_reconocido"], f["sim"]) for f in filas
        if f["id_reconocido"] not in ("", "Desconocido", None) and f["sim"] != ""
    ]
    if ids:
        top_id, casos = Counter(n for n, _ in ids).most_common(1)[0]
        sims = [s for n, s in ids if n == top_id]
        sim_media = round(float(np.mean(sims)), 4)
    else:
        top_id, casos, sim_media = "", 0, ""
    return {
        "n_segmento": n_segmento,
        "etiqueta": etiqueta,
        "gt_real": etiqueta == "real",
        "desde": round(desde, 2),
        "hasta": round(hasta, 2),
        "duracion": round(hasta - desde, 2),
        "frames": len(filas),
        "con_cara": len(evaluadas),
        "sin_cara": sum(1 for f in filas if f["decision"] == "sin_cara"),
        "omitidos": sum(1 for f in filas if f["decision"] == "omitido"),
        "frames_real": sum(1 for f in filas if f["decision"] == "real"),
        "frames_foto": sum(1 for f in filas if f["decision"] == "foto"),
        "pct_frames_real": (
            round(100 * sum(1 for f in filas if f["decision"] == "real") / len(evaluadas), 1)
            if evaluadas else ""
        ),
        "score_mediana": round(float(np.median(scores)), 3) if scores else "",
        "score_media": round(float(np.mean(scores)), 3) if scores else "",
        "score_min": round(float(np.min(scores)), 3) if scores else "",
        "score_max": round(float(np.max(scores)), 3) if scores else "",
        "decision_mayoria": (
            Counter(decisiones).most_common(1)[0][0] if decisiones else ""
        ),
        "s_voto_estable": (
            round(evaluadas[idx]["segundo"] - desde, 2) if decision is not None else ""
        ),
        "id_reconocido": top_id,
        "id_casos": casos,
        "sim_media": sim_media,
    }


# ── Presentación ───────────────────────────────────────────────────
def fmt(x, n: int = 3) -> str:
    if x is None or x == "":
        return "—"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{float(x):.{n}f}"


def anotar(frame, etiqueta: str, fila: dict, bbox=None):
    img = frame.copy()
    color = (0, 180, 0) if fila["decision"] == "real" else (
        (0, 0, 220) if fila["decision"] == "foto" else (160, 160, 160)
    )
    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    lineas = [
        f"Segmento {fila['segmento'] or '-'}  {etiqueta.upper()}",
        f"score_real {fmt(fila['score_real'])}  -> {fila['decision'].upper()}",
        f"id {fila['id_reconocido'] or '-'}  sim {fmt(fila['sim'])}",
        f"cara {fmt(fila['ancho_cara'], 0)} px  fps {fmt(fila['fps'], 1)}",
        "1 real  2 papel  3 telefono  4 otra  e pausa  q salir",
    ]
    y = 28
    for texto in lineas:
        cv2.putText(img, texto, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, texto, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    color, 1, cv2.LINE_AA)
        y += 28
    return img


# ── Modelos ────────────────────────────────────────────────────────
def cargar_identidad():
    """Centroides desde pgvector. Devuelve (nombres, matriz) o None."""
    try:
        db = cargar_centroides()
        if not db:
            return None
        claves = sorted(db)
        nombres = [alias_de(k, k) for k in claves]
        matriz = np.array(
            [np.asarray(db[k]).reshape(-1) for k in claves], dtype="float32"
        )
        normas = np.linalg.norm(matriz, axis=1, keepdims=True)
        matriz = matriz / np.maximum(normas, 1e-6)
        return nombres, matriz
    except Exception as e:
        print(f"[AVISO] Identidad desactivada ({e})")
        return None


def reconocer_rostro(nombres, matriz, recognizer, frame, landmarks):
    try:
        vec = recognizer.get_normalized_embedding(frame, landmarks)
    except Exception:
        return "Desconocido", ""
    if vec is None or matriz is None:
        return "Desconocido", ""
    sims = matriz @ vec.reshape(-1)
    i = int(np.argmax(sims))
    return nombres[i], round(float(sims[i]), 4)


# ── Lectura de frames ──────────────────────────────────────────────
def crear_lector(args):
    """Devuelve (leer_frame, descripcion, es_webcam)."""
    if args.carpeta:
        rutas = sorted(
            p for p in Path(args.carpeta).iterdir()
            if p.suffix.lower() in EXTENSIONES_IMAGEN
        )
        estado = {"rutas": rutas}

        def leer():
            if not estado["rutas"]:
                return False, None
            return True, cv2.imread(str(estado["rutas"].pop(0)))

        return leer, f"carpeta:{args.carpeta}", False
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {args.video}")
        return (lambda: cap.read()), args.video, False
    fuente = int(args.fuente) if str(args.fuente).isdigit() else args.fuente
    cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        raise RuntimeError(
            f"No se pudo abrir la cámara '{fuente}'. "
            "Revisa /dev/video* o usa --video/--carpeta."
        )
    return (lambda: cap.read()), f"camara:{fuente}", True


# ── CSV e informe ──────────────────────────────────────────────────
def escribir_csv_frames(ruta: Path, filas: list[dict]) -> None:
    campos = ["segundo", "segmento", "etiqueta", "gt_real", "decision",
              "is_real", "score_real", "conf", "conf_det", "ancho_cara",
              "id_reconocido", "sim", "fps"]
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for fila in filas:
            w.writerow({k: fila.get(k, "") for k in campos})


def escribir_csv_segmentos(ruta: Path, segmentos: list[dict]) -> None:
    if not segmentos:
        return
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(segmentos[0].keys()))
        w.writeheader()
        w.writerows(segmentos)


def tabla_umbrales(filas: list[dict]) -> list[dict]:
    salida = []
    evaluadas = [f for f in filas if f["decision"] in ("real", "foto")]
    for t in UMBRALES_TABLA:
        m = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for f in evaluadas:
            # Misma regla que el motor: la clase ganadora debe ser 'real' y
            # superar el umbral de confianza. No se usa el score continuo.
            decision = "real" if (f["is_real"] and f["conf"] >= t) else "foto"
            if f["gt_real"]:
                m["tp" if decision == "real" else "fn"] += 1
            else:
                m["fp" if decision == "real" else "tn"] += 1
        met = metricas_desde_matriz(m)
        salida.append({
            "umbral": t,
            **m,
            "precision": round(met["precision"], 4),
            "recall": round(met["recall"], 4),
            "f1": round(met["f1"], 4),
            "accuracy": round(met["accuracy"], 4),
            "n": met["n"],
        })
    return salida


def escribir_csv_umbrales(ruta: Path, filas: list[dict]) -> None:
    filas_t = tabla_umbrales(filas)
    if not filas_t:
        return
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas_t[0].keys()))
        w.writeheader()
        w.writerows(filas_t)


def escribir_csv_umbrales_sujeto(ruta: Path, filas: list[dict],
                                 segmentos: list[dict]) -> None:
    operador = identidad_operador(filas)
    mostrados = sujetos_mostrados(filas, segmentos, operador)
    subset = filas_sujeto_mostrado(filas, mostrados, operador)
    if not subset:
        return
    filas_t = tabla_umbrales(subset)
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas_t[0].keys()))
        w.writeheader()
        w.writerows(filas_t)


def escribir_csv_matriz(ruta: Path, filas: list[dict], segmentos: list[dict]) -> None:
    operador = identidad_operador(filas)
    mostrados = sujetos_mostrados(filas, segmentos, operador)
    mlimpia = matriz_sujeto_mostrado(filas, mostrados, operador)
    filas_m = []
    for nivel, m in [
        ("frame_todos", matriz_confusion(filas)),
        ("segmento_todos", matriz_por_segmento(segmentos)),
    ]:
        met = metricas_desde_matriz(m)
        filas_m.append({
            "nivel": nivel,
            **m,
            "precision": round(met["precision"], 4),
            "recall": round(met["recall"], 4),
            "f1": round(met["f1"], 4),
            "accuracy": round(met["accuracy"], 4),
        })
    if sum(mlimpia.values()) > 0:
        met = metricas_desde_matriz(mlimpia)
        filas_m.append({
            "nivel": "sujeto_mostrado",
            **mlimpia,
            "precision": round(met["precision"], 4),
            "recall": round(met["recall"], 4),
            "f1": round(met["f1"], 4),
            "accuracy": round(met["accuracy"], 4),
        })
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas_m[0].keys()))
        w.writeheader()
        w.writerows(filas_m)


def escribir_csv_identidad(ruta: Path, filas: list[dict]) -> None:
    filas_id = tabla_por_identidad(filas)
    if not filas_id:
        return
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas_id[0].keys()))
        w.writeheader()
        w.writerows(filas_id)


def escribir_informe(ruta: Path, meta: dict, filas: list[dict],
                     segmentos: list[dict]) -> None:
    lineas = []
    add = lineas.append
    evaluadas = [f for f in filas if f["decision"] in ("real", "foto")]

    add("# Informe de prueba de liveness")
    add("")
    add(f"**Fecha:** {meta['fecha']}  ")
    add(f"**Fuente:** `{meta['fuente']}` · **Umbral:** {meta['umbral']} · "
        f"**Votos estables:** {meta['votos']} · "
        f"**Tamaño mínimo:** {meta['tamano_min']} px  ")
    add(f"**Duración:** {meta['duracion']:.1f} s · **Frames:** {len(filas)} · "
        f"**Con cara evaluada:** {len(evaluadas)} · "
        f"**Escenarios:** {len(segmentos)}  ")
    add(f"**Modelo:** MiniFASNet (uniface 3.3.0), el mismo del motor. "
        f"**Identidad:** {'ArcFace + pgvector' if meta['identidad'] else 'desactivada'}")
    add("")
    add("> `score_real` es continuo: 1 = cara en vivo, 0 = foto. La decisión")
    add("> binaria replica el motor: `real` solo si el modelo confirma y")
    add(f"> `confidence >= {meta['umbral']}`. `omitido` = cara menor de")
    add(f"> {meta['tamano_min']} px o perfil extremo; `sin_cara` = sin detección.")
    add("")
    add("## 1. Resultados por escenario")
    add("")
    add("| # | Escenario | GT | Duración | Frames | Con cara | Omitidos | Sin cara | Frames real | Frames foto | % real | Score mediana | Decisión | s voto estable | ID | Sim media |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in segmentos:
        add(
            f"| {s['n_segmento']} | {s['etiqueta']} | {'real' if s['gt_real'] else 'ataque'} | "
            f"{s['duracion']:.2f} | {s['frames']} | {s['con_cara']} | {s['omitidos']} | "
            f"{s['sin_cara']} | {s['frames_real']} | {s['frames_foto']} | "
            f"{fmt(s['pct_frames_real'], 1)} % | {fmt(s['score_mediana'])} | "
            f"{s['decision_mayoria'] or '—'} | {fmt(s['s_voto_estable'], 2)} | "
            f"{s['id_reconocido'] or '—'} | {fmt(s['sim_media'], 3)} |"
        )
    add("")
    add("`% real` se calcula sobre frames con cara evaluada, no sobre el total.")
    add("")
    add("> Si durante un ataque el operador queda en cuadro, el `ID` dominante del")
    add("> escenario puede ser el operador y no el estudiante mostrado. La tabla")
    add("> 2.1 separa ambos casos por identidad detectada.")
    add("")

    mf = matriz_confusion(filas)
    metf = metricas_desde_matriz(mf)
    ms = matriz_por_segmento(segmentos)
    mets = metricas_desde_matriz(ms)
    operador = identidad_operador(filas)
    mostrados = sujetos_mostrados(filas, segmentos, operador)
    mlimpia = matriz_sujeto_mostrado(filas, mostrados, operador)
    metlimpia = metricas_desde_matriz(mlimpia)

    add("## 2. Matriz y métricas")
    add("")
    add("| Nivel | TP | FP | FN | TN | Precision | Recall | F1 | Accuracy |")
    add("|---|---|---|---|---|---|---|---|---|")
    add(f"| Por frame (todos) | {mf['tp']} | {mf['fp']} | {mf['fn']} | {mf['tn']} | "
        f"{metf['precision']:.3f} | {metf['recall']:.3f} | {metf['f1']:.3f} | "
        f"{metf['accuracy']:.3f} |")
    add(f"| Por escenario (todos) | {ms['tp']} | {ms['fp']} | {ms['fn']} | {ms['tn']} | "
        f"{mets['precision']:.3f} | {mets['recall']:.3f} | {mets['f1']:.3f} | "
        f"{mets['accuracy']:.3f} |")
    limpia_n = sum(mlimpia.values())
    if limpia_n:
        add(f"| **Por sujeto mostrado** | {mlimpia['tp']} | {mlimpia['fp']} | "
            f"{mlimpia['fn']} | {mlimpia['tn']} | {metlimpia['precision']:.3f} | "
            f"{metlimpia['recall']:.3f} | {metlimpia['f1']:.3f} | "
            f"{metlimpia['accuracy']:.3f} |")
    add("")
    add("Positivo = cara real. TP: real aceptada; FN: real rechazada; FP: ataque "
        "aceptado; TN: ataque rechazado.")
    add("")
    if limpia_n:
        add("> **Por sujeto mostrado** es la lectura limpia: toma como positivos los")
        add(f"> frames de los escenarios `real` (operador: `{operador or '—'}`) y como")
        add("> negativos solo los frames de los ataques cuya identidad detectada es el")
        add("> sujeto mostrado (la foto del estudiante), no el operador que quedó en")
        add("> cuadro. Los frames del operador dentro de un escenario de ataque se")
        add("> excluyen de esta matriz. Las dos primeras filas sí los incluyen, por eso")
        add("> pueden mostrar falsos aceptados que en realidad son caras reales.")
        if mostrados:
            add(">")
            add("> Sujetos mostrados: " + ", ".join(
                f"segmento {n} → `{m}`" for n, m in sorted(mostrados.items())
            ) + ".")
        add("")
    else:
        add("> No se pudo calcular **Por sujeto mostrado**: falta un segmento")
        add("> `real` reconocido o un ataque con identidad reconocida.")
        add("")
    add("La tabla de umbrales usa la regla del motor (`is_real` y `confidence >= t`),")
    add("no solo el score continuo. El AUC usa el score continuo.")
    add("")
    base = mlimpia if (mlimpia["tp"] + mlimpia["fp"] + mlimpia["fn"] + mlimpia["tn"]) else mf
    if base["tp"] + base["fn"] == 0:
        add("> Sin frames reales: precision, recall y F1 de la clase real no son")
        add("> estimables; la accuracy de 1.000 solo refleja que todos los ataques")
        add("> se rechazaron.")
        add("")
    if base["fp"] + base["tn"] == 0:
        add("> Sin frames de ataque: especificidad y FPR no son estimables; la")
        add("> accuracy de 1.000 solo refleja que todas las caras reales se")
        add("> aceptaron.")
        add("")

    add("### 2.1 Decisión por identidad detectada")
    add("")
    add("| ID detectada | Frames | Frames real | Frames foto | % foto | Score mediana | Segmentos |")
    add("|---|---|---|---|---|---|---|")
    for r in tabla_por_identidad(filas):
        add(f"| {r['id_reconocido']} | {r['frames']} | {r['frames_real']} | "
            f"{r['frames_foto']} | {r['pct_foto']:.1f} % | {fmt(r['score_mediana'])} | "
            f"{r['segmentos']} |")
    add("")
    add("Esta tabla separa automáticamente a quien opera la cámara del estudiante")
    add("mostrado en el teléfono o papel: un ataque correcto aparece como")
    add("`ID del estudiante` con 100 % de frames `foto`.")
    add("")

    add("## 3. Sensibilidad del umbral (por frame)")
    add("")
    add("| Umbral | TP | FP | FN | TN | Precision | Recall | F1 | Accuracy |")
    add("|---|---|---|---|---|---|---|---|---|")
    for r in tabla_umbrales(filas):
        add(f"| {r['umbral']:.2f} | {r['tp']} | {r['fp']} | {r['fn']} | {r['tn']} | "
            f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
            f"{r['accuracy']:.3f} |")
    add("")
    pos = [f["score_real"] for f in evaluadas if f["gt_real"]]
    neg = [f["score_real"] for f in evaluadas if not f["gt_real"]]
    if pos and neg:
        add(f"AUC del score continuo (crudo, reales vs todos los frames de ataque): "
            f"**{auc_rango(pos, neg):.4f}** ({len(pos)} reales, {len(neg)} de ataque).")
    else:
        add("AUC del score continuo (crudo): **no estimable** (faltan frames reales "
            "o de ataque).")
    if not neg:
        add("")
        add("> No hubo frames de ataque en esta corrida: precision, especificidad "
            "y AUC no son estimables. Repite la prueba con papel y teléfono.")
    if not pos:
        add("")
        add("> No hubo frames reales en esta corrida: recall y AUC no son "
            "estimables. Repite la prueba con tu cara en vivo.")
    add("")

    subset = filas_sujeto_mostrado(filas, mostrados, operador)
    pos_l = [f["score_real"] for f in subset if f["gt_real"]]
    neg_l = [f["score_real"] for f in subset if not f["gt_real"]]
    if pos_l and neg_l:
        add("### 3.1 Sensibilidad por sujeto mostrado")
        add("")
        add("| Umbral | TP | FP | FN | TN | Precision | Recall | F1 | Accuracy |")
        add("|---|---|---|---|---|---|---|---|---|")
        for r in tabla_umbrales(subset):
            add(f"| {r['umbral']:.2f} | {r['tp']} | {r['fp']} | {r['fn']} | {r['tn']} | "
                f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
                f"{r['accuracy']:.3f} |")
        add("")
        add(f"AUC del score continuo (por sujeto mostrado): "
            f"**{auc_rango(pos_l, neg_l):.4f}** ({len(pos_l)} reales, "
            f"{len(neg_l)} del sujeto mostrado).")
        add("")

    add("## 4. Lectura")
    add("")
    if mlimpia["tp"] + mlimpia["fp"] + mlimpia["fn"] + mlimpia["tn"]:
        n_ataques = mlimpia["tn"] + mlimpia["fp"]
        n_reales = mlimpia["tp"] + mlimpia["fn"]
        add(f"- **Falsos aceptados (sujeto mostrado):** {mlimpia['fp']} frames.")
        add(f"- **Falsos rechazados (operador):** {mlimpia['fn']} frames.")
        if n_ataques:
            add(f"- Del sujeto mostrado se rechaza el "
                f"{100 * mlimpia['tn'] / n_ataques:.1f} % de los frames "
                f"({mlimpia['tn']}/{n_ataques}).")
        if n_reales:
            add(f"- Del operador se acepta el "
                f"{100 * mlimpia['tp'] / n_reales:.1f} % de los frames "
                f"({mlimpia['tp']}/{n_reales}).")
        add(f"- En la matriz cruda (todas las filas) aparecen {mf['fp']} frames de "
            f"ataque aceptados y {mf['fn']} reales rechazados; esa matriz mezcla "
            "los frames del operador que quedaron en cuadro durante el ataque. La "
            "fila **Por sujeto mostrado** los excluye.")
    else:
        falso_acepto = [f for f in filas if not f["gt_real"] and f["decision"] == "real"]
        falso_rech = [f for f in filas if f["gt_real"] and f["decision"] == "foto"]
        add(f"- **Falsos aceptados (ataque aceptado):** {len(falso_acepto)} frames.")
        add(f"- **Falsos rechazos (cara real rechazada):** {len(falso_rech)} frames.")
        ataques = [f for f in evaluadas if not f["gt_real"]]
        if ataques:
            pct = 100 * sum(1 for f in ataques if f["decision"] == "foto") / len(ataques)
            add(f"- En los ataques, el {pct:.1f} % de los frames se rechazan como foto.")
        reales = [f for f in evaluadas if f["gt_real"]]
        if reales:
            pct = 100 * sum(1 for f in reales if f["decision"] == "real") / len(reales)
            add(f"- En las caras reales, el {pct:.1f} % de los frames se aceptan como real.")
    add("")
    if len(segmentos) > 1:
        add("La decisión por escenario (mayoría de frames) es la que se acerca al "
            "uso real con votos; la tabla de umbrales muestra el compromiso "
            "precisión/recall si se mueve el corte.")
    add("")

    add("## 5. Identidad reconocida")
    add("")
    if not meta["identidad"]:
        add("Identidad desactivada (`--sin-identidad`); no se evaluó en esta corrida.")
        add("")
    else:
        add("La identidad es solo informativa: el anti-spoofing decide si la cara")
        add("está viva; ArcFace dice a quién se parece. En un ataque con la foto de")
        add("un estudiante, lo esperable es que la identidad coincida con ese")
        add("estudiante y la decisión sea `foto`.")
        add("")
        add("| # | Escenario | Operador | Sujeto mostrado |")
        add("|---|---|---|---|")
        for s in segmentos:
            mostrado = "—" if s["gt_real"] else mostrados.get(s["n_segmento"], "—")
            add(f"| {s['n_segmento']} | {s['etiqueta']} | {operador or '—'} | {mostrado} |")
        add("")
        add("Los conteos por identidad y el score de liveness de cada una están en")
        add("la tabla 2.1 y en `por_identidad.csv`.")
        add("")

    add("## 6. Capturas y archivos")
    add("")
    add("- `liveness_frames.csv`: una fila por frame con la decisión y el score.")
    add("- `liveness_segmentos.csv`: resumen por escenario.")
    add("- `umbrales.csv`: precisión/recall/F1 en 0.50, 0.75 y 0.90.")
    add("- `umbrales_sujeto.csv`: lo mismo, restringido a operador vs sujeto")
    add("  mostrado (lectura limpia).")
    add("- `matriz_confusion.csv`: matrices por frame, por escenario y **por")
    add("  sujeto mostrado**.")
    add("- `por_identidad.csv`: decisión por identidad detectada (sección 2.1).")
    add("- `capturas/`: un frame anotado por segundo (si no se usó `--sin-capturas`).")
    add("")
    add("## 7. Validez")
    add("")
    add("- Es una prueba **controlada de demostración**, no una estimación")
    add("  poblacional: pocos escenarios, un operador y una sola sesión.")
    add("- **Matriz por sujeto mostrado:** asume que el operador es la identidad")
    add("  dominante de los segmentos `real` y que el sujeto mostrado es la")
    add("  identidad no-operador dominante de cada ataque. Si el ataque no se")
    add("  reconoce o el operador no aparece, esa fila queda vacía.")
    add("- Para que la matriz cruda del escenario sea limpia, el teléfono debe")
    add("  cubrir el cuadro; si el operador queda visible, usar la fila **Por")
    add("  sujeto mostrado**.")
    add("- El motor real exige **3 votos** y omite caras <60 px; aquí se reporta")
    add("  el frame suelto y la mayoría por escenario, que es más estricta.")
    add("- El brillo del teléfono, el reflejo de la pantalla y la distancia")
    add("  influyen en MiniFASNet. Repite cada escenario 2-3 veces y con luz")
    add("  estable antes de sacar conclusiones.")
    add("- La identidad depende de que el estudiante esté registrado en pgvector")
    add("  y de que la foto del teléfono conserve rasgos suficientes.")
    add("")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text("\n".join(lineas), encoding="utf-8")


# ── Principal ──────────────────────────────────────────────────────
def _fila_desde_csv(fila: dict) -> dict:
    """Convierte los strings de liveness_frames.csv a tipos usables."""
    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return ""

    def booleano(x):
        if x == "True":
            return True
        if x == "False":
            return False
        return None

    return {
        "segundo": num(fila.get("segundo")),
        "segmento": int(fila["segmento"]) if fila.get("segmento") else "",
        "etiqueta": fila.get("etiqueta", ""),
        "gt_real": booleano(fila.get("gt_real")),
        "decision": fila.get("decision", ""),
        "is_real": booleano(fila.get("is_real")),
        "score_real": num(fila.get("score_real")),
        "conf": num(fila.get("conf")),
        "conf_det": num(fila.get("conf_det")),
        "ancho_cara": num(fila.get("ancho_cara")),
        "id_reconocido": fila.get("id_reconocido", ""),
        "sim": num(fila.get("sim")),
        "fps": num(fila.get("fps")),
    }


def reanalizar(directorio: Path, args) -> None:
    """Regenera el informe y los CSV de una corrida ya guardada."""
    csv_frames = directorio / "liveness_frames.csv"
    if not csv_frames.exists():
        raise SystemExit(f"No existe {csv_frames}")

    with open(csv_frames, encoding="utf-8") as f:
        filas = [_fila_desde_csv(r) for r in csv.DictReader(f)]
    filas = [f for f in filas if f["segmento"] != ""]

    grupos = defaultdict(list)
    for fila in filas:
        grupos[fila["segmento"]].append(fila)

    segmentos = []
    for n in sorted(grupos):
        grupo = grupos[n]
        etiqueta = grupo[0]["etiqueta"] or "real"
        desde = min(float(f["segundo"]) for f in grupo)
        hasta = max(float(f["segundo"]) for f in grupo)
        segmentos.append(
            estadisticas_segmento(grupo, etiqueta, int(n), desde, hasta, args.votos)
        )

    fuente = str(directorio)
    umbral = args.umbral
    informe_viejo = directorio / "informe_liveness.md"
    if informe_viejo.exists():
        texto = informe_viejo.read_text(encoding="utf-8")
        m = re.search(r"\*\*Fuente:\*\* `([^`]+)`", texto)
        if m:
            fuente = m.group(1)
        m = re.search(r"\*\*Umbral:\*\* ([\d.]+)", texto)
        if m:
            umbral = float(m.group(1))

    meta = {
        "fecha": time.strftime("%Y-%m-%d %H:%M:%S") + " (reanálisis)",
        "fuente": fuente,
        "umbral": umbral,
        "votos": args.votos,
        "tamano_min": args.tamano_min,
        "duracion": max((float(f["segundo"]) for f in filas), default=0.0),
        "identidad": any(f["id_reconocido"] not in ("", None) for f in filas),
    }
    escribir_csv_segmentos(directorio / "liveness_segmentos.csv", segmentos)
    escribir_csv_umbrales(directorio / "umbrales.csv", filas)
    escribir_csv_umbrales_sujeto(directorio / "umbrales_sujeto.csv", filas, segmentos)
    escribir_csv_matriz(directorio / "matriz_confusion.csv", filas, segmentos)
    escribir_csv_identidad(directorio / "por_identidad.csv", filas)
    escribir_informe(directorio / "informe_liveness.md", meta, filas, segmentos)
    print(f"[OK] Reanálisis de {directorio}")
    print(f"     Frames: {len(filas)} · Escenarios: {len(segmentos)}")


def main():
    parser = argparse.ArgumentParser(description="Prueba de liveness controlada por operador")
    parser.add_argument("--fuente", default="0", help="Índice de cámara (0) u otro índice")
    parser.add_argument("--video", default="", help="Ruta de video en lugar de cámara")
    parser.add_argument("--carpeta", default="", help="Carpeta de imágenes en lugar de cámara")
    parser.add_argument("--etiqueta", default="real",
                        help="Etiqueta GT para --video/--carpeta (real|papel|telefono|otra)")
    parser.add_argument("--umbral", type=float, default=UMBRAL_DEFAULT,
                        help="Umbral de confianza del liveness (0.90 en el motor)")
    parser.add_argument("--tamano-min", type=int, default=TAMANO_MIN_DEFAULT,
                        help="Ancho mínimo de cara para evaluar liveness")
    parser.add_argument("--votos", type=int, default=VOTOS_DEFAULT,
                        help="Votos consecutivos iguales para considerar estable")
    parser.add_argument("--salida", default="", help="Carpeta de salida de la corrida")
    parser.add_argument("--sin-identidad", action="store_true", help="No reconocer identidad")
    parser.add_argument("--sin-ventana", action="store_true", help="No abrir ventana (headless)")
    parser.add_argument("--sin-capturas", action="store_true", help="No guardar capturas")
    parser.add_argument("--max-segundos", type=float, default=0, help="Fin forzado tras N s")
    parser.add_argument("--max-frames", type=int, default=0, help="Fin forzado tras N frames")
    parser.add_argument("--reinforme", default="",
                        help="Reanaliza una corrida existente (carpeta con liveness_frames.csv)")
    args = parser.parse_args()

    if args.reinforme:
        reanalizar(Path(args.reinforme), args)
        return

    if args.video or args.carpeta:
        if args.etiqueta not in ETIQUETAS_VALIDAS:
            parser.error(f"--etiqueta debe ser una de: {sorted(ETIQUETAS_VALIDAS)}")

    salida = Path(args.salida) if args.salida else (
        SALIDA_DEFAULT / time.strftime("run_%Y%m%d_%H%M%S")
    )
    capturas = salida / "capturas"
    capturas.mkdir(parents=True, exist_ok=True)

    try:
        leer_frame, fuente, es_webcam = crear_lector(args)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print("[INFO] Cargando modelos (detector, recognizer, spoofer)...")
    detector = create_detector("retinaface")
    spoofer = create_spoofer()
    identidad = None
    recognizer = None
    if not args.sin_identidad:
        identidad = cargar_identidad()
        if identidad:
            recognizer = create_recognizer("arcface")
            print(f"[OK] Identidad cargada: {len(identidad[0])} centroides")
        else:
            print("[AVISO] Sin identidad: no se reconocerá a quién pertenece el rostro")

    # En archivos, toda la corrida es un solo segmento con --etiqueta.
    if args.video or args.carpeta:
        etiqueta_actual = args.etiqueta
        n_segmento = 1
        t_segmento_inicio = 0.0
    else:
        etiqueta_actual = "pausa"
        n_segmento = 0
        t_segmento_inicio = None

    filas: list[dict] = []
    buffer: list[dict] = []
    segmentos: list[dict] = []
    t_inicio = time.time()
    t_ultimo_frame = 0.0
    t_ultima_captura = 0.0
    i_captura = 0
    medidor = {"t0": t_inicio, "frames": 0}

    csv_frames = salida / "liveness_frames.csv"
    csv_frames.parent.mkdir(parents=True, exist_ok=True)
    csv_handle = open(csv_frames, "w", newline="", encoding="utf-8")
    campos = ["segundo", "segmento", "etiqueta", "gt_real", "decision",
              "is_real", "score_real", "conf", "conf_det", "ancho_cara",
              "id_reconocido", "sim", "fps"]
    writer = csv.DictWriter(csv_handle, fieldnames=campos)
    writer.writeheader()
    csv_abierto = True

    def cerrar_segmento(hasta: float):
        nonlocal buffer
        if etiqueta_actual != "pausa" and buffer:
            segmentos.append(estadisticas_segmento(
                buffer, etiqueta_actual, n_segmento,
                t_segmento_inicio if t_segmento_inicio is not None else 0.0,
                hasta, args.votos,
            ))
        buffer = []

    print(
        "\n[INFO] Marca el escenario con el teclado ANTES de presentar el rostro:\n"
        "  1 = REAL (tu cara en vivo)\n"
        "  2 = PAPEL (foto impresa)\n"
        "  3 = TELEFONO (foto en el teléfono)\n"
        "  4 = OTRA (otro tipo de ataque)\n"
        "  e = cerrar segmento actual (pausa)\n"
        "  q = terminar y escribir informe\n"
    )

    try:
        while True:
            ok, frame = leer_frame()
            if not ok or frame is None:
                break
            t_frame = time.time() - t_inicio
            if args.max_segundos and t_frame >= args.max_segundos:
                break
            if args.max_frames and len(filas) >= args.max_frames:
                break

            medidor["frames"] += 1
            fps = medidor["frames"] / max(1e-6, time.time() - medidor["t0"])

            caras = detector.detect(frame)
            rostro = None
            bbox = None
            mejor_ancho = 0.0
            for c in caras:
                ancho = float(c.bbox[2] - c.bbox[0])
                if ancho > mejor_ancho:
                    rostro, bbox, mejor_ancho = c, c.bbox, ancho

            decision = "sin_cara"
            is_real = None
            conf = None
            conf_det = ""
            id_n = ""
            sim = ""
            if rostro is not None:
                conf_det = round(float(getattr(rostro, "confidence", 0.0)), 4)
                if mejor_ancho < args.tamano_min:
                    decision = "omitido"
                else:
                    res = spoofer.predict(frame, bbox)
                    is_real = bool(res.is_real)
                    conf = round(float(res.confidence), 4)
                    decision = decidir(is_real, conf, args.umbral)
                    if identidad and recognizer is not None and rostro.landmarks is not None:
                        id_n, sim = reconocer_rostro(
                            identidad[0], identidad[1], recognizer, frame, rostro.landmarks
                        )

            fila = {
                "segundo": round(t_frame, 2),
                "segmento": n_segmento if etiqueta_actual != "pausa" else "",
                "etiqueta": etiqueta_actual if etiqueta_actual != "pausa" else "",
                "gt_real": etiqueta_actual == "real" if etiqueta_actual != "pausa" else "",
                "decision": decision,
                "is_real": is_real,
                "score_real": (
                    round(score_real(is_real, conf), 4)
                    if decision in ("real", "foto") else ""
                ),
                "conf": conf,
                "conf_det": conf_det,
                "ancho_cara": round(mejor_ancho, 1) if rostro is not None else "",
                "id_reconocido": id_n,
                "sim": sim,
                "fps": round(fps, 2),
            }

            if etiqueta_actual != "pausa":
                filas.append(fila)
                buffer.append(fila)
                writer.writerow({
                    k: (
                        "True" if fila[k] is True else
                        "False" if fila[k] is False else
                        fila[k] if fila[k] is not None else ""
                    )
                    for k in campos
                })
                csv_handle.flush()

            t_now = time.time()
            if (
                not args.sin_capturas
                and etiqueta_actual != "pausa"
                and t_now - t_ultima_captura > CAPTURA_CADA_S
            ):
                t_ultima_captura = t_now
                i_captura += 1
                cv2.imwrite(
                    str(capturas / f"cap_seg{n_segmento}_{etiqueta_actual}_{i_captura:04d}.jpg"),
                    anotar(frame, etiqueta_actual, fila, bbox),
                )

            if not args.sin_ventana:
                cv2.imshow("Prueba de liveness", anotar(frame, etiqueta_actual, fila, bbox))
                tecla = cv2.waitKey(1) & 0xFF
            else:
                tecla = 255

            if tecla in (ord("q"), ord("Q"), 27):
                break
            if tecla == ord("e"):
                cerrar_segmento(t_frame)
                etiqueta_actual = "pausa"
                t_segmento_inicio = None
                continue
            tecla_texto = chr(tecla) if tecla != 255 else ""
            if tecla_texto in TECLAS_ETIQUETAS:
                nueva = TECLAS_ETIQUETAS[tecla_texto]
                if nueva == etiqueta_actual:
                    continue
                cerrar_segmento(t_frame)
                n_segmento += 1
                etiqueta_actual = nueva
                t_segmento_inicio = t_frame
                print(f"[SEG] #{n_segmento} {nueva.upper()} desde t={t_frame:.1f}s")

            t_ultimo_frame = t_frame
    except KeyboardInterrupt:
        print("\n[INFO] Interrumpido; escribiendo informe con lo acumulado...")
    finally:
        if etiqueta_actual != "pausa":
            cerrar_segmento(t_ultimo_frame)
        if csv_abierto:
            csv_handle.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        escribir_csv_frames(csv_frames, filas)
        escribir_csv_segmentos(salida / "liveness_segmentos.csv", segmentos)
        escribir_csv_umbrales(salida / "umbrales.csv", filas)
        escribir_csv_umbrales_sujeto(salida / "umbrales_sujeto.csv", filas, segmentos)
        escribir_csv_matriz(salida / "matriz_confusion.csv", filas, segmentos)
        escribir_csv_identidad(salida / "por_identidad.csv", filas)
        meta = {
            "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
            "fuente": fuente,
            "umbral": args.umbral,
            "votos": args.votos,
            "tamano_min": args.tamano_min,
            "duracion": time.time() - t_inicio,
            "identidad": identidad is not None,
        }
        informe = salida / "informe_liveness.md"
        escribir_informe(informe, meta, filas, segmentos)
        print(
            f"\n[OK] Corrida en {salida}\n"
            f"     Frames: {len(filas)} · Escenarios: {len(segmentos)} · "
            f"Segundos: {meta['duracion']:.1f}\n"
            f"     Informe: {informe}"
        )
        if not filas:
            print("     (Sin frames registrados: revisa la cámara o la carpeta)")


if __name__ == "__main__":
    main()
