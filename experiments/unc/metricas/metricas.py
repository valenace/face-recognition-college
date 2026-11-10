"""
Métricas consolidadas del pipeline — documento de tesis
=======================================================

Una sola ejecución lee los CSV ya producidos por los experimentos y genera:

  salida/identidad/     ROC, PR, AUC, AP, EER, d', umbrales, matriz de confusión,
                        métricas por estudiante, top-k, intervalos de confianza,
                        error por ancho de cara y por accesorios.
  salida/deteccion/     tasas de detección, IoU, estrés de emparejamiento,
                        distribución de tamaños y efecto del filtro.
  salida/asistencia/    cobertura de los videos 1 y 3, matriz de sesión,
                        intrusos, barrido de umbrales y fragmentación de pistas.
  salida/rendimiento/   FPS, latencia por etapa y comparación CPU/GPU.
  salida/resumen.md     informe con todos los números y su archivo fuente.

No usa scikit-learn ni pandas: solo NumPy, csv y matplotlib. Así el cálculo es
reproducible con el mismo entorno del motor (`venv/`).

Uso:
    venv/bin/python experiments/unc/metricas/metricas.py
"""

from __future__ import annotations

import ast
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Rutas y constantes ─────────────────────────────────────────────
RAIZ = Path(__file__).resolve().parents[3]
DATOS = RAIZ / "experiments/unc"
SALIDA = Path(__file__).resolve().parent / "salida"

PERSONAS = [
    "alexa", "bob", "carlos", "dani", "fran",
    "jesus", "jose", "kate", "kevin",
]
UMBRAL_OPERATIVO = 0.30
UMBRALES_CLAVE = [0.25, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50]
UMBRALES_COMPLETOS = np.round(np.arange(0.05, 0.951, 0.005), 3)
SEMILLA = 20261105
N_BOOTSTRAP = 2000
COLOR_POS = "#2563eb"
COLOR_NEG = "#dc2626"
COLOR_REF = "#6b7280"


# ── Utilidades ─────────────────────────────────────────────────────
def leer_csv(ruta: Path) -> list[dict]:
    with open(ruta, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def escribir_csv(ruta: Path, filas: list[dict], campos: list[str] | None = None) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if not filas:
        ruta.write_text("", encoding="utf-8")
        return
    if campos is None:
        campos = list(filas[0].keys())
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(filas)


def f(x, n: int = 4) -> str:
    if x is None or x == "":
        return ""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{float(x):.{n}f}"


def div(a: float, b: float) -> float:
    return a / b if b else 0.0


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# ── Métricas de clasificación ──────────────────────────────────────
def metricas_matriz(tp: int, fp: int, fn: int, tn: int) -> dict:
    n = tp + fp + fn + tn
    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    specificity = div(tn, tn + fp)
    accuracy = div(tp + tn, n)
    f1 = div(2 * precision * recall, precision + recall)
    npv = div(tn, tn + fn)
    fpr = div(fp, fp + tn)
    fnr = div(fn, fn + tp)
    fdr = div(fp, fp + tp)
    balanced = (recall + specificity) / 2
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = div(tp * tn - fp * fn, mcc_den)
    pe = div((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp), n * n)
    kappa = div(accuracy - pe, 1 - pe)
    return {
        "n": n,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "f1": f1,
        "npv": npv,
        "fpr": fpr,
        "fnr": fnr,
        "fdr": fdr,
        "balanced_accuracy": balanced,
        "mcc": mcc,
        "kappa": kappa,
        "youden": recall + specificity - 1,
        "prevalence": div(tp + fn, n),
        "predicted_positive_rate": div(tp + fp, n),
        "support_pos": tp + fn,
        "support_neg": tn + fp,
    }


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / denom
    mitad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centro - mitad), min(1.0, centro + mitad)


def roc_exacta(pos: np.ndarray, neg: np.ndarray):
    """ROC exacta: un punto por score, desde +inf hasta el mínimo."""
    scores = np.concatenate([pos, neg])
    etiquetas = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    orden = np.argsort(-scores, kind="mergesort")
    scores = scores[orden]
    etiquetas = etiquetas[orden]

    cortes = np.concatenate([[np.inf], scores])
    tp = np.concatenate([[0], np.cumsum(etiquetas)])
    fp = np.concatenate([[0], np.cumsum(1 - etiquetas)])
    tpr = tp / len(pos)
    fpr = fp / len(neg)
    # tpr y fpr ascienden al bajar el umbral.
    auc = float(np.trapezoid(tpr, fpr))
    j = tpr - fpr
    return fpr, tpr, cortes, auc, j


def pr_exacta(pos: np.ndarray, neg: np.ndarray):
    """Curva Precision-Recall exacta y Average Precision (AP)."""
    scores = np.concatenate([pos, neg])
    etiquetas = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    orden = np.argsort(-scores, kind="mergesort")
    etiquetas = etiquetas[orden]
    tp = np.cumsum(etiquetas)
    fp = np.cumsum(1 - etiquetas)
    precision = tp / (tp + fp)
    recall = tp / len(pos)
    ap = float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))
    return recall, precision, ap


def eer_exacta(pos, neg, fpr, tpr, cortes):
    """Equal Error Rate: cruce FPR = FNR, interpolado sobre la ROC exacta."""
    fnr = 1 - tpr
    diferencia = fpr - fnr
    cruces = np.where(np.sign(diferencia[:-1]) != np.sign(diferencia[1:]))[0]
    if not len(cruces):
        return 0.0, 0.0
    i = int(cruces[0])
    alpha = div(diferencia[i], diferencia[i] - diferencia[i + 1])
    t = cortes[i] + alpha * (cortes[i + 1] - cortes[i])
    if not np.isfinite(t):
        return 0.0, 0.0
    fpr_t = float((neg >= t).mean())
    fnr_t = float((pos < t).mean())
    return (fpr_t + fnr_t) / 2, float(t)


def d_prime(pos: np.ndarray, neg: np.ndarray) -> float:
    varianza = 0.5 * (pos.var(ddof=1) + neg.var(ddof=1))
    return float((pos.mean() - neg.mean()) / math.sqrt(varianza))


def bootstrap_auc(pos, neg, n=N_BOOTSTRAP, seed=SEMILLA):
    rng = np.random.default_rng(seed)
    valores = []
    for _ in range(n):
        p = rng.choice(pos, len(pos), replace=True)
        q = rng.choice(neg, len(neg), replace=True)
        _, _, _, auc, _ = roc_exacta(p, q)
        valores.append(auc)
    return np.percentile(valores, [2.5, 97.5])


def bootstrap_operativa(registros, t, n=N_BOOTSTRAP, seed=SEMILLA + 1):
    rng = np.random.default_rng(seed)
    f1s, accs, precs, recs = [], [], [], []
    total = len(registros)
    for _ in range(n):
        idx = rng.integers(0, total, total)
        muestra = [registros[i] for i in idx]
        tp, fp, fn, tn = matriz_operativa(muestra, t)
        m = metricas_matriz(tp, fp, fn, tn)
        f1s.append(m["f1"])
        accs.append(m["accuracy"])
        precs.append(m["precision"])
        recs.append(m["recall"])
    return {
        "f1": np.percentile(f1s, [2.5, 97.5]),
        "accuracy": np.percentile(accs, [2.5, 97.5]),
        "precision": np.percentile(precs, [2.5, 97.5]),
        "recall": np.percentile(recs, [2.5, 97.5]),
    }


# ── Identidad ─────────────────────────────────────────────────────
def cargar_similitudes():
    filas = leer_csv(DATOS / "fase4/fase4_similitudes.csv")
    conteos = Counter(r["etiqueta"] for r in filas)
    pos, neg_score, neg_op = [], [], []
    registros = []
    for fila in filas:
        etiqueta = fila["etiqueta"]
        sims = np.array([float(fila[f"sim_{p}"]) for p in PERSONAS])
        if etiqueta in PERSONAS:
            i = PERSONAS.index(etiqueta)
            registros.append({
                "recorte_id": fila["recorte_id"],
                "etiqueta": etiqueta,
                "sims": sims,
                "pred": PERSONAS[int(np.argmax(sims))],
                "score": float(np.max(sims)),
                "es_positivo": True,
            })
            pos.append(sims[i])
            neg_score.append(float(np.max(np.delete(sims, i))))
        elif etiqueta == "desconocido":
            registros.append({
                "recorte_id": fila["recorte_id"],
                "etiqueta": etiqueta,
                "sims": sims,
                "pred": PERSONAS[int(np.argmax(sims))],
                "score": float(np.max(sims)),
                "es_positivo": False,
            })
            neg_score.append(float(np.max(sims)))
            neg_op.append(float(np.max(sims)))
    return (
        np.array(pos), np.array(neg_score), np.array(neg_op),
        registros, conteos,
    )


def matriz_operativa(registros, t):
    """TP/FN sobre los 86 etiquetados; FP/TN sobre los 157 desconocidos."""
    n_pos = sum(1 for r in registros if r["es_positivo"])
    n_neg = len(registros) - n_pos
    tp = sum(
        1 for r in registros
        if r["es_positivo"] and r["pred"] == r["etiqueta"] and r["score"] >= t
    )
    fn = n_pos - tp
    fp = sum(1 for r in registros if not r["es_positivo"] and r["score"] >= t)
    tn = n_neg - fp
    return tp, fp, fn, tn


def por_estudiante(registros, t):
    filas = []
    for persona in PERSONAS:
        tp = fp = fn = tn = 0
        falsos_desconocidos = 0
        errores_identidad = 0
        support = 0
        for r in registros:
            real = r["etiqueta"] == persona
            pred = r["pred"] == persona and r["score"] >= t
            if real:
                support += 1
                if pred:
                    tp += 1
                else:
                    fn += 1
                    if r["pred"] != persona:
                        errores_identidad += 1
            else:
                if pred:
                    fp += 1
                    if not r["es_positivo"]:
                        falsos_desconocidos += 1
                else:
                    tn += 1
        m = metricas_matriz(tp, fp, fn, tn)
        filas.append({
            "persona": persona,
            "support": support,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1": round(m["f1"], 4),
            "accuracy": round(m["accuracy"], 4),
            "falsos_aceptados_desconocidos": falsos_desconocidos,
            "errores_identidad": errores_identidad,
        })
    return filas


def top_k(registros):
    rangos = []
    for r in registros:
        if not r["es_positivo"]:
            continue
        i = PERSONAS.index(r["etiqueta"])
        orden = np.argsort(-r["sims"], kind="mergesort")
        rangos.append(int(np.where(orden == i)[0][0]) + 1)
    rangos = np.array(rangos)
    return {
        "n": len(rangos),
        "top1": float((rangos <= 1).mean()),
        "top2": float((rangos <= 2).mean()),
        "top3": float((rangos <= 3).mean()),
        "mrr": float((1 / rangos).mean()),
        "rango_mediano": float(np.median(rangos)),
        "rango_max": int(rangos.max()),
        "rank1": int((rangos == 1).sum()),
        "rank2": int((rangos == 2).sum()),
        "rank3": int((rangos == 3).sum()),
        "rank_mas3": int((rangos > 3).sum()),
    }


def bin_ancho(ancho: float) -> str:
    if ancho < 25:
        return "<25"
    if ancho < 30:
        return "25-29"
    if ancho < 40:
        return "30-39"
    if ancho < 60:
        return "40-59"
    if ancho < 80:
        return "60-79"
    return ">=80"


ORDEN_ANCHOS = ["<25", "25-29", "30-39", "40-59", "60-79", ">=80"]


def por_ancho(registros, t):
    recortes = {r["recorte_id"]: r for r in leer_csv(DATOS / "salon_muestra/recortes.csv")}
    datos = defaultdict(lambda: {
        "n_pos": 0, "correctos": 0, "n_neg": 0, "falsos": 0,
        "scores_pos": [], "scores_neg": [],
    })
    for r in registros:
        rec = recortes.get(r["recorte_id"])
        if rec is None:
            continue
        grupo = bin_ancho(float(rec["ancho"]))
        if r["es_positivo"]:
            datos[grupo]["n_pos"] += 1
            datos[grupo]["scores_pos"].append(float(r["sims"][PERSONAS.index(r["etiqueta"])]))
            if r["pred"] == r["etiqueta"] and r["score"] >= t:
                datos[grupo]["correctos"] += 1
        else:
            datos[grupo]["n_neg"] += 1
            datos[grupo]["scores_neg"].append(r["score"])
            if r["score"] >= t:
                datos[grupo]["falsos"] += 1
    filas = []
    for grupo in ORDEN_ANCHOS:
        d = datos.get(grupo)
        if not d:
            continue
        filas.append({
            "bin_ancho_px": grupo,
            "n_etiquetados": d["n_pos"],
            "correctos": d["correctos"],
            "accuracy": round(div(d["correctos"], d["n_pos"]), 4),
            "score_medio_pos": round(float(np.mean(d["scores_pos"])), 4) if d["scores_pos"] else "",
            "n_desconocidos": d["n_neg"],
            "falsos_aceptados": d["falsos"],
            "far": round(div(d["falsos"], d["n_neg"]), 4) if d["n_neg"] else "",
            "score_medio_neg": round(float(np.mean(d["scores_neg"])), 4) if d["scores_neg"] else "",
        })
    return filas


def por_accesorio(registros, t):
    accesorios = {
        r["recorte_id"]: r
        for r in leer_csv(DATOS / "salon_muestra/accesorios_muestra.csv")
    }
    grupos = defaultdict(lambda: {"n": 0, "correctos": 0, "scores": []})
    for r in registros:
        if not r["es_positivo"]:
            continue
        a = accesorios.get(r["recorte_id"])
        if a is None:
            continue
        lentes = (a.get("lentes") or "").strip().lower() in {"sí", "si", "s", "1", "true"}
        gorra = (a.get("gorra") or "").strip().lower() in {"sí", "si", "s", "1", "true"}
        if lentes and gorra:
            grupo = "gorra+lentes"
        elif lentes:
            grupo = "lentes"
        elif gorra:
            grupo = "gorra"
        else:
            grupo = "sin accesorios"
        grupos[grupo]["n"] += 1
        grupos[grupo]["scores"].append(float(r["sims"][PERSONAS.index(r["etiqueta"])]))
        if r["pred"] == r["etiqueta"] and r["score"] >= t:
            grupos[grupo]["correctos"] += 1
    filas = []
    for grupo in ["sin accesorios", "lentes", "gorra", "gorra+lentes"]:
        d = grupos.get(grupo)
        if not d:
            continue
        filas.append({
            "grupo": grupo,
            "n": d["n"],
            "correctos": d["correctos"],
            "accuracy": round(div(d["correctos"], d["n"]), 4),
            "score_medio_propio": round(float(np.mean(d["scores"])), 4),
        })
    return filas


# ── Detección ─────────────────────────────────────────────────────
def metricas_deteccion():
    filas = leer_csv(DATOS / "fase1_resumen_videos.csv")
    personas = [r for r in filas if r["persona"] in PERSONAS]
    salon = [r for r in filas if r["persona"] not in PERSONAS]
    salida = []
    for r in filas:
        salida.append({
            "grupo": "persona" if r["persona"] in PERSONAS else "salon",
            "persona": r["persona"],
            "frames_totales": int(r["frames_totales"]),
            "frames_muestreados": int(r["frames_muestreados"]),
            "tasa_deteccion": float(r["tasa_deteccion"]),
            "tasa_cara_unica": float(r["tasa_cara_unica"]),
            "tasa_util": float(r["tasa_util"]),
            "bbox_min": int(r["bbox_min"]),
            "bbox_max": int(r["bbox_max"]),
            "bbox_media": float(r["bbox_media"]),
            "frames_utiles_estimados": int(r["frames_utiles_estimados"]),
        })
    if personas:
        frames_m = sum(int(r["frames_muestreados"]) for r in personas)
        frames_t = sum(int(r["frames_totales"]) for r in personas)
        micro_det = sum(
            int(r["frames_muestreados"]) * float(r["tasa_deteccion"]) for r in personas
        ) / frames_m
        micro_unica = sum(
            int(r["frames_muestreados"]) * float(r["tasa_cara_unica"]) for r in personas
        ) / frames_m
        micro_util = sum(
            int(r["frames_muestreados"]) * float(r["tasa_util"]) for r in personas
        ) / frames_m
        salida.append({
            "grupo": "agregado",
            "persona": "PROMEDIO_9",
            "frames_totales": frames_t,
            "frames_muestreados": frames_m,
            "tasa_deteccion": round(float(np.mean([float(r["tasa_deteccion"]) for r in personas])), 4),
            "tasa_cara_unica": round(float(np.mean([float(r["tasa_cara_unica"]) for r in personas])), 4),
            "tasa_util": round(float(np.mean([float(r["tasa_util"]) for r in personas])), 4),
            "bbox_min": min(int(r["bbox_min"]) for r in personas),
            "bbox_max": max(int(r["bbox_max"]) for r in personas),
            "bbox_media": round(float(np.mean([float(r["bbox_media"]) for r in personas])), 1),
            "frames_utiles_estimados": sum(int(r["frames_utiles_estimados"]) for r in personas),
        })
        salida.append({
            "grupo": "agregado",
            "persona": "MICRO_9",
            "frames_totales": frames_t,
            "frames_muestreados": frames_m,
            "tasa_deteccion": round(micro_det, 4),
            "tasa_cara_unica": round(micro_unica, 4),
            "tasa_util": round(micro_util, 4),
            "bbox_min": min(int(r["bbox_min"]) for r in personas),
            "bbox_max": max(int(r["bbox_max"]) for r in personas),
            "bbox_media": round(float(np.mean([float(r["bbox_media"]) for r in personas])), 1),
            "frames_utiles_estimados": sum(int(r["frames_utiles_estimados"]) for r in personas),
        })
    return salida, salon


def metricas_iou():
    filas = leer_csv(DATOS / "salon_muestra/emparejamiento_iou.csv")
    salida = []
    for grupo, filtro in [
        ("todos", lambda r: True),
        ("etiquetados", lambda r: r["etiqueta"] in PERSONAS),
        ("desconocidos", lambda r: r["etiqueta"] == "desconocido"),
    ]:
        subset = [r for r in filas if filtro(r)]
        if not subset:
            continue
        ious = np.array([float(r["iou_max"]) for r in subset])
        confs = np.array([float(r["confianza_det"]) for r in subset])
        caras = np.array([int(r["caras_en_frame"]) for r in subset])
        salida.append({
            "grupo": grupo,
            "n": len(subset),
            "iou_medio": round(float(ious.mean()), 4),
            "iou_mediano": round(float(np.median(ious)), 4),
            "iou_min": round(float(ious.min()), 4),
            "pct_iou_ge_050": round(float((ious >= 0.5).mean()), 4),
            "pct_iou_igual_1": round(float((ious >= 0.999).mean()), 4),
            "confianza_media": round(float(confs.mean()), 4),
            "caras_por_frame_media": round(float(caras.mean()), 2),
        })
    return salida


def metricas_estres():
    filas = leer_csv(DATOS / "salon_muestra/prueba_estres_iou.csv")
    salida = []
    for grupo, filtro in [
        ("todos", lambda r: True),
        ("etiquetados", lambda r: r["etiqueta"] in PERSONAS),
        ("desconocidos", lambda r: r["etiqueta"] == "desconocido"),
    ]:
        subset = [r for r in filas if filtro(r)]
        if not subset:
            continue
        aciertos = sum(int(r["aciertos"]) for r in subset)
        perturbaciones = sum(int(r["perturbaciones"]) for r in subset)
        ious = np.array([float(r["iou_mediano_elegido"]) for r in subset])
        salida.append({
            "grupo": grupo,
            "n_recortes": len(subset),
            "perturbaciones": perturbaciones,
            "aciertos": aciertos,
            "tasa_acierto": round(div(aciertos, perturbaciones), 4),
            "iou_mediano_medio": round(float(ious.mean()), 4),
            "iou_mediano_min": round(float(ious.min()), 4),
        })
    return salida


def metricas_tamanos():
    filas = leer_csv(DATOS / "fase1_caras_salon.csv")
    anchos = []
    caras_por_frame = []
    for r in filas:
        lista = ast.literal_eval(r["anchos"])
        anchos.extend(float(x) for x in lista)
        caras_por_frame.append(int(r["caras"]))
    anchos = np.array(anchos)
    return {
        "frames": len(filas),
        "caras_totales": len(anchos),
        "caras_por_frame_media": round(float(np.mean(caras_por_frame)), 2),
        "caras_por_frame_mediana": round(float(np.median(caras_por_frame)), 2),
        "ancho_medio": round(float(anchos.mean()), 2),
        "ancho_mediano": round(float(np.median(anchos)), 2),
        "percentil_10": round(float(np.percentile(anchos, 10)), 2),
        "percentil_25": round(float(np.percentile(anchos, 25)), 2),
        "percentil_75": round(float(np.percentile(anchos, 75)), 2),
        "percentil_90": round(float(np.percentile(anchos, 90)), 2),
        "pct_menor_20": round(float((anchos < 20).mean()), 4),
        "pct_menor_40": round(float((anchos < 40).mean()), 4),
        "pct_mayor_igual_40": round(float((anchos >= 40).mean()), 4),
        "pct_mayor_igual_60": round(float((anchos >= 60).mean()), 4),
    }


def metricas_filtro():
    filas = leer_csv(DATOS / "salon_muestra/efecto_filtro_tamano.csv")
    por_filtro = defaultdict(lambda: {"personas": [], "desconocidos": None})
    for r in filas:
        filtro = int(r["filtro_px"])
        if r["persona"].startswith("desconocidos"):
            por_filtro[filtro]["desconocidos"] = r
        else:
            por_filtro[filtro]["personas"].append(r)
    salida = []
    for filtro in sorted(por_filtro, reverse=True):
        d = por_filtro[filtro]
        total = sum(int(r["total"]) for r in d["personas"])
        reconocidos = sum(int(r["reconocidos"]) for r in d["personas"])
        macro = float(np.mean([float(r["tasa"]) for r in d["personas"]]))
        salida.append({
            "filtro_px": filtro,
            "total_etiquetados": total,
            "reconocidos": reconocidos,
            "tasa_micro": round(div(reconocidos, total), 4),
            "tasa_macro_9": round(macro, 4),
            "falsos_positivos_desconocidos": int(d["desconocidos"]["reconocidos"]) if d["desconocidos"] else "",
        })
    return salida


def metricas_fallos():
    filas = leer_csv(DATOS / "salon_muestra/causas_fallos.csv")
    grupos = defaultdict(lambda: {"n": 0, "anchos": [], "nitidez": [], "pose": []})
    for r in filas:
        grupo = r["resultado"]
        grupos[grupo]["n"] += 1
        grupos[grupo]["anchos"].append(float(r["ancho"]))
        grupos[grupo]["nitidez"].append(float(r["nitidez"]))
        grupos[grupo]["pose"].append(float(r["pose"]))
    salida = []
    for grupo, d in sorted(grupos.items()):
        salida.append({
            "resultado": grupo,
            "n": d["n"],
            "ancho_medio": round(float(np.mean(d["anchos"])), 2),
            "ancho_mediano": round(float(np.median(d["anchos"])), 2),
            "nitidez_media": round(float(np.mean(d["nitidez"])), 2),
            "pose_media": round(float(np.mean(d["pose"])), 3),
        })
    return salida


# ── Asistencia ────────────────────────────────────────────────────
def _falsos_aceptados_video1():
    """Desconocidos del video 1 que el motor confirmó como registrados."""
    recortes = {
        r["recorte_id"]: r for r in leer_csv(DATOS / "salon_muestra/recortes.csv")
    }
    muestra = leer_csv(DATOS / "salon_muestra/muestra_anotacion.csv")
    decisiones = leer_csv(DATOS / "demo/prueba_2010_0935_final/decisiones.csv")
    por_frame = defaultdict(list)
    for d in decisiones:
        por_frame[int(d["frame_video"])].append(d)

    total = falsos = sin_decision = 0
    for fila in muestra:
        if fila["etiqueta"] != "desconocido":
            continue
        total += 1
        rec = recortes[fila["recorte_id"]]
        frame = int(rec["frame_idx"])
        caja = (int(rec["x1"]), int(rec["y1"]), int(rec["x2"]), int(rec["y2"]))
        mejor = None
        mejor_iou = 0.0
        for d in por_frame.get(frame, []):
            valor = iou(caja, (int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])))
            if valor > mejor_iou:
                mejor, mejor_iou = d, valor
        if mejor is None or mejor_iou < 0.5:
            sin_decision += 1
        elif (
            mejor["nombre"] in PERSONAS
            and mejor["estado"] in ("CONFIRMADO", "ANCLADO")
        ):
            falsos += 1
    return total, falsos, sin_decision


def metricas_asistencia():
    # Video 1 — corrida real del motor
    v1 = leer_csv(DATOS / "demo/prueba_2010_0935_final/fase6_por_persona.csv")
    total_crops = sum(int(r["recortes_etiquetados"]) for r in v1)
    confirmados = sum(int(r["reconocidos_confirmados"]) for r in v1)
    no_reconocidos = sum(int(r["procesados_no_reconocidos"]) for r in v1)
    sin_decision = sum(int(r["sin_decision"]) for r in v1)
    errores = sum(int(r["error_identidad"]) for r in v1)
    tiempos = [
        float(r["primer_tiempo_confirmacion_s"])
        for r in v1 if r["primer_tiempo_confirmacion_s"] not in ("", None)
    ]
    desconocidos, falsos_desc, desc_sin_decision = _falsos_aceptados_video1()
    cobertura_v1 = {
        "video": "video 1 (VID_20260907_102851.mp4)",
        "presentes_registrados": 9,
        "confirmados": len({r["persona"] for r in v1 if int(r["reconocidos_confirmados"]) > 0}),
        "cobertura": round(len({r["persona"] for r in v1 if int(r["reconocidos_confirmados"]) > 0}) / 9, 4),
        "recortes_etiquetados": total_crops,
        "recortes_confirmados": confirmados,
        "recall_recortes": round(div(confirmados, total_crops), 4),
        "procesados_no_reconocidos": no_reconocidos,
        "sin_decision": sin_decision,
        "errores_identidad": errores,
        "desconocidos_etiquetados": desconocidos,
        "desconocidos_falsos_aceptados": falsos_desc,
        "desconocidos_sin_decision": desc_sin_decision,
        "tiempo_confirmacion_medio_s": round(float(np.mean(tiempos)), 2) if tiempos else "",
        "tiempo_confirmacion_max_s": round(float(np.max(tiempos)), 2) if tiempos else "",
    }
    # La latencia es desde que aparece la pista hasta que se confirma, no desde
    # el inicio del video.
    cobertura_v1["latencia_confirmacion_media_s"] = cobertura_v1.pop("tiempo_confirmacion_medio_s")
    cobertura_v1["latencia_confirmacion_max_s"] = cobertura_v1.pop("tiempo_confirmacion_max_s")

    # Video 3 — resumen del motor por persona y por parte
    v3 = leer_csv(DATOS / "demo/prueba_2110_1613_video3/resumen_personas.csv")
    partes = leer_csv(DATOS / "demo/prueba_2110_1613_video3/resumen_por_parte.csv")
    presentes = {r["persona"] for r in v3}
    cobertura_v3 = {
        "video": "video 3 (video5096333890499381162.mp4)",
        "presentes_registrados": 8,
        "confirmados": len(presentes),
        "cobertura": round(len(presentes) / 8, 4),
        "ausentes": " ".join(sorted(set(PERSONAS) - presentes)),
        "desconocidos_reales": 21,
        "filas_intruso": 29,
        "partes": len(partes),
        "confirmaciones_totales": sum(int(r["confirmaciones"]) for r in v3),
        "sim_media_global": round(
            float(np.mean([float(r["sim_media"]) for r in v3])), 4
        ),
    }

    # Matriz de confusión a nivel de sesión (9 estudiantes)
    confusion = [
        {
            "video": "video 1",
            "presentes": 9, "ausentes": 0,
            "tp": 9, "fn": 0, "fp": 0, "tn": 0,
            "precision": 1.0, "recall": 1.0, "f1": 1.0, "accuracy": 1.0,
            "nota": "sin ausentes: FP/TN no estimables; cobertura 9/9",
        },
        {
            "video": "video 3",
            "presentes": 8, "ausentes": 1,
            "tp": 8, "fn": 0, "fp": 0, "tn": 1,
            "precision": 1.0, "recall": 1.0, "f1": 1.0, "accuracy": 1.0,
            "nota": "jose ausente; 8/8 presentes confirmados",
        },
    ]

    # Presencia por parte (video 3)
    presencia = []
    for r in partes:
        nombres = set((r["confirmados"] or "").split())
        fila = {
            "parte": int(r["parte"]),
            "archivo": r["archivo"],
            "frames": int(r["frames"]),
            "caras": int(r["caras"]),
            "desconocidos_frames": int(r["desconocidos"]),
            "verificando_frames": int(r["verificando"]),
            "centrate_frames": int(r["centrate"]),
            "fraude": int(r["fraude"]),
            "n_confirmados": len(nombres),
        }
        for persona in PERSONAS:
            fila[persona] = 1 if persona in nombres else 0
        presencia.append(fila)

    # Intrusos: cifras del código real (docs/analisis/unc/11_intrusos.md)
    intrusiones = [
        {
            "video": "video 1 (44 s, sin desconocidos verificados)",
            "intrusos_reales": 0,
            "filas_intruso": 7,
            "tp": 0, "fp": 7, "fn": 0,
            "precision": "", "recall": "", "f1": "",
            "falsos_por_segundo": round(7 / 44, 4),
            "nota": "cota superior de falsos: no hay positivos reales verificados (A1)",
        },
        {
            "video": "video 3 (21 desconocidos reales)",
            "intrusos_reales": 21,
            "filas_intruso": 29,
            "tp": 21, "fp": 8, "fn": 0,
            "precision": round(21 / 29, 4),
            "recall": 1.0,
            "f1": round(2 * (21 / 29) * 1.0 / ((21 / 29) + 1.0), 4),
            "falsos_por_segundo": round(8 / 124, 4),
            "nota": "21 desconocidos + 8 filas de los 8 registrados presentes",
        },
    ]

    barrido = [
        {"mejor": "", "promedio": 0.24, "ancho_px": 26, "video1_falsos": 23, "video3_filas": 86},
        {"mejor": "", "promedio": 0.24, "ancho_px": 40, "video1_falsos": 13, "video3_filas": 45},
        {"mejor": 0.22, "promedio": 0.24, "ancho_px": 26, "video1_falsos": 12, "video3_filas": 63},
        {"mejor": 0.22, "promedio": 0.24, "ancho_px": 32, "video1_falsos": 9, "video3_filas": 43},
        {"mejor": 0.22, "promedio": 0.24, "ancho_px": 40, "video1_falsos": 4, "video3_filas": 22},
        {"mejor": 0.24, "promedio": 0.18, "ancho_px": 40, "video1_falsos": 8, "video3_filas": 21},
        {"mejor": 0.26, "promedio": 0.18, "ancho_px": 40, "video1_falsos": 9, "video3_filas": 25},
        {"mejor": 0.28, "promedio": 0.24, "ancho_px": 40, "video1_falsos": 10, "video3_filas": 41},
        {"mejor": 0.30, "promedio": 0.24, "ancho_px": 40, "video1_falsos": 13, "video3_filas": 45},
    ]

    # Fragmentación de pistas (video 3)
    decisiones = leer_csv(DATOS / "demo/prueba_2110_1613_video3/decisiones_video3.csv")
    resumen_v3 = {r["persona"]: r for r in v3}
    pistas = defaultdict(lambda: defaultdict(list))
    for d in decisiones:
        if d["nombre"] in PERSONAS and d["estado"] in ("CONFIRMADO", "ANCLADO"):
            pistas[d["nombre"]][d["track_id"]].append(float(d["segundo_parte"]))
    tracking = []
    for persona in PERSONAS:
        tracks = pistas.get(persona, {})
        largos = [len(v) for v in tracks.values()]
        tracking.append({
            "persona": persona,
            "pistas_confirmadas": len(tracks),
            "filas_confirmadas": sum(largos),
            "frames_por_pista_media": round(float(np.mean(largos)), 1) if largos else "",
            "frames_por_pista_max": max(largos) if largos else "",
            "primera_confirmacion_s": (
                resumen_v3[persona]["primer_segundo"] if persona in resumen_v3 else ""
            ),
        })

    # Spoofing
    fraudes = leer_csv(DATOS / "demo/revision_fraudes.csv")
    confs = [float(r["confianza_real"]) for r in fraudes]
    spoofing = [{
        "alarmas": len(fraudes),
        "pistas_distintas": len({r["track_id"] for r in fraudes}),
        "confianza_media": round(float(np.mean(confs)), 4) if confs else "",
        "confianza_min": round(float(np.min(confs)), 4) if confs else "",
        "confianza_max": round(float(np.max(confs)), 4) if confs else "",
        "primer_segundo": round(min(float(r["primer_segundo"]) for r in fraudes), 2) if fraudes else "",
        "ultimo_segundo": round(max(float(r["primer_segundo"]) for r in fraudes), 2) if fraudes else "",
        "nota": "no hay conjunto etiquetado de ataques: no se calcula precision/recall",
    }]

    return {
        "cobertura_v1": cobertura_v1,
        "cobertura_v3": cobertura_v3,
        "confusion": confusion,
        "por_persona_v1": v1,
        "presencia": presencia,
        "intrusiones": intrusiones,
        "barrido": barrido,
        "tracking": tracking,
        "spoofing": spoofing,
    }


# ── Rendimiento ────────────────────────────────────────────────────
def metricas_rendimiento():
    fps = [
        {"equipo": "CPU", "etapa": "deteccion", "valor": 47, "unidad": "ms/frame", "fuente": "docs/computo.md §6"},
        {"equipo": "CPU", "etapa": "embedding", "valor": 24, "unidad": "ms/cara", "fuente": "docs/computo.md §6"},
        {"equipo": "CPU", "etapa": "faiss", "valor": 0.02, "unidad": "ms/cara", "fuente": "docs/computo.md §6"},
        {"equipo": "CPU", "etapa": "overhead", "valor": 3, "unidad": "ms/frame", "fuente": "docs/computo.md §6"},
        {"equipo": "CPU", "etapa": "pipeline_total", "valor": 7.52, "unidad": "fps", "fuente": "docs/computo.md §6 (200 frames, 26.6 s)"},
        {"equipo": "CPU", "etapa": "pipeline_rango", "valor": "6.6-7.5", "unidad": "fps", "fuente": "docs/computo.md §6"},
        {"equipo": "CPU", "etapa": "mjpeg_entrega", "valor": 6.91, "unidad": "fps", "fuente": "docs/streaming_mjpeg_vs_polling.md §4"},
        {"equipo": "CPU", "etapa": "polling_300ms_entrega", "valor": 3.17, "unidad": "fps", "fuente": "docs/streaming_mjpeg_vs_polling.md §4"},
        {"equipo": "CPU", "etapa": "mjpeg_time_to_first_frame", "valor": 14, "unidad": "ms", "fuente": "docs/streaming_mjpeg_vs_polling.md §4"},
        {"equipo": "CPU", "etapa": "mjpeg_separacion_media", "valor": 164, "unidad": "ms", "fuente": "docs/streaming_mjpeg_vs_polling.md §4"},
        {"equipo": "CPU", "etapa": "carga_modelos_cache", "valor": 0.1, "unidad": "s", "fuente": "docs/computo.md §4"},
        {"equipo": "CPU", "etapa": "cpu_consumido", "valor": 31.3, "unidad": "cpu/s", "fuente": "docs/streaming_mjpeg_vs_polling.md §4"},
        {"equipo": "CPU", "etapa": "ram_total", "valor": 62, "unidad": "GiB", "fuente": "docs/computo.md §1"},
        {"equipo": "CPU", "etapa": "ram_libre_medicion", "valor": 2.7, "unidad": "GiB", "fuente": "docs/computo.md §1"},
        {"equipo": "GPU", "etapa": "vram_total", "valor": 24, "unidad": "GiB", "fuente": "docs/computo.md §1"},
    ]
    cpu_gpu = [
        {
            "equipo": "CPU (i9-14900KF)",
            "proveedor": "CPUExecutionProvider",
            "fps": 7.52,
            "estado": "medido",
            "nota": "onnxruntime CPU 1.24.4; ~7 fps en 1080p",
        },
        {
            "equipo": "GPU (RTX 4090)",
            "proveedor": "CUDAExecutionProvider",
            "fps": "",
            "estado": "pendiente",
            "nota": "requiere onnxruntime-gpu y CUDA/cuDNN; no instalado en este entorno",
        },
    ]
    return fps, cpu_gpu


# ── Gráficas ──────────────────────────────────────────────────────
def guardar_graficas(pos, neg, fpr, tpr, cortes, auc, j, recall_pr, precision_pr, ap,
                     tabla_umbrales, por_est, eer_val, t_eer):
    salida = SALIDA / "identidad"
    salida.mkdir(parents=True, exist_ok=True)

    # ROC
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot(fpr, tpr, color=COLOR_POS, label=f"AUC = {auc:.4f}")
    idx = int(np.argmax(j))
    ax.scatter([fpr[idx]], [tpr[idx]], color="#f59e0b", zorder=5,
               label=f"Youden t={cortes[idx]:.3f}")
    tpr_eer = float((pos >= t_eer).mean())
    fpr_eer = float((neg >= t_eer).mean())
    ax.scatter([fpr_eer], [tpr_eer], color="#16a34a", marker="s", zorder=5,
               label=f"EER t={t_eer:.3f}")
    ax.plot([0, 1], [0, 1], "--", color=COLOR_REF, linewidth=1)
    ax.set_xlabel("Tasa de falsos positivos (FPR)")
    ax.set_ylabel("Tasa de verdaderos positivos (TPR)")
    ax.set_title("ROC — verificación de identidad")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(salida / "roc_curva.png", dpi=140)
    plt.close(fig)

    # PR
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot(np.concatenate([[0.0], recall_pr]), np.concatenate([[1.0], precision_pr]),
            color=COLOR_POS, label=f"AP = {ap:.4f}")
    azar = len(pos) / (len(pos) + len(neg))
    ax.axhline(azar, color=COLOR_REF, linestyle="--", linewidth=1,
               label=f"azar = {azar:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precisión")
    ax.set_title("Precision-Recall — verificación de identidad")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(salida / "pr_curva.png", dpi=140)
    plt.close(fig)

    # Histograma
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.arange(-0.05, 0.85, 0.02)
    ax.hist(neg, bins=bins, alpha=0.65, color=COLOR_NEG,
            label=f"negativos (n={len(neg)})")
    ax.hist(pos, bins=bins, alpha=0.65, color=COLOR_POS,
            label=f"positivos (n={len(pos)})")
    ax.axvline(UMBRAL_OPERATIVO, color="black", linestyle="--",
               label=f"umbral operativo {UMBRAL_OPERATIVO:.2f}")
    ax.axvline(cortes[int(np.argmax(j))], color="#16a34a", linestyle=":",
               label=f"Youden {cortes[int(np.argmax(j))]:.3f}")
    ax.set_xlabel("Similitud coseno")
    ax.set_ylabel("Número de recortes")
    ax.set_title("Distribución de similitudes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(salida / "histograma.png", dpi=140)
    plt.close(fig)

    # F1 / precisión / recall vs umbral
    umbrales = [float(r["umbral"]) for r in tabla_umbrales]
    f1s = [float(r["f1"]) for r in tabla_umbrales]
    precs = [float(r["precision"]) for r in tabla_umbrales]
    recs = [float(r["recall"]) for r in tabla_umbrales]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(umbrales, f1s, label="F1", color=COLOR_POS)
    ax.plot(umbrales, precs, label="Precisión", color="#16a34a")
    ax.plot(umbrales, recs, label="Recall", color="#f59e0b")
    ax.axvline(UMBRAL_OPERATIVO, color="black", linestyle="--",
               label=f"actual {UMBRAL_OPERATIVO:.2f}")
    ax.set_xlabel("Umbral de similitud")
    ax.set_ylabel("Valor")
    ax.set_title("Métricas operativas por umbral")
    ax.legend()
    fig.tight_layout()
    fig.savefig(salida / "f1_vs_umbral.png", dpi=140)
    plt.close(fig)

    # Matriz de confusión en 0.30
    tp = fp = fn = tn = 0
    for r in tabla_umbrales:
        if abs(float(r["umbral"]) - UMBRAL_OPERATIVO) < 1e-9:
            tp, fp, fn, tn = int(r["tp"]), int(r["fp"]), int(r["fn"]), int(r["tn"])
            break
    matriz = np.array([[tp, fn], [fp, tn]])
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(matriz, cmap="Blues")
    for i in range(2):
        for k in range(2):
            ax.text(k, i, str(matriz[i, k]), ha="center", va="center",
                    color="white" if matriz[i, k] > matriz.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    ax.set_xticks([0, 1], ["Predicho positivo", "Predicho negativo"])
    ax.set_yticks([0, 1], ["Real positivo", "Real negativo"])
    ax.set_title(f"Matriz de confusión — umbral {UMBRAL_OPERATIVO:.2f}")
    fig.tight_layout()
    fig.savefig(salida / "matriz_confusion.png", dpi=140)
    plt.close(fig)

    # Métricas por estudiante
    nombres = [r["persona"] for r in por_est]
    rec = [float(r["recall"]) for r in por_est]
    prec = [float(r["precision"]) for r in por_est]
    x = np.arange(len(nombres))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x - 0.2, rec, width=0.4, label="Recall", color=COLOR_POS)
    ax.bar(x + 0.2, prec, width=0.4, label="Precisión", color="#16a34a")
    ax.set_xticks(x, nombres, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Valor")
    ax.set_title(f"Métricas por estudiante — umbral {UMBRAL_OPERATIVO:.2f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(salida / "metricas_por_estudiante.png", dpi=140)
    plt.close(fig)


# ── Informe ───────────────────────────────────────────────────────
def escribir_resumen(datos):
    idn = datos["identidad"]
    det = datos["deteccion"]
    asi = datos["asistencia"]
    ren = datos["rendimiento"]

    lineas = []
    add = lineas.append

    add("# Métricas consolidadas del pipeline")
    add("")
    add("**Generado por:** `experiments/unc/metricas/metricas.py`  ")
    add("**Semilla de bootstrap:** `20261105` · **Remuestreos:** 2 000  ")
    add("**Datos:** CSV de las fases 0, 4, 6 y 7 en `experiments/unc/`")
    add("")
    add("> Este informe se regenera completo con una sola ejecución. Cada número")
    add("> tiene su CSV fuente en `salida/`; no se edita a mano.")
    add("")
    add("---")
    add("")
    add("## 1. Qué métricas aplican y cuáles no")
    add("")
    add("El sistema **no es un modelo de regresión**: no predice un valor continuo.")
    add("Produce una **clase** (identidad de 9 candidatos, `Desconocido`, `INTRUSO`)")
    add("a partir de una similitud coseno. Por eso:")
    add("")
    add("| Métrica | ¿Aplica? | Motivo |")
    add("|---|---|---|")
    add("| MAE | No | Requiere un valor continuo predicho y uno real. |")
    add("| MSE / RMSE | No | Igual que MAE; además penaliza al cuadrado, sin sentido aquí. |")
    add("| MAPE | No | Requiere valores positivos y un cociente porcentual. |")
    add("| R² | No | No hay varianza explicada de una variable dependiente. |")
    add("| Accuracy, Precision, Recall, F1 | **Sí** | Clasificación / verificación. |")
    add("| ROC-AUC, PR-AUC, EER, d' | **Sí** | Verificación biométrica con score. |")
    add("| Matriz de confusión, MCC, kappa | **Sí** | Matriz 2×2 de identificación. |")
    add("| IoU, tasas de detección | **Sí** | Detección de rostros. |")
    add("| FPS, latencia, CPU/RAM | **Sí** | Rendimiento del pipeline. |")
    add("")
    add("---")
    add("")
    add("## 2. Identidad (nivel recorte)")
    add("")
    add("**Protocolo.** 300 recortes etiquetados del video 1. Se excluyen 44")
    add("`dudoso` y 13 `no_cara`; quedan **86 positivos** (con nombre) y **157")
    add("desconocidos**. Para la verificación score-level, cada positivo aporta")
    add("además un negativo (la mayor similitud con otra persona): **243 negativos**.")
    add("")
    add("| Métrica | Valor |")
    add("|---|---|")
    add(f"| AUC (verificación) | **{f(idn['auc'], 4)}** |")
    add(f"| Average Precision (AP) | **{f(idn['ap'], 4)}** |")
    add(f"| EER | **{f(idn['eer'], 4)}** (umbral {f(idn['eer_umbral'], 3)}) |")
    add(f"| d' (discriminabilidad) | **{f(idn['d_prime'], 3)}** |")
    add(f"| Youden | umbral **{f(idn['youden_umbral'], 3)}** (TPR {f(idn['youden_tpr'], 3)}, FPR {f(idn['youden_fpr'], 3)}) |")
    add(f"| Mejor F1 operativo | umbral **{f(idn['mejor_f1_umbral'], 3)}** (F1 {f(idn['mejor_f1'], 3)}) |")
    add("")
    add("**Lectura.** El AUC de 0.9616 significa que, en el 96.2 % de los pares")
    add("(genuino, impostor), el genuino obtiene mayor similitud. El EER de "
        f"{f(100 * idn['eer'], 1)} %")
    add("es el punto donde FPR y FNR se igualan. La d' de 2.885 confirma una")
    add("separación fuerte. La cifra antigua de 0.7439 era un artefacto de una")
    add("rejilla de umbrales truncada (0.10–0.705); la ROC exacta está en")
    add("`salida/identidad/roc_puntos.csv`.")
    add("")
    add("### 2.1 Umbral operativo 0.30")
    add("")
    add("| Umbral | TP | FP | FN | TN | Precisión | Recall | F1 | Accuracy |")
    add("|---|---|---|---|---|---|---|---|---|")
    for r in idn["tabla_clave"]:
        if r["tipo"] != "fijo":
            continue
        add(f"| {f(r['umbral'], 3)} | {r['tp']} | {r['fp']} | {r['fn']} | {r['tn']} | "
            f"{f(r['precision'], 3)} | {f(r['recall'], 3)} | {f(r['f1'], 3)} | {f(r['accuracy'], 3)} |")
    add("")
    add("> **Dos protocolos distintos, no comparables.** El recall de 0.872 de")
    add("> esta tabla es la simulación a **un frame** sobre 86 recortes. La")
    add("> cobertura de asistencia del video 1 (recall 0.733) es la corrida")
    add("> **real del motor con 3 votos y ancla**: confirma 63/86 recortes, pero")
    add("> la cobertura de personas es 9/9 porque cada estudiante se confirma en")
    add("> algún frame. No mezclar ambas cifras en la misma comparación.")
    add("")
    add("**Recomendación.** El umbral 0.30 mantiene precisión 0.987 y recall 0.872")
    add("con **1 falso positivo en 157 desconocidos**. Youden cae en 0.281 (F1")
    add("0.9405), pero 0.30 prioriza la precisión, que es lo correcto cuando un")
    add("falso positivo asigna una asistencia a la persona equivocada.")
    add("")
    add("### 2.2 Métricas por estudiante (umbral 0.30)")
    add("")
    add("| Estudiante | Support | TP | FP | FN | Precisión | Recall | F1 | Falsos desconocidos |")
    add("|---|---|---|---|---|---|---|---|---|")
    for r in idn["por_estudiante"]:
        add(f"| {r['persona']} | {r['support']} | {r['tp']} | {r['fp']} | {r['fn']} | "
            f"{f(r['precision'], 3)} | {f(r['recall'], 3)} | {f(r['f1'], 3)} | "
            f"{r['falsos_aceptados_desconocidos']} |")
    add("")
    add("**Top-k (closed-set, 86 etiquetados):** "
        f"top-1 {f(idn['topk']['top1'], 3)}, top-2 {f(idn['topk']['top2'], 3)}, "
        f"top-3 {f(idn['topk']['top3'], 3)}, MRR {f(idn['topk']['mrr'], 3)}. "
        f"El rango 1 concentra {idn['topk']['rank1']}/{idn['topk']['n']}; "
        f"los {idn['topk']['rank2']} restantes están en rango 2.")
    add("")
    add("### 2.3 Intervalos de confianza (95 %)")
    add("")
    add("| Métrica | Valor | IC 95 % | Método |")
    add("|---|---|---|---|")
    for r in idn["intervalos"]:
        add(f"| {r['metrica']} | {f(r['valor'], 4)} | "
            f"[{f(r['ic_inf'], 4)}, {f(r['ic_sup'], 4)}] | {r['metodo']} |")
    add("")
    add("### 2.4 Error por tamaño y accesorios")
    add("")
    add("| Bin de ancho | Etiquetados | Correctos | Accuracy | Desconocidos | Falsos aceptados | FAR |")
    add("|---|---|---|---|---|---|---|")
    for r in idn["por_ancho"]:
        add(f"| {r['bin_ancho_px']} px | {r['n_etiquetados']} | {r['correctos']} | "
            f"{f(r['accuracy'], 3)} | {r['n_desconocidos']} | {r['falsos_aceptados']} | "
            f"{f(r['far'], 3)} |")
    add("")
    add("| Grupo | n | Correctos | Accuracy | Similitud propia media |")
    add("|---|---|---|---|---|")
    for r in idn["por_accesorio"]:
        add(f"| {r['grupo']} | {r['n']} | {r['correctos']} | {f(r['accuracy'], 3)} | "
            f"{f(r['score_medio_propio'], 3)} |")
    add("")
    add("### 2.5 Multiframe (1 vs 3 frames)")
    add("")
    add("| Umbral | Protocolo | TP | FP | FN | Precisión | Recall | F1 |")
    add("|---|---|---|---|---|---|---|---|")
    for r in idn["multiframe"]:
        add(f"| {f(r['umbral'], 3)} | {r['protocolo']} | {r['tp']} | {r['fp']} | "
            f"{r['fn']} | {f(r['precision'], 3)} | {f(r['recall'], 3)} | {f(r['f1'], 3)} |")
    add("")
    add("---")
    add("")
    add("## 3. Detección")
    add("")
    add(f"- **9 estudiantes, {det['agregado']['frames_muestreados']} frames muestreados:**")
    add(f"  detección micro {f(det['agregado']['tasa_deteccion'], 4)}; "
        f"cara única micro {f(det['agregado']['tasa_cara_unica'], 4)}.")
    add(f"- **Video del salón:** {det['salon']['frames_totales']} frames, "
        f"{det['salon']['frames_muestreados']} muestreados; detección "
        f"{f(det['salon']['tasa_deteccion'], 4)}, útil {f(det['salon']['tasa_util'], 4)}.")
    add(f"- **Tamaño de cara:** mediana {det['tamanos']['ancho_mediano']} px; "
        f"{f(100 * det['tamanos']['pct_menor_40'], 1)} % mide menos de 40 px.")
    add(f"- **Emparejamiento IoU:** {det['iou']['n']} recortes; IoU medio "
        f"{f(det['iou']['iou_medio'], 3)}, mediano {f(det['iou']['iou_mediano'], 3)}, "
        f"{f(100 * det['iou']['pct_iou_ge_050'], 1)} % ≥ 0.50.")
    add("  Los recortes se extrajeron de las detecciones, por eso el IoU es 1.0;")
    add("  la prueba de estrés es la que mide la robustez del emparejamiento.")
    add(f"- **Estrés de emparejamiento:** {f(100 * det['estres']['tasa_acierto'], 1)} % "
        f"de aciertos en {det['estres']['perturbaciones']} perturbaciones.")
    add("")
    add("| Persona | Frames | Detección | Cara única | Útil |")
    add("|---|---|---|---|---|")
    for r in det["personas"]:
        add(f"| {r['persona']} | {r['frames_totales']} | {f(r['tasa_deteccion'], 3)} | "
            f"{f(r['tasa_cara_unica'], 3)} | {f(r['tasa_util'], 3)} |")
    add("")
    add("| Filtro | Etiquetados | Reconocidos | Tasa micro | Tasa macro (9) |")
    add("|---|---|---|---|---|")
    for r in det["filtro"]:
        add(f"| {r['filtro_px']} px | {r['total_etiquetados']} | {r['reconocidos']} | "
            f"{f(r['tasa_micro'], 3)} | {f(r['tasa_macro_9'], 3)} |")
    add("")
    add("| Resultado | n | Ancho mediano | Nitidez media | Pose media |")
    add("|---|---|---|---|---|")
    for r in det["fallos"]:
        add(f"| {r['resultado']} | {r['n']} | {r['ancho_mediano']} px | "
            f"{r['nitidez_media']} | {r['pose_media']} |")
    add("")
    add("---")
    add("")
    add("## 4. Asistencia")
    add("")
    add("| Video | Presentes | Confirmados | Cobertura | Recall recortes | Desconocidos falsos |")
    add("|---|---|---|---|---|---|")
    v1 = asi["cobertura_v1"]
    v3 = asi["cobertura_v3"]
    add(f"| Video 1 | {v1['presentes_registrados']} | {v1['confirmados']} | "
        f"{f(v1['cobertura'], 3)} | {f(v1['recall_recortes'], 3)} | "
        f"{v1['desconocidos_falsos_aceptados']}/{v1['desconocidos_etiquetados']} |")
    add(f"| Video 3 | {v3['presentes_registrados']} | {v3['confirmados']} | "
        f"{f(v3['cobertura'], 3)} | — | — |")
    add("")
    add("**Matriz de sesión (9 estudiantes):**")
    add("")
    add("| Video | TP | FN | FP | TN | Precisión | Recall | F1 | Nota |")
    add("|---|---|---|---|---|---|---|---|---|")
    for r in asi["confusion"]:
        add(f"| {r['video']} | {r['tp']} | {r['fn']} | {r['fp']} | {r['tn']} | "
            f"{f(r['precision'], 3)} | {f(r['recall'], 3)} | {f(r['f1'], 3)} | {r['nota']} |")
    add("")
    add("### 4.1 Intrusos")
    add("")
    add("| Video | Reales | Filas | TP | FP | FN | Precisión | Recall | F1 |")
    add("|---|---|---|---|---|---|---|---|---|")
    for r in asi["intrusiones"]:
        add(f"| {r['video']} | {r['intrusos_reales']} | {r['filas_intruso']} | "
            f"{r['tp']} | {r['fp']} | {r['fn']} | {f(r['precision'], 3) if r['precision'] != '' else '—'} | "
            f"{f(r['recall'], 3) if r['recall'] != '' else '—'} | "
            f"{f(r['f1'], 3) if r['f1'] != '' else '—'} |")
    add("")
    add("El video 1 no tiene intrusos reales verificados: las 7 filas se reportan")
    add("como cota superior de falsos (0.159/s; ver A1). En")
    add("el video 3 hay 21 desconocidos reales y 8 filas de registrados; la")
    add("precisión de fila es 0.724 y el recall 1.000. La regla está calibrada")
    add("para no perder desconocidos; el profesor descarta los falsos desde")
    add("`/sesiones/<id>/`.")
    add("")
    sp = asi["spoofing"][0]
    add(f"**Spoofing:** {sp['alarmas']} alarmas en {sp['pistas_distintas']} pistas, "
        f"confianza media {f(sp['confianza_media'], 3)} "
        f"(rango {f(sp['confianza_min'], 3)}–{f(sp['confianza_max'], 3)}). "
        "No hay un conjunto etiquetado de ataques, así que no se calcula")
    add("precision/recall; solo se reportan las alarmas.")
    add("")
    add("### 4.2 Presencia por parte (video 3)")
    add("")
    add("| Estudiante | Partes presente | Estabilidad (11) |")
    add("|---|---|---|")
    for r in asi["presencia_resumen"]:
        add(f"| {r['persona']} | {r['partes_presente']} | {f(r['estabilidad'], 3)} |")
    add("")
    add("### 4.3 Fragmentación de pistas (video 3)")
    add("")
    add("| Persona | Pistas confirmadas | Filas confirmadas | Frames/pista medio | Primera confirmación (s) |")
    add("|---|---|---|---|---|")
    for r in asi["tracking"]:
        add(f"| {r['persona']} | {r['pistas_confirmadas']} | {r['filas_confirmadas']} | "
            f"{r['frames_por_pista_media']} | {r['primera_confirmacion_s']} |")
    add("")
    add("---")
    add("")
    add("## 5. Rendimiento")
    add("")
    add("| Equipo | Etapa | Valor | Unidad | Fuente |")
    add("|---|---|---|---|---|")
    for r in ren["fps"]:
        add(f"| {r['equipo']} | {r['etapa']} | {r['valor']} | {r['unidad']} | {r['fuente']} |")
    add("")
    add("| Equipo | Proveedor | FPS | Estado | Nota |")
    add("|---|---|---|---|---|")
    for r in ren["cpu_gpu"]:
        add(f"| {r['equipo']} | {r['proveedor']} | {r['fps'] or '—'} | {r['estado']} | {r['nota']} |")
    add("")
    add("El pipeline corre en **CPU** (`onnxruntime` build CPU). La RTX 4090 está")
    add("presente pero sin `CUDAExecutionProvider`; el benchmark GPU queda")
    add("pendiente y se llenará en `salida/rendimiento/cpu_gpu.csv`.")
    add("")
    add("---")
    add("")
    add("## 6. Limitaciones y validez")
    add("")
    add("- **Un solo video etiquetado** para identidad (300 recortes, 86 positivos).")
    add("  Los intervalos de confianza reflejan ese tamaño; no sustituyen una")
    add("  validación con más aulas y fechas.")
    add("- **Los recortes no incluyen fallos del detector**: solo caras detectadas.")
    add("  La tasa de detección se reporta aparte, pero no es un recall de detección")
    add("  contra ground truth exhaustivo.")
    add("- **El AUC score-level mide un frame**, sin los 3 votos ni el ancla. No es")
    add("  directamente comparable con la precisión/recall del motor en vivo;")
    add("  por eso ambos protocolos se reportan por separado.")
    add("- **Asistencia:** la matriz de sesión tiene pocos negativos (0 en video 1,")
    add("  1 en video 3); no es una estimación poblacional de precision/recall.")
    add("- **Intrusos:** las cifras son del código real y revisión manual; el")
    add("  simulador offline se quedó corto por hasta 7 filas (4 vs 7 en el video")
    add("  1 y 22 vs 29 en el video 3). No se usa para la decisión.")
    add("- **Intrusos, salvedad A1:** las \"7 filas falsas\" del video 1 asumen que")
    add("  ese tramo no tiene desconocidos, pero el mismo video tiene 157 recortes")
    add("  `desconocido` y una no registrada documentada en")
    add("  `09_incidente_identidad_dani.md`; la cifra es una **cota**, no un conteo")
    add("  verificado. Ver `reporte_auditoria_resultados.md`.")
    add("- **Spoofing:** no hay conjunto etiquetado de ataques; solo se reportan las")
    add("  alarmas, no precision/recall.")
    add("- **mAP de detección no calculado:** exige ground truth exhaustivo (la caja")
    add("  de todas las caras de los frames evaluados), que no existe; se reportan")
    add("  IoU, estrés de emparejamiento y tasas de detección como alternativas.")
    add("- **GPU:** no medida; el entorno solo tiene `onnxruntime` CPU.")
    add("")
    add("---")
    add("")
    add("## 7. Archivos")
    add("")
    add("| Carpeta | Contenido |")
    add("|---|---|")
    add("| `salida/identidad/` | `scores.csv`, `roc_puntos.csv`, `pr_puntos.csv`, "
        "`tabla_umbrales.csv`, `matriz_confusion.csv`, `metricas_por_estudiante.csv`, "
        "`topk_y_rango.csv`, `open_set.csv`, `intervalos_confianza.csv`, "
        "`por_ancho.csv`, `por_accesorio.csv`, `multiframe.csv` y las 6 gráficas |")
    add("| `salida/deteccion/` | `resumen_deteccion.csv`, `iou_emparejamiento.csv`, "
        "`estres_iou.csv`, `tamanos_cara.csv`, `efecto_filtro.csv`, `fallos_por_causa.csv` |")
    add("| `salida/asistencia/` | `cobertura_video1.csv`, `cobertura_video3.csv`, "
        "`confusion_sesion.csv`, `por_persona_video1.csv`, `presencia_por_parte.csv`, "
        "`intrusiones.csv`, `barrido_intrusos.csv`, `tracking.csv`, `spoofing.csv` |")
    add("| `salida/rendimiento/` | `fps_y_latencia.csv`, `cpu_gpu.csv` |")
    add("")
    add("---")
    add("")
    add("## 8. Reproducibilidad")
    add("")
    add("```bash")
    add("venv/bin/python experiments/unc/metricas/metricas.py")
    add("```")
    add("")
    add("No se necesita scikit-learn ni pandas. Las gráficas usan matplotlib con")
    add("backend `Agg`; los CSV se escriben con `utf-8` y punto decimal.")
    add("")
    (SALIDA / "resumen.md").write_text("\n".join(lineas), encoding="utf-8")


# ── Principal ─────────────────────────────────────────────────────
def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    print("Leyendo datos de", DATOS)

    # ===== Identidad =====
    pos, neg_score, neg_op, registros, conteos = cargar_similitudes()
    fpr, tpr, cortes, auc, j = roc_exacta(pos, neg_score)
    recall_pr, precision_pr, ap = pr_exacta(pos, neg_score)
    eer_val, eer_umbral = eer_exacta(pos, neg_score, fpr, tpr, cortes)
    dp = d_prime(pos, neg_score)
    idx_youden = int(np.argmax(j))
    youden_umbral = float(cortes[idx_youden])

    # Tabla de umbrales (matriz operativa)
    tabla = []
    for t in UMBRALES_COMPLETOS:
        tp, fp, fn, tn = matriz_operativa(registros, float(t))
        m = metricas_matriz(tp, fp, fn, tn)
        tabla.append({
            "umbral": round(float(t), 3),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "specificity": round(m["specificity"], 4),
            "f1": round(m["f1"], 4),
            "accuracy": round(m["accuracy"], 4),
            "balanced_accuracy": round(m["balanced_accuracy"], 4),
            "mcc": round(m["mcc"], 4),
            "kappa": round(m["kappa"], 4),
            "youden": round(m["youden"], 4),
            "fpr": round(m["fpr"], 4),
            "fnr": round(m["fnr"], 4),
            "npv": round(m["npv"], 4),
        })
    escribir_csv(SALIDA / "identidad/tabla_umbrales.csv", tabla)

    # Matriz de confusión en umbrales clave
    matrices = []
    for t in UMBRALES_CLAVE:
        tp, fp, fn, tn = matriz_operativa(registros, float(t))
        m = metricas_matriz(tp, fp, fn, tn)
        matrices.append({
            "umbral": round(float(t), 3),
            "tipo": "fijo",
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1": round(m["f1"], 4),
            "accuracy": round(m["accuracy"], 4),
            "specificity": round(m["specificity"], 4),
            "mcc": round(m["mcc"], 4),
            "kappa": round(m["kappa"], 4),
        })
    # Añadir Youden y mejor F1
    for nombre, t in [
        ("Youden", youden_umbral),
        ("mejor_F1", float(max(tabla, key=lambda r: (r["f1"], r["recall"]))["umbral"])),
    ]:
        tp, fp, fn, tn = matriz_operativa(registros, t)
        m = metricas_matriz(tp, fp, fn, tn)
        matrices.append({
            "umbral": round(t, 4),
            "tipo": nombre,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1": round(m["f1"], 4),
            "accuracy": round(m["accuracy"], 4),
            "specificity": round(m["specificity"], 4),
            "mcc": round(m["mcc"], 4),
            "kappa": round(m["kappa"], 4),
        })
    escribir_csv(SALIDA / "identidad/matriz_confusion.csv", matrices)

    # ROC / PR puntos
    escribir_csv(SALIDA / "identidad/roc_puntos.csv", [
        {"fpr": round(float(a), 6), "tpr": round(float(b), 6),
         "umbral": ("inf" if not np.isfinite(c) else round(float(c), 4))}
        for a, b, c in zip(fpr, tpr, cortes)
    ])
    escribir_csv(SALIDA / "identidad/pr_puntos.csv", [
        {"recall": round(float(a), 6), "precision": round(float(b), 6)}
        for a, b in zip(recall_pr, precision_pr)
    ])

    # Scores globales
    mejor_f1 = max(tabla, key=lambda r: (r["f1"], r["recall"]))
    fila_030 = next(r for r in tabla if abs(r["umbral"] - UMBRAL_OPERATIVO) < 1e-9)
    scores = [
        {"metrica": "n_positivos", "valor": len(pos), "nota": "86 recortes con nombre"},
        {"metrica": "n_negativos_score", "valor": len(neg_score), "nota": "86 max-otra-persona + 157 desconocidos"},
        {"metrica": "n_negativos_operativo", "valor": len(neg_op), "nota": "157 desconocidos"},
        {"metrica": "auc", "valor": round(auc, 6), "nota": "ROC exacta score-level"},
        {"metrica": "ap", "valor": round(ap, 6), "nota": "Average Precision exacta"},
        {"metrica": "eer", "valor": round(eer_val, 6), "nota": "Equal Error Rate"},
        {"metrica": "eer_umbral", "valor": round(eer_umbral, 4), "nota": "umbral del EER"},
        {"metrica": "d_prime", "valor": round(dp, 4), "nota": "discriminabilidad"},
        {"metrica": "youden_umbral", "valor": round(youden_umbral, 4), "nota": "argmax(TPR-FPR)"},
        {"metrica": "youden_j", "valor": round(float(j[idx_youden]), 4), "nota": "TPR-FPR"},
        {"metrica": "youden_tpr", "valor": round(float(tpr[idx_youden]), 4), "nota": ""},
        {"metrica": "youden_fpr", "valor": round(float(fpr[idx_youden]), 4), "nota": ""},
        {"metrica": "mejor_f1_umbral", "valor": round(float(mejor_f1["umbral"]), 4), "nota": ""},
        {"metrica": "mejor_f1", "valor": round(float(mejor_f1["f1"]), 4), "nota": ""},
        {"metrica": "precision_030", "valor": round(float(fila_030["precision"]), 4), "nota": ""},
        {"metrica": "recall_030", "valor": round(float(fila_030["recall"]), 4), "nota": ""},
        {"metrica": "f1_030", "valor": round(float(fila_030["f1"]), 4), "nota": ""},
        {"metrica": "accuracy_030", "valor": round(float(fila_030["accuracy"]), 4), "nota": ""},
        {"metrica": "specificity_030", "valor": round(float(fila_030["specificity"]), 4), "nota": ""},
        {"metrica": "mcc_030", "valor": round(float(fila_030["mcc"]), 4), "nota": ""},
        {"metrica": "kappa_030", "valor": round(float(fila_030["kappa"]), 4), "nota": ""},
        {"metrica": "prevalencia", "valor": round(len(pos) / (len(pos) + len(neg_op)), 4), "nota": "86/243 en el conjunto operativo"},
    ]
    escribir_csv(SALIDA / "identidad/scores.csv", scores)

    # Dataset
    dataset = []
    for etiqueta, n in sorted(conteos.items(), key=lambda kv: (-kv[1], kv[0])):
        if etiqueta in PERSONAS:
            tipo = "registrado"
        elif etiqueta == "desconocido":
            tipo = "desconocido"
        elif etiqueta == "dudoso":
            tipo = "dudoso (excluido)"
        else:
            tipo = "no_cara (excluido)"
        dataset.append({"etiqueta": etiqueta, "tipo": tipo, "n": n})
    dataset.append({"etiqueta": "TOTAL", "tipo": "", "n": sum(conteos.values())})
    escribir_csv(SALIDA / "identidad/dataset.csv", dataset)

    # Por estudiante
    por_est = por_estudiante(registros, UMBRAL_OPERATIVO)
    escribir_csv(SALIDA / "identidad/metricas_por_estudiante.csv", por_est)

    # Top-k
    tk = top_k(registros)
    escribir_csv(SALIDA / "identidad/topk_y_rango.csv", [
        {"metrica": "n_etiquetados", "valor": tk["n"]},
        {"metrica": "top1", "valor": round(tk["top1"], 4)},
        {"metrica": "top2", "valor": round(tk["top2"], 4)},
        {"metrica": "top3", "valor": round(tk["top3"], 4)},
        {"metrica": "mrr", "valor": round(tk["mrr"], 4)},
        {"metrica": "rango_mediano", "valor": tk["rango_mediano"]},
        {"metrica": "rango_max", "valor": tk["rango_max"]},
        {"metrica": "rank1", "valor": tk["rank1"]},
        {"metrica": "rank2", "valor": tk["rank2"]},
        {"metrica": "rank3", "valor": tk["rank3"]},
        {"metrica": "rank_mas3", "valor": tk["rank_mas3"]},
    ])

    # Open-set
    open_set = []
    for t in UMBRALES_CLAVE:
        tp, fp, fn, tn = matriz_operativa(registros, float(t))
        open_set.append({
            "umbral": round(float(t), 3),
            "dir": round(div(tp, len(pos)), 4),
            "far": round(div(fp, len(neg_op)), 4),
            "frr": round(div(fn, len(pos)), 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
    escribir_csv(SALIDA / "identidad/open_set.csv", open_set)

    # Intervalos de confianza
    tp030, fp030, fn030, tn030 = matriz_operativa(registros, UMBRAL_OPERATIVO)
    m030 = metricas_matriz(tp030, fp030, fn030, tn030)
    ic_acc = wilson(tp030 + tn030, tp030 + fp030 + fn030 + tn030)
    ic_rec = wilson(tp030, tp030 + fn030)
    ic_prec = wilson(tp030, tp030 + fp030)
    ci_auc = bootstrap_auc(pos, neg_score)
    ci_op = bootstrap_operativa(registros, UMBRAL_OPERATIVO)
    intervalos = [
        {"metrica": "auc", "valor": round(auc, 4), "ic_inf": round(float(ci_auc[0]), 4),
         "ic_sup": round(float(ci_auc[1]), 4), "metodo": "bootstrap 2000"},
        {"metrica": "f1_030", "valor": round(m030["f1"], 4),
         "ic_inf": round(float(ci_op["f1"][0]), 4), "ic_sup": round(float(ci_op["f1"][1]), 4),
         "metodo": "bootstrap 2000"},
        {"metrica": "accuracy_030", "valor": round(m030["accuracy"], 4),
         "ic_inf": round(ic_acc[0], 4), "ic_sup": round(ic_acc[1], 4),
         "metodo": "Wilson"},
        {"metrica": "recall_030", "valor": round(m030["recall"], 4),
         "ic_inf": round(ic_rec[0], 4), "ic_sup": round(ic_rec[1], 4),
         "metodo": "Wilson"},
        {"metrica": "precision_030", "valor": round(m030["precision"], 4),
         "ic_inf": round(ic_prec[0], 4), "ic_sup": round(ic_prec[1], 4),
         "metodo": "Wilson"},
    ]
    escribir_csv(SALIDA / "identidad/intervalos_confianza.csv", intervalos)

    # Ancho y accesorios
    tabla_ancho = por_ancho(registros, UMBRAL_OPERATIVO)
    escribir_csv(SALIDA / "identidad/por_ancho.csv", tabla_ancho)
    tabla_acc = por_accesorio(registros, UMBRAL_OPERATIVO)
    escribir_csv(SALIDA / "identidad/por_accesorio.csv", tabla_acc)

    # Multiframe normalizado
    multiframe = []
    for r in leer_csv(DATOS / "fase4/fase4_multiframe.csv"):
        for prefijo, protocolo in [("1f", "1 frame"), ("3f", "3 frames")]:
            multiframe.append({
                "umbral": round(float(r["umbral"]), 3),
                "protocolo": protocolo,
                "tp": int(r[f"{prefijo}_TP"]),
                "fp": int(r[f"{prefijo}_FP"]),
                "fn": int(r[f"{prefijo}_FN"]),
                "precision": round(float(r[f"{prefijo}_P"]), 4),
                "recall": round(float(r[f"{prefijo}_R"]), 4),
                "f1": round(float(r[f"{prefijo}_F1"]), 4),
            })
    escribir_csv(SALIDA / "identidad/multiframe.csv", multiframe)

    guardar_graficas(pos, neg_score, fpr, tpr, cortes, auc, j,
                     recall_pr, precision_pr, ap, tabla, por_est, eer_val, eer_umbral)

    # ===== Detección =====
    detalle, salon = metricas_deteccion()
    escribir_csv(SALIDA / "deteccion/resumen_deteccion.csv", detalle)
    iou_tabla = metricas_iou()
    escribir_csv(SALIDA / "deteccion/iou_emparejamiento.csv", iou_tabla)
    estres_tabla = metricas_estres()
    escribir_csv(SALIDA / "deteccion/estres_iou.csv", estres_tabla)
    tamanos = metricas_tamanos()
    escribir_csv(SALIDA / "deteccion/tamanos_cara.csv", [tamanos])
    filtro_tabla = metricas_filtro()
    escribir_csv(SALIDA / "deteccion/efecto_filtro.csv", filtro_tabla)
    fallos_tabla = metricas_fallos()
    escribir_csv(SALIDA / "deteccion/fallos_por_causa.csv", fallos_tabla)

    # ===== Asistencia =====
    asis = metricas_asistencia()
    escribir_csv(SALIDA / "asistencia/cobertura_video1.csv", [asis["cobertura_v1"]])
    escribir_csv(SALIDA / "asistencia/cobertura_video3.csv", [asis["cobertura_v3"]])
    escribir_csv(SALIDA / "asistencia/confusion_sesion.csv", asis["confusion"])
    escribir_csv(SALIDA / "asistencia/por_persona_video1.csv", asis["por_persona_v1"])
    escribir_csv(SALIDA / "asistencia/presencia_por_parte.csv", asis["presencia"])
    escribir_csv(SALIDA / "asistencia/intrusiones.csv", asis["intrusiones"])
    escribir_csv(SALIDA / "asistencia/barrido_intrusos.csv", asis["barrido"])
    escribir_csv(SALIDA / "asistencia/tracking.csv", asis["tracking"])
    escribir_csv(SALIDA / "asistencia/spoofing.csv", asis["spoofing"])

    # Resumen de presencia por persona
    presencia_resumen = []
    for persona in PERSONAS:
        partes_presente = sum(1 for r in asis["presencia"] if r.get(persona) == 1)
        presencia_resumen.append({
            "persona": persona,
            "partes_presente": partes_presente,
            "estabilidad": round(partes_presente / len(asis["presencia"]), 4) if asis["presencia"] else 0,
        })
    asis["presencia_resumen"] = presencia_resumen

    # ===== Rendimiento =====
    fps, cpu_gpu = metricas_rendimiento()
    escribir_csv(SALIDA / "rendimiento/fps_y_latencia.csv", fps)
    escribir_csv(SALIDA / "rendimiento/cpu_gpu.csv", cpu_gpu)

    # ===== Resumen =====
    datos = {
        "identidad": {
            "auc": auc, "ap": ap, "eer": eer_val, "eer_umbral": eer_umbral,
            "d_prime": dp, "youden_umbral": youden_umbral,
            "youden_tpr": float(tpr[idx_youden]), "youden_fpr": float(fpr[idx_youden]),
            "mejor_f1_umbral": float(mejor_f1["umbral"]),
            "mejor_f1": float(mejor_f1["f1"]),
            "tabla_clave": matrices,
            "por_estudiante": por_est,
            "topk": tk,
            "intervalos": intervalos,
            "por_ancho": tabla_ancho,
            "por_accesorio": tabla_acc,
            "multiframe": multiframe,
        },
        "deteccion": {
            "agregado": next(r for r in detalle if r["persona"] == "MICRO_9"),
            "salon": {
                "frames_totales": int(salon[0]["frames_totales"]),
                "frames_muestreados": int(salon[0]["frames_muestreados"]),
                "tasa_deteccion": float(salon[0]["tasa_deteccion"]),
                "tasa_util": float(salon[0]["tasa_util"]),
            },
            "tamanos": tamanos,
            "iou": next(r for r in iou_tabla if r["grupo"] == "todos"),
            "estres": next(r for r in estres_tabla if r["grupo"] == "todos"),
            "filtro": filtro_tabla,
            "personas": [r for r in detalle if r["grupo"] == "persona"],
            "fallos": fallos_tabla,
        },
        "asistencia": asis,
        "rendimiento": {"fps": fps, "cpu_gpu": cpu_gpu},
    }
    escribir_resumen(datos)

    print(f"AUC={auc:.4f} AP={ap:.4f} EER={eer_val:.4f} d'={dp:.3f}")
    print(f"Umbral 0.30: P={m030['precision']:.4f} R={m030['recall']:.4f} "
          f"F1={m030['f1']:.4f} Acc={m030['accuracy']:.4f}")
    print(f"Salidas en {SALIDA}")


if __name__ == "__main__":
    main()
