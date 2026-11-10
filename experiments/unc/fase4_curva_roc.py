"""
Fase 4 — Curva ROC, AUC, Youden y métricas operativas por umbral.

Entrada: experiments/unc/fase4/fase4_similitudes.csv (similitud de cada recorte
etiquetado contra los 9 centroides).

Dos lecturas:
  1. Score: positivos = similitud con la persona etiquetada; negativos = máxima
     similitud con cualquier otra persona (o con las 9, si la etiqueta es desconocido).
     Con eso se calculan ROC, AUC y el índice de Youden.
  2. Operativa: se simula la decisión del motor (argmax + umbral). Se reportan
     precisión, recall y F1 por umbral, y se elige el mejor F1.

Salidas: experiments/unc/fase4/fase4_curva_roc.png, fase4_histograma.png,
         fase4_f1_vs_umbral.png, fase4_curva_pr.png, fase4_tabla_umbrales.csv
"""

import csv
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parents[2]
SALIDA = RAIZ / "experiments/unc/fase4"
ENTRADA = SALIDA / "fase4_similitudes.csv"
PERSONAS = ["alexa", "bob", "carlos", "dani", "fran", "jesus", "jose", "kate", "kevin"]
UMBRALES = np.round(np.arange(0.10, 0.71, 0.005), 3)


def cargar():
    with open(ENTRADA, encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    pos, neg, registros = [], [], []
    for fila in filas:
        etiqueta = fila["etiqueta"]
        sims = np.array([float(fila[f"sim_{p}"]) for p in PERSONAS])
        if etiqueta in PERSONAS:
            i = PERSONAS.index(etiqueta)
            pos.append(sims[i])
            neg.append(float(np.max(np.delete(sims, i))))
            registros.append(("positivo", sims, i))
        elif etiqueta == "desconocido":
            neg.append(float(np.max(sims)))
            registros.append(("negativo", sims, None))
        # dudoso y no_cara se excluyen
    return np.array(pos), np.array(neg), registros


def roc(pos, neg):
    """ROC exacta a partir de los scores ordenados.

    La versión anterior evaluaba solo umbrales 0.10-0.705; al no incluir los
    extremos, el área quedaba truncada (0.744). Esta versión usa un punto por
    cada score único, desde +inf hasta el mínimo, e incluye el origen (0,0).
    """
    scores = np.concatenate([pos, neg])
    etiquetas = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    orden = np.argsort(-scores)
    scores = scores[orden]
    etiquetas = etiquetas[orden]

    cortes = np.concatenate([[np.inf], scores])
    tp = np.concatenate([[0], np.cumsum(etiquetas)])
    fp = np.concatenate([[0], np.cumsum(1 - etiquetas)])
    tpr = tp / len(pos)
    fpr = fp / len(neg)

    # tpr y fpr ascienden al bajar el umbral; la regla trapezoidal es directa.
    auc = float(np.trapezoid(tpr, fpr))
    j = tpr - fpr
    return tpr, fpr, auc, j, cortes


def pr_curva(pos, neg):
    """Curva Precision-Recall exacta (un punto por score único)."""
    scores = np.concatenate([pos, neg])
    etiquetas = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    orden = np.argsort(-scores)
    etiquetas = etiquetas[orden]
    tp = np.cumsum(etiquetas)
    fp = np.cumsum(1 - etiquetas)
    precision = tp / (tp + fp)
    recall = tp / len(pos)
    return recall, precision


def pr_auc(recall, precision):
    """Average Precision (AP) exacta, estilo sklearn, sin interpolar."""
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def eer_desde_roc(fpr, tpr, cortes, pos, neg):
    """Equal Error Rate: cruce FPR = FNR interpolado sobre la ROC exacta."""
    fnr = 1 - tpr
    diferencia = fpr - fnr
    cruces = np.where(np.sign(diferencia[:-1]) != np.sign(diferencia[1:]))[0]
    if not len(cruces):
        return 0.0, 0.0
    i = int(cruces[0])
    alpha = diferencia[i] / (diferencia[i] - diferencia[i + 1]) if diferencia[i] != diferencia[i + 1] else 0.0
    t = cortes[i] + alpha * (cortes[i + 1] - cortes[i])
    if not np.isfinite(t):
        return 0.0, 0.0
    return float(((neg >= t).mean() + (pos < t).mean()) / 2), float(t)


def d_prime(pos, neg):
    """Discriminabilidad d' (medias separadas por la desviación combinada)."""
    varianza = 0.5 * (pos.var(ddof=1) + neg.var(ddof=1))
    return float((pos.mean() - neg.mean()) / np.sqrt(varianza))


def operativa(registros, t):
    tp = fp = fn = tn = 0
    for tipo, sims, i in registros:
        mejor = int(np.argmax(sims))
        if tipo == "positivo":
            if mejor == i and sims[i] >= t:
                tp += 1
            else:
                fn += 1
        else:
            if sims[mejor] >= t:
                fp += 1
            else:
                tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return tp, fp, fn, tn, precision, recall, f1


def main():
    pos, neg, registros = cargar()
    print(f"Positivos: {len(pos)} | Negativos: {len(neg)}")

    tpr, fpr, auc, j, cortes = roc(pos, neg)
    idx_youden = int(np.argmax(j))
    t_youden = float(cortes[idx_youden])
    recall_pr, precision_pr = pr_curva(pos, neg)
    ap = pr_auc(recall_pr, precision_pr)
    eer_val, t_eer = eer_desde_roc(fpr, tpr, cortes, pos, neg)
    dp = d_prime(pos, neg)
    print(
        f"AUC: {auc:.4f} | AP: {ap:.4f} | EER: {eer_val:.4f} (umbral {t_eer:.3f}) "
        f"| d': {dp:.3f}"
    )
    print(f"Youden: umbral {t_youden:.3f} (TPR {tpr[idx_youden]:.3f}, FPR {fpr[idx_youden]:.3f})")

    # Operativas
    ops = {float(t): operativa(registros, float(t)) for t in UMBRALES}
    t_mejor_f1 = max(ops, key=lambda t: (ops[t][6], ops[t][5]))
    print(f"Mejor F1 operativo: umbral {t_mejor_f1:.3f} -> "
          f"P={ops[t_mejor_f1][4]:.3f} R={ops[t_mejor_f1][5]:.3f} F1={ops[t_mejor_f1][6]:.3f}")

    with open(SALIDA / "fase4_tabla_umbrales.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["umbral", "TP", "FP", "FN", "TN", "precision", "recall", "F1"])
        for t in UMBRALES:
            tp, fp, fn, tn, p, r, f1 = ops[float(t)]
            writer.writerow([f"{t:.3f}", tp, fp, fn, tn, f"{p:.3f}", f"{r:.3f}", f"{f1:.3f}"])

    print("\nUmbral   TP  FP  FN   TN   Prec   Recall  F1")
    for t in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, round(t_mejor_f1, 3)]:
        tp, fp, fn, tn, p, r, f1 = operativa(registros, t)
        print(f"{t:.3f}   {tp:3d} {fp:3d} {fn:3d} {tn:4d}   {p:.3f}  {r:.3f}  {f1:.3f}")

    # Histograma bimodal
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(neg, bins=np.arange(0, 0.8, 0.02), alpha=0.7, label="negativos (otra persona / desconocido)", color="tab:red")
    ax.hist(pos, bins=np.arange(0, 0.8, 0.02), alpha=0.7, label="positivos (su propia persona)", color="tab:blue")
    ax.axvline(0.35, color="black", linestyle="--", label="umbral actual 0.35")
    ax.axvline(t_mejor_f1, color="green", linestyle="--", label=f"mejor F1 {t_mejor_f1:.2f}")
    ax.set_xlabel("Similitud coseno")
    ax.set_ylabel("Número de recortes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SALIDA / "fase4_histograma.png", dpi=120)

    # ROC
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.scatter([fpr[idx_youden]], [tpr[idx_youden]], color="red", label=f"Youden t={t_youden:.2f}")
    tpr_eer = (pos >= t_eer).mean()
    fpr_eer = (neg >= t_eer).mean()
    ax.scatter([fpr_eer], [tpr_eer], color="orange", marker="s", label=f"EER t={t_eer:.2f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SALIDA / "fase4_curva_roc.png", dpi=120)

    # F1 vs umbral
    f1s = [ops[float(t)][6] for t in UMBRALES]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(UMBRALES, f1s)
    ax.axvline(0.35, color="black", linestyle="--", label="actual 0.35")
    ax.axvline(t_mejor_f1, color="green", linestyle="--", label=f"mejor {t_mejor_f1:.2f}")
    ax.set_xlabel("Umbral")
    ax.set_ylabel("F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SALIDA / "fase4_f1_vs_umbral.png", dpi=120)

    # Precision-Recall score-level
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall_pr, precision_pr, label=f"AP = {ap:.4f}")
    azar = len(pos) / (len(pos) + len(neg))
    ax.axhline(azar, color="gray", linestyle="--", label=f"azar = {azar:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precisión")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SALIDA / "fase4_curva_pr.png", dpi=120)

    print(f"\nSalidas en {SALIDA}")


if __name__ == "__main__":
    main()
